"""
Inference pipeline with dual scoring: Classifier + FAISS.

Usage:
  # Single file inference
  python inference.py \
    --model_path ./saved_models/lora_faiss/checkpoint-best-acc/model.bin \
    --faiss_dir ./saved_models/lora_faiss/faiss_index \
    --input_file ../dataset/test.jsonl \
    --output_file ./predictions_hybrid.json \
    --beta 0.6 --top_k 5

  # Or use from Python:
    from inference import HybridPredictor
    predictor = HybridPredictor(model_path, faiss_dir, args)
    result = predictor.predict(code_string)
    # result = {
    #     "cls_prob": 0.73,
    #     "faiss_score": 0.8,
    #     "hybrid_score": 0.758,
    #     "prediction": 1,
    #     "neighbors": [...]
    # }
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from model import GNNReGVD
from faiss_index import FAISSIndexManager
from run import TextDataset, set_seed

logger = logging.getLogger(__name__)


class HybridPredictor:
    """
    Dual-mode inference: classifier probability + FAISS nearest-neighbor score.

    Final score = beta * cls_prob + (1 - beta) * faiss_score

    Where:
      - cls_prob: binary classifier output P(vulnerable)
      - faiss_score: fraction of top-K neighbors that are vulnerable
      - beta: weighting factor (0.6 by default, tunable on validation set)
    """

    def __init__(self, model, tokenizer, faiss_manager, args, beta=0.6, top_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.faiss_manager = faiss_manager
        self.args = args
        self.beta = beta
        self.top_k = top_k
        self.device = args.device

        self.model.eval()
        self.model.to(self.device)

    def predict_single(self, code_string):
        """
        Predict vulnerability for a single code snippet.

        Args:
            code_string: raw source code (string)

        Returns:
            dict with cls_prob, faiss_score, hybrid_score, prediction, neighbors
        """
        # Tokenize
        code = ' '.join(code_string.split())
        code_tokens = self.tokenizer.tokenize(code)[:self.args.block_size - 2]
        source_tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.args.block_size - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id] * padding_length

        input_ids = torch.tensor([source_ids]).to(self.device)

        with torch.no_grad():
            output = self.model(input_ids)
            if isinstance(output, tuple):
                prob = output[0]
                embeddings = output[1] if len(output) > 1 else None
            else:
                prob = output
                embeddings = None

        cls_prob = float(prob[0, 0].cpu())

        # FAISS score
        faiss_score = 0.5  # neutral default
        neighbors = []
        if embeddings is not None and self.faiss_manager is not None and self.faiss_manager.size > 0:
            embedding_np = embeddings[0].cpu().numpy()
            faiss_score = self.faiss_manager.compute_faiss_score(embedding_np, top_k=self.top_k)
            neighbors = self.faiss_manager.search(embedding_np, top_k=self.top_k)

        # Hybrid score
        hybrid_score = self.beta * cls_prob + (1 - self.beta) * faiss_score
        prediction = 1 if hybrid_score > 0.5 else 0

        return {
            "cls_prob": round(cls_prob, 4),
            "faiss_score": round(faiss_score, 4),
            "hybrid_score": round(hybrid_score, 4),
            "prediction": prediction,
            "neighbors": neighbors,
        }

    def predict_batch(self, dataset):
        """
        Run hybrid inference on a TextDataset.

        Returns:
            list of result dicts, overall accuracy (if labels available)
        """
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset),
                                batch_size=self.args.eval_batch_size)

        all_results = []
        all_labels = []
        all_hybrid_preds = []
        all_cls_preds = []

        for batch in dataloader:
            inputs = batch[0].to(self.device)
            labels = batch[1].numpy()

            with torch.no_grad():
                output = self.model(inputs)
                if isinstance(output, tuple):
                    probs = output[0]
                    embeddings = output[1] if len(output) > 1 else None
                else:
                    probs = output
                    embeddings = None

            for i in range(len(labels)):
                cls_prob = float(probs[i, 0].cpu())

                faiss_score = 0.5
                if embeddings is not None and self.faiss_manager is not None and self.faiss_manager.size > 0:
                    emb = embeddings[i].cpu().numpy()
                    faiss_score = self.faiss_manager.compute_faiss_score(emb, top_k=self.top_k)

                hybrid_score = self.beta * cls_prob + (1 - self.beta) * faiss_score
                hybrid_pred = 1 if hybrid_score > 0.5 else 0
                cls_pred = 1 if cls_prob > 0.5 else 0

                all_results.append({
                    "cls_prob": round(cls_prob, 4),
                    "faiss_score": round(faiss_score, 4),
                    "hybrid_score": round(hybrid_score, 4),
                    "prediction": hybrid_pred,
                    "label": int(labels[i]),
                })
                all_labels.append(int(labels[i]))
                all_hybrid_preds.append(hybrid_pred)
                all_cls_preds.append(cls_pred)

        # Compute accuracies
        all_labels = np.array(all_labels)
        cls_acc = np.mean(all_labels == np.array(all_cls_preds))
        hybrid_acc = np.mean(all_labels == np.array(all_hybrid_preds))

        summary = {
            "num_samples": len(all_results),
            "cls_only_acc": round(cls_acc, 4),
            "hybrid_acc": round(hybrid_acc, 4),
            "beta": self.beta,
            "top_k": self.top_k,
            "faiss_index_size": self.faiss_manager.size if self.faiss_manager else 0,
        }

        return all_results, summary


def find_optimal_beta(predictor, dataset, betas=None):
    """
    Grid search for optimal beta on a validation set.

    Args:
        predictor: HybridPredictor
        dataset: validation TextDataset
        betas: list of beta values to try

    Returns:
        best_beta, results_per_beta
    """
    if betas is None:
        betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {}
    for beta in betas:
        predictor.beta = beta
        _, summary = predictor.predict_batch(dataset)
        results[beta] = summary['hybrid_acc']
        logger.info(f"  beta={beta:.1f} -> hybrid_acc={summary['hybrid_acc']}")

    best_beta = max(results, key=results.get)
    logger.info(f"Best beta: {best_beta} (acc={results[best_beta]})")

    return best_beta, results


def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--faiss_dir", type=str, default="",
                        help="Path to FAISS index (omit for classifier-only mode)")
    parser.add_argument("--input_file", type=str, required=True,
                        help="JSONL file to run inference on")
    parser.add_argument("--output_file", type=str, default="./predictions_hybrid.json")

    # Inference parameters
    parser.add_argument("--beta", type=float, default=0.6,
                        help="Weight for classifier vs FAISS (1.0=classifier only)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of FAISS neighbors for scoring")
    parser.add_argument("--find_beta", action='store_true',
                        help="Grid search for optimal beta (requires labeled data)")

    # Model architecture (must match training)
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str)
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str)
    parser.add_argument("--block_size", default=400, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--feature_dim_size", default=768, type=int)
    parser.add_argument("--num_GNN_layers", default=2, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--gnn", default="ReGCN", type=str)
    parser.add_argument("--format", default="uni", type=str)
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--remove_residual", default=False, action='store_true')
    parser.add_argument("--att_op", default='mul', type=str)

    # LoRA / FAISS config (must match training)
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--use_faiss", action='store_true')
    parser.add_argument("--embed_dim", type=int, default=512)

    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(args.n_gpu, 1)
    args.local_rank = -1

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    set_seed(args.seed)

    # Load model
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 1
    encoder = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    model = GNNReGVD(encoder, config, tokenizer, args)
    logger.info(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Load FAISS index
    faiss_manager = None
    if args.faiss_dir and os.path.exists(args.faiss_dir):
        faiss_manager = FAISSIndexManager.load(args.faiss_dir)
        logger.info(f"Loaded FAISS index: {faiss_manager.size} vectors")
    else:
        logger.info("No FAISS index loaded. Running classifier-only mode.")
        args.beta = 1.0  # force classifier only

    # Create predictor
    predictor = HybridPredictor(model, tokenizer, faiss_manager, args,
                                 beta=args.beta, top_k=args.top_k)

    # Load dataset
    dataset = TextDataset(tokenizer, args, args.input_file)
    logger.info(f"Loaded {len(dataset)} samples from {args.input_file}")

    # Optionally find best beta
    if args.find_beta:
        best_beta, beta_results = find_optimal_beta(predictor, dataset)
        predictor.beta = best_beta
        logger.info(f"Using optimal beta={best_beta}")

    # Run inference
    # results, summary = predictor.predict_batch(dataset)
    summary = predictor.predict_single("void process_request_extra(const char* user, const char* cmd) { char logbuf[64] = \"CMD:\"; char tmp[32]; strcpy(tmp, user); strcat(logbuf, tmp); strcat(logbuf, \":\"); strcat(logbuf, cmd); if (logbuf[0] != '\\0') { printf(\"%s\\n\", logbuf); } }")

    # Print summary
    logger.info("=" * 50)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 50)
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    # Save results
    # output = {
    #     "summary": summary,
    #     "predictions": results,
    # }
    # with open(args.output_file, 'w') as f:
    #     json.dump(output, f, indent=2)
    # logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
