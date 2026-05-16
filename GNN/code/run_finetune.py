"""
Fine-tuning script for incremental updates on new data.

Two modes:
  Mode A — Hot Update (FAISS only, zero retraining):
    python run_finetune.py --mode hot_update \
      --model_path ./saved_models/lora_faiss/checkpoint-best-acc/model.bin \
      --new_data ../dataset/new_samples.jsonl \
      --faiss_dir ./saved_models/lora_faiss/faiss_index \
      --output_dir ./saved_models/lora_faiss

  Mode B — LoRA Fine-tune (periodic retraining + FAISS rebuild):
    python run_finetune.py --mode lora_finetune \
      --model_path ./saved_models/lora_faiss/checkpoint-best-acc/model.bin \
      --new_data ../dataset/new_samples.jsonl \
      --train_data_file ../dataset/train.jsonl \
      --eval_data_file ../dataset/valid.jsonl \
      --faiss_dir ./saved_models/lora_faiss/faiss_index \
      --output_dir ./saved_models/lora_faiss_v2 \
      --epoch 10 --learning_rate 1e-4

The new_data file format is the same JSONL:
  {"func": "...", "target": 0, "idx": 99999}
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from model import GNNReGVD
from faiss_index import FAISSIndexManager
from losses import SupervisedContrastiveLoss, TripletMarginLossWithMining
from run import TextDataset, set_seed, evaluate

logger = logging.getLogger(__name__)


def hot_update(args, model, tokenizer):
    """
    Mode A — Hot Update: zero retraining.

    1. Load existing FAISS index
    2. Run new data through the frozen model to get embeddings
    3. Add embeddings to FAISS index
    4. Save updated index

    The model is NOT modified. Only the FAISS index grows.
    """
    logger.info("=" * 50)
    logger.info("MODE A: Hot Update (FAISS only, zero retraining)")
    logger.info("=" * 50)

    # Load new data
    new_dataset = TextDataset(tokenizer, args, args.new_data)
    logger.info(f"New samples: {len(new_dataset)}")

    # Load existing FAISS index
    faiss_manager = FAISSIndexManager.load(args.faiss_dir)
    logger.info(f"Existing FAISS index: {faiss_manager.size} vectors")

    # Extract embeddings from new data (frozen model)
    model.eval()
    model.to(args.device)

    dataloader = DataLoader(new_dataset, sampler=SequentialSampler(new_dataset),
                            batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    new_embeddings = []
    new_metadata = []
    sample_counter = 0

    for batch in dataloader:
        inputs = batch[0].to(args.device)
        labels = batch[1].numpy()

        with torch.no_grad():
            embeddings = model.get_embedding(inputs)
            new_embeddings.append(embeddings.cpu().numpy())

        for i, label in enumerate(labels):
            new_metadata.append({
                "idx": new_dataset.examples[sample_counter].idx,
                "label": int(label),
                "func_snippet": new_dataset.get_func_snippet(sample_counter),
            })
            sample_counter += 1

    new_embeddings = np.concatenate(new_embeddings, axis=0).astype(np.float32)

    # Add to existing index
    faiss_manager.add(new_embeddings, new_metadata)

    # Save updated index
    faiss_manager.save(args.faiss_dir)
    logger.info(f"Hot update complete. FAISS index now has {faiss_manager.size} vectors.")


def lora_finetune(args, model, tokenizer):
    """
    Mode B — LoRA Fine-tune: periodic retraining.

    1. Load pre-trained model
    2. Fine-tune LoRA adapters + GNN + heads on new data
       (optionally mixed with original training data)
    3. Rebuild FAISS index from scratch with updated embeddings
    4. Save everything
    """
    logger.info("=" * 50)
    logger.info("MODE B: LoRA Fine-tune (periodic retraining)")
    logger.info("=" * 50)

    model.to(args.device)

    # ── Prepare combined dataset ──
    # New data is mandatory
    new_dataset = TextDataset(tokenizer, args, args.new_data)
    logger.info(f"New samples: {len(new_dataset)}")

    # Optionally mix with original training data (for replay / avoiding catastrophic forgetting)
    if args.train_data_file and os.path.exists(args.train_data_file) and args.replay_ratio > 0:
        old_dataset = TextDataset(tokenizer, args, args.train_data_file, sample_percent=args.replay_ratio)
        logger.info(f"Replay samples from original data: {len(old_dataset)} (ratio={args.replay_ratio})")

        # Combine datasets
        combined_examples = new_dataset.examples + old_dataset.examples
        random.shuffle(combined_examples)
        new_dataset.examples = combined_examples
        logger.info(f"Combined training set: {len(new_dataset)} samples")

    # ── Contrastive loss ──
    contrastive_loss_fn = None
    if args.use_faiss:
        if args.contrastive_loss == "triplet":
            contrastive_loss_fn = TripletMarginLossWithMining(margin=0.3)
        else:
            contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.07)

    # ── Optimizer: only trainable params (LoRA + GNN + heads) ──
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    train_dataloader = DataLoader(new_dataset, sampler=RandomSampler(new_dataset),
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    total_steps = args.epoch * len(train_dataloader)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1),
                                                num_training_steps=total_steps)

    # Print trainable info
    info = model.get_trainable_params_info()
    logger.info(f"Trainable: {info['trainable_params']:,} / {info['total_params']:,} ({info['trainable_percent']}%)")

    # ── Training loop ──
    best_acc = 0.0
    model.zero_grad()

    for epoch in range(args.epoch):
        model.train()
        tr_loss = 0.0
        tr_num = 0

        for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)

            output = model(inputs, labels)

            if args.use_faiss and len(output) == 3:
                cls_loss, logits, embeddings = output
                if embeddings is not None and contrastive_loss_fn is not None:
                    con_loss = contrastive_loss_fn(embeddings, labels)
                    alpha = 1.0 - args.contrastive_weight
                    loss = alpha * cls_loss + args.contrastive_weight * con_loss
                else:
                    loss = cls_loss
            else:
                loss = output[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            tr_loss += loss.item()
            tr_num += 1

        avg_loss = round(tr_loss / max(tr_num, 1), 5)
        logger.info(f"Finetune epoch {epoch}: loss={avg_loss}")

        # Evaluate if eval data available
        if args.eval_data_file and os.path.exists(args.eval_data_file):
            result = evaluate(args, model, tokenizer, eval_when_training=True)
            logger.info(f"  eval_acc={result['eval_acc']}, eval_loss={round(result['eval_loss'], 4)}")

            if result['eval_acc'] > best_acc:
                best_acc = result['eval_acc']
                # Save best model
                save_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))
                logger.info(f"  New best acc: {best_acc}. Saved to {save_dir}")

    # ── Rebuild FAISS index ──
    if args.use_faiss:
        logger.info("Rebuilding FAISS index with updated model...")

        # Load best model if we saved one
        best_path = os.path.join(args.output_dir, 'checkpoint-best-acc', 'model.bin')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            model.to(args.device)

        # Rebuild from ALL available data
        all_data_files = [args.new_data]
        if args.train_data_file and os.path.exists(args.train_data_file):
            all_data_files.append(args.train_data_file)

        all_embeddings = []
        all_metadata = []
        model.eval()

        for data_file in all_data_files:
            dataset = TextDataset(tokenizer, args, data_file)
            dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset),
                                    batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
            sample_counter = 0

            for batch in dataloader:
                inputs = batch[0].to(args.device)
                labels = batch[1].numpy()

                with torch.no_grad():
                    embeddings = model.get_embedding(inputs)
                    all_embeddings.append(embeddings.cpu().numpy())

                for i, label in enumerate(labels):
                    all_metadata.append({
                        "idx": dataset.examples[sample_counter].idx,
                        "label": int(label),
                        "func_snippet": dataset.get_func_snippet(sample_counter),
                    })
                    sample_counter += 1

        all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

        embed_dim = getattr(args, 'embed_dim', 512)
        index_type = "ivf" if len(all_embeddings) > 50000 else "flat"
        manager = FAISSIndexManager(embed_dim=embed_dim, index_type=index_type)
        manager.build_index(all_embeddings, all_metadata)

        faiss_save_dir = os.path.join(args.output_dir, "faiss_index")
        manager.save(faiss_save_dir)
        logger.info(f"FAISS index rebuilt: {manager.size} vectors saved to {faiss_save_dir}")

    logger.info("Fine-tuning complete.")


def main():
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument("--mode", type=str, required=True, choices=["hot_update", "lora_finetune"],
                        help="hot_update: FAISS only (Mode A). lora_finetune: retrain LoRA + rebuild (Mode B)")

    # Paths
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint (model.bin)")
    parser.add_argument("--new_data", type=str, required=True,
                        help="Path to new JSONL data file")
    parser.add_argument("--faiss_dir", type=str, default="",
                        help="Path to existing FAISS index directory (for hot_update)")
    parser.add_argument("--output_dir", type=str, default="./saved_models/finetuned")
    parser.add_argument("--train_data_file", type=str, default="",
                        help="Original training data (for replay in Mode B)")
    parser.add_argument("--eval_data_file", type=str, default="",
                        help="Validation data for evaluation during fine-tuning")

    # Model architecture (must match original training)
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

    # LoRA
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # FAISS
    parser.add_argument("--use_faiss", action='store_true')
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--contrastive_weight", type=float, default=0.3)
    parser.add_argument("--contrastive_loss", type=str, default="supcon", choices=["supcon", "triplet"])

    # Training (Mode B only)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Lower LR for fine-tuning (default=1e-4)")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--replay_ratio", type=float, default=0.2,
                        help="Fraction of original training data to mix in (anti-forgetting)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action='store_true')

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
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

    # Load encoder
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 1
    encoder = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # Build model with same architecture
    model = GNNReGVD(encoder, config, tokenizer, args)

    # Load trained weights
    logger.info(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Dispatch
    if args.mode == "hot_update":
        hot_update(args, model, tokenizer)
    else:
        lora_finetune(args, model, tokenizer)


if __name__ == "__main__":
    main()
