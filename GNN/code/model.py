# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Extended with LoRA adapters and FAISS embedding head
# for hybrid vulnerability detection pipeline.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
from modelGNN_updates import *
from utils import preprocess_features, preprocess_adj
from utils import *

def _get_device(tensor_or_module):
    """Infer device from a tensor or module instead of relying on a global variable."""
    if isinstance(tensor_or_module, torch.Tensor):
        return tensor_or_module.device
    if isinstance(tensor_or_module, nn.Module):
        try:
            return next(tensor_or_module.parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


# ─── LoRA layer ──────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation for a nn.Linear layer.
    Wraps an existing Linear and adds trainable low-rank matrices A, B.
    Output = original_linear(x) + x @ A @ B * scaling
    """

    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

        # Low-rank trainable matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


def apply_lora_to_model(encoder, rank=8, alpha=16, target_modules=None):
    """
    Apply LoRA adapters to GraphCodeBERT/RoBERTa encoder.

    Replaces query and value projection layers in each attention head
    with LoRA-wrapped versions.

    Args:
        encoder: RobertaForSequenceClassification or RobertaModel
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA scaling factor
        target_modules: list of substrings to match (default: query and value)

    Returns:
        encoder with LoRA applied, count of LoRA parameters
    """
    if target_modules is None:
        target_modules = ["query", "value"]

    lora_param_count = 0

    # First, freeze ALL encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False

    # Find and replace target linear layers with LoRA versions
    for name, module in encoder.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Navigate to parent module and replace
                parts = name.split(".")
                parent = encoder
                for part in parts[:-1]:
                    parent = getattr(parent, part)

                attr_name = parts[-1]
                original_linear = getattr(parent, attr_name)
                lora_layer = LoRALinear(original_linear, rank=rank, alpha=alpha)
                setattr(parent, attr_name, lora_layer)

                lora_param_count += rank * original_linear.in_features + rank * original_linear.out_features
                break

    return encoder, lora_param_count


# ─── Embedding Head (for FAISS) ─────────────────────────────────────────────

class EmbeddingHead(nn.Module):
    """
    Projects GNN output to a fixed-size L2-normalized embedding for FAISS.
    """

    def __init__(self, input_dim, embed_dim=512, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) — GNN graph-level output
        Returns:
            (batch_size, embed_dim) — L2-normalized embeddings
        """
        projected = self.projector(x.float())
        return F.normalize(projected, p=2, dim=1)


# ─── Classification Head (unchanged from original) ──────────────────────────

class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = args.hidden_size
        self.dense = nn.Linear(input_size, args.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# ─── Original Model wrapper (unchanged) ─────────────────────────────────────

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


# ─── GNNReGVD with LoRA + Embedding Head ────────────────────────────────────

class GNNReGVD(nn.Module):
    """
    Graph Neural Network for ReGVD — extended with:
      - LoRA adapters on GraphCodeBERT (optional, controlled by args.use_lora)
      - Embedding head for FAISS (optional, controlled by args.use_faiss)

    Forward returns:
      - With labels: (total_loss, cls_prob, embeddings)
      - Without labels: (cls_prob, embeddings)

    Where embeddings is None if use_faiss=False.
    """

    def __init__(self, encoder, config, tokenizer, args):
        super(GNNReGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # LoRA: apply to encoder if enabled
        self.use_lora = getattr(args, 'use_lora', False)
        self.lora_param_count = 0
        if self.use_lora:
            lora_rank = getattr(args, 'lora_rank', 8)
            lora_alpha = getattr(args, 'lora_alpha', 16)
            self.encoder, self.lora_param_count = apply_lora_to_model(
                self.encoder, rank=lora_rank, alpha=lora_alpha
            )

        # Word embeddings from encoder (for graph construction)
        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer

        # GNN
        if args.gnn == "ReGGNN":
            self.gnn = ReGGNN(
                feature_dim_size=args.feature_dim_size,
                hidden_size=args.hidden_size,
                num_GNN_layers=args.num_GNN_layers,
                dropout=config.hidden_dropout_prob,
                residual=not args.remove_residual,
                att_op=args.att_op
            )
        else:
            self.gnn = ReGCN(
                feature_dim_size=args.feature_dim_size,
                hidden_size=args.hidden_size,
                num_GNN_layers=args.num_GNN_layers,
                dropout=config.hidden_dropout_prob,
                residual=not args.remove_residual,
                att_op=args.att_op
            )

        gnn_out_dim = self.gnn.out_dim

        # Classification head (binary: vuln/safe)
        self.classifier = PredictionClassification(config, args, input_size=gnn_out_dim)

        # FAISS embedding head (optional)
        self.use_faiss = getattr(args, 'use_faiss', False)
        self.embedding_head = None
        if self.use_faiss:
            embed_dim = getattr(args, 'embed_dim', 512)
            self.embedding_head = EmbeddingHead(
                input_dim=gnn_out_dim,
                embed_dim=embed_dim,
                dropout=config.hidden_dropout_prob
            )

    def forward(self, input_ids=None, labels=None):
        # Infer device from input tensor (not from global variable)
        _dev = input_ids.device

        # ── Construct graph ──
        if self.args.format == "uni":
            adj, x_feature = build_graph(
                input_ids.cpu().detach().numpy(),
                self.w_embeddings,
                window_size=self.args.window_size
            )
        else:
            adj, x_feature = build_graph_text(
                input_ids.cpu().detach().numpy(),
                self.w_embeddings,
                window_size=self.args.window_size
            )

        # ── Preprocessing ──
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)

        # ── GNN forward ──
        gnn_output = self.gnn(
            adj_feature.to(_dev).float(),
            adj.to(_dev).float(),
            adj_mask.to(_dev).float()
        )

        # ── Classification head ──
        logits = self.classifier(gnn_output)
        prob = torch.sigmoid(logits)

        # ── Embedding head (for FAISS) ──
        embeddings = None
        if self.use_faiss and self.embedding_head is not None:
            embeddings = self.embedding_head(gnn_output)

        # ── Loss computation ──
        if labels is not None:
            labels = labels.float()
            cls_loss = torch.log(prob[:, 0] + 1e-10) * labels + \
                       torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            cls_loss = -cls_loss.mean()
            return cls_loss, prob, embeddings
        else:
            return prob, embeddings

    def get_embedding(self, input_ids):
        """
        Get only the FAISS embedding for a batch (no classification).
        Used for building/updating the FAISS index.
        """
        _dev = input_ids.device

        with torch.no_grad():
            if self.args.format == "uni":
                adj, x_feature = build_graph(
                    input_ids.cpu().detach().numpy(),
                    self.w_embeddings,
                    window_size=self.args.window_size
                )
            else:
                adj, x_feature = build_graph_text(
                    input_ids.cpu().detach().numpy(),
                    self.w_embeddings,
                    window_size=self.args.window_size
                )

            adj, adj_mask = preprocess_adj(adj)
            adj_feature = preprocess_features(x_feature)
            adj = torch.from_numpy(adj)
            adj_mask = torch.from_numpy(adj_mask)
            adj_feature = torch.from_numpy(adj_feature)

            gnn_output = self.gnn(
                adj_feature.to(_dev).float(),
                adj.to(_dev).float(),
                adj_mask.to(_dev).float()
            )

            if self.embedding_head is not None:
                return self.embedding_head(gnn_output)
            else:
                return gnn_output

    def get_trainable_params_info(self):
        """Print info about trainable vs frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        info = {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "trainable_percent": round(100 * trainable / max(total, 1), 2),
            "lora_params": self.lora_param_count,
        }
        return info


# ─── DevignModel (unchanged from original) ──────────────────────────────────

class DevignModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DevignModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer

        self.gnn = GGGNN(
            feature_dim_size=args.feature_dim_size, hidden_size=args.hidden_size,
            num_GNN_layers=args.num_GNN_layers, num_classes=args.num_classes,
            dropout=config.hidden_dropout_prob
        )

        self.conv_l1 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = args.feature_dim_size + args.hidden_size
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=args.num_classes)
        self.mlp_y = nn.Linear(in_features=args.hidden_size, out_features=args.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, labels=None):
        _dev = input_ids.device

        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings)
        else:
            adj, x_feature = build_graph_text(input_ids.cpu().detach().numpy(), self.w_embeddings)

        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature).to(_dev).float()

        outputs = self.gnn(adj_feature.to(_dev).float(), adj.to(_dev).float(),
                           adj_mask.to(_dev).float())

        c_i = torch.cat((outputs, adj_feature), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(nn.functional.relu(self.conv_l1(outputs.transpose(1, 2))))
        Y_2 = self.maxpool2(nn.functional.relu(self.conv_l2(Y_1))).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(nn.functional.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(nn.functional.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        prob = self.sigmoid(avg)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
