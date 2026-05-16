"""
Contrastive and Triplet losses for embedding head training.
Used to learn embeddings where vulnerable code clusters together
and is separated from safe code in the FAISS vector space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    Given a batch of embeddings and binary labels (vuln/safe),
    pulls same-class embeddings together and pushes different-class apart.

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch_size, embed_dim) — L2-normalized embeddings
            labels: (batch_size,) — binary labels (0=safe, 1=vuln)
        Returns:
            scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 normalize (should already be normalized, but just in case)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Cosine similarity matrix: (B, B)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask: same class = 1, different class = 0
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float().to(device)

        # Remove self-similarity from positive mask
        eye = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - eye

        # Count positives per sample
        num_pos = mask_pos.sum(dim=1)

        # If a sample has no positives (only one of its class in batch), skip it
        valid = num_pos > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Log-softmax over similarities (exclude self)
        # Subtract max for numerical stability
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Mask out self
        exp_logits = torch.exp(logits) * (1 - eye)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positive pairs
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (num_pos + 1e-12)

        # Loss: only for valid samples
        loss = -mean_log_prob[valid].mean()

        return loss


class TripletMarginLossWithMining(nn.Module):
    """
    Online hard triplet mining + triplet margin loss.
    For each anchor, finds the hardest positive and hardest negative in the batch.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch_size, embed_dim) — L2-normalized
            labels: (batch_size,) — binary labels
        Returns:
            scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Pairwise distance matrix
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        labels = labels.view(-1)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_neg = 1.0 - mask_pos

        eye = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - eye  # remove self

        # Hardest positive: max distance among positives
        dist_pos = dist_matrix * mask_pos
        hardest_pos, _ = dist_pos.max(dim=1)

        # Hardest negative: min distance among negatives (set non-neg to large value)
        large_val = dist_matrix.max().item() + 1.0
        dist_neg = dist_matrix + (1.0 - mask_neg) * large_val
        hardest_neg, _ = dist_neg.min(dim=1)

        # Only valid anchors (have both pos and neg)
        has_pos = mask_pos.sum(dim=1) > 0
        has_neg = mask_neg.sum(dim=1) > 0
        valid = has_pos & has_neg

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        triplet_loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)

        return triplet_loss.mean()
