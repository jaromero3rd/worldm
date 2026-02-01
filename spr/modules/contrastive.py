"""Modified based on https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py
    With logsumexp trick, vectorized, and better numerical stability."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelDifference(nn.Module):
    def __init__(self, distance_type="l1"):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        if self.distance_type == "l1":
            return torch.abs(labels[:, None] - labels[None, :])
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type="cos"):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        if self.similarity_type == "l2":
            return -(features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == "cos":
            return F.cosine_similarity(features[:, None, :], features[None, :, :], dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2.0, label_diff="l1", feature_sim="l2"):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        n = logits.shape[0]

        # remove self (diagonal)
        eye_mask = (1 - torch.eye(n).to(logits.device)).bool()
        logits = logits.masked_select(eye_mask).view(n, n - 1)
        label_diffs = label_diffs.masked_select(eye_mask).view(n, n - 1)

        # [i, j, k]: for sample i, if sample j is the positive sample, the entry is true if sample k is negative sample
        neg_mask = label_diffs.unsqueeze(1) >= label_diffs.unsqueeze(-1)  # [n, n-1, n-1]

        # Set logits for non-negative samples to -inf so they don't contribute to the sum
        masked_logits = torch.where(neg_mask, logits.unsqueeze(1), -torch.inf)

        # Compute denominator, logsumexp trick
        log_denominator = torch.logsumexp(masked_logits, dim=-1)  # Shape: [n, n-1]

        # Clamp the output to a large negative number, avoid -inf
        log_denominator = torch.clamp(log_denominator, min=-100)

        # Average over log probabilities
        pos_log_probs = logits - log_denominator  # [n, n-1]
        return -pos_log_probs.mean()


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for clustering representations of the same policy."""

    def __init__(self, temperature=0.07, topk=None):
        """
        Args:
            temperature: contrastive temperature
            topk: if not None, use top-k hardest negatives per sample
        """
        super().__init__()
        self.temperature = temperature
        self.topk = topk

    def forward(self, features, policy_labels):
        # Normalize representations
        features = F.normalize(features, dim=1)

        # Compute similarity logits
        logits = torch.mm(features, features.T) / self.temperature  # [batch, batch]

        # Positive mask (same policy, exclude self)
        positive_mask = policy_labels.unsqueeze(1) == policy_labels.unsqueeze(0)
        positive_mask.fill_diagonal_(False)

        # Negative mask (different policy)
        negative_mask = ~positive_mask
        negative_mask.fill_diagonal_(False)

        if self.topk is not None:
            # For each sample, select top-k hardest negatives
            neg_logits = logits.masked_fill(~negative_mask, -float("inf"))  # [B,B]
            topk_vals, _ = torch.topk(neg_logits, k=min(self.topk, logits.shape[0] - 1), dim=1)

            # Build denominator logits: positives + top-k negatives
            # Expand topk_vals to compare when computing logsumexp
            # We replace all negatives not in top-k with -inf
            threshold = topk_vals[:, -1].unsqueeze(1)  # [B,1]
            hard_neg_mask = neg_logits >= threshold  # [B,B]
            denom_mask = positive_mask | hard_neg_mask
        else:
            # Standard InfoNCE: use all negatives
            denom_mask = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)

        # Apply mask to logits (keep only positives + selected negatives)
        logits_for_denom = logits.masked_fill(~denom_mask, -torch.inf)

        # Compute denominator logsumexp
        log_denominator = torch.logsumexp(logits_for_denom, dim=1, keepdim=True)
        log_prob = logits - log_denominator

        # Compute positive log likelihood
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss
