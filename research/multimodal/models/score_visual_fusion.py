"""Geometry-preserving visual score residual for LLM2Rec.

The text item table and SASRec state stay authoritative. Frozen visual item
vectors form a separate score table; a zero-initialized scalar controls their
contribution to the full-catalog text logits.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VisualScoreResidual(nn.Module):
    """Add standardized visual preference scores without changing text embeddings."""

    def __init__(
        self,
        catalog_visual: Tensor,
        availability: Tensor,
        *,
        padding_idx: int = 0,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        if catalog_visual.ndim != 2:
            raise ValueError("catalog_visual must be rank-2")
        if availability.ndim != 1 or availability.shape[0] != catalog_visual.shape[0]:
            raise ValueError("availability must be rank-1 with one value per catalog row")
        if not torch.isfinite(catalog_visual).all() or not torch.isfinite(availability).all():
            raise ValueError("visual inputs must be finite")
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("availability must be binary")
        if padding_idx < 0 or padding_idx >= catalog_visual.shape[0]:
            raise ValueError("padding_idx is outside the catalog")

        present = availability == 1
        if present.any():
            norms = torch.linalg.vector_norm(catalog_visual[present], dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
                raise ValueError("available visual rows must be L2-normalized")
        if torch.any(catalog_visual[~present] != 0):
            raise ValueError("unavailable visual rows must be zero")

        self.padding_idx = padding_idx
        self.register_buffer("catalog_visual", catalog_visual)
        self.register_buffer("availability", availability)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def history_visual(self, item_seqs: Tensor) -> Tensor:
        """Build a parameter-free recency-weighted visual history representation."""
        if item_seqs.ndim != 2:
            raise ValueError("item_seqs must be rank-2")
        if item_seqs.dtype != torch.long:
            raise ValueError("item_seqs must contain torch.long indices")
        if item_seqs.numel() and (item_seqs.min() < 0 or item_seqs.max() >= len(self.availability)):
            raise ValueError("item_seqs contains an item outside the visual catalog")

        visual = self.catalog_visual[item_seqs]
        available = self.availability[item_seqs]
        valid = (item_seqs != self.padding_idx).to(visual.dtype) * available.to(visual.dtype)
        recency = torch.arange(
            1,
            item_seqs.shape[1] + 1,
            device=item_seqs.device,
            dtype=visual.dtype,
        ).view(1, -1)
        weights = valid * recency
        history = (visual * weights.unsqueeze(-1)).sum(dim=1)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        history = history / denominator
        return F.normalize(history, dim=-1, eps=1e-12)

    def standardized_visual_logits(self, item_seqs: Tensor) -> Tensor:
        """Return per-user standardized cosine scores over the full catalog."""
        user_visual = self.history_visual(item_seqs)
        scores = user_visual @ self.catalog_visual.transpose(0, 1)
        candidate_scores = torch.cat(
            (scores[:, : self.padding_idx], scores[:, self.padding_idx + 1 :]),
            dim=1,
        )
        mean = candidate_scores.mean(dim=1, keepdim=True)
        std = candidate_scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        standardized = (scores - mean) / std
        standardized[:, self.padding_idx] = 0
        no_visual_history = torch.linalg.vector_norm(user_visual, dim=1) == 0
        standardized[no_visual_history] = 0
        return standardized

    def forward(self, text_logits: Tensor, item_seqs: Tensor) -> Tensor:
        if text_logits.ndim != 2:
            raise ValueError("text_logits must be rank-2")
        if text_logits.shape[0] != item_seqs.shape[0]:
            raise ValueError("text_logits and item_seqs must have the same batch size")
        if text_logits.shape[1] != self.catalog_visual.shape[0]:
            raise ValueError("text logits and visual catalog sizes differ")
        visual_logits = self.standardized_visual_logits(item_seqs).to(text_logits.dtype)
        return text_logits + self.alpha.to(text_logits.dtype) * visual_logits
