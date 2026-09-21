"""Memory-bounded residual head for randomized visual interventions."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class InterventionalVisualResidual(nn.Module):
    """Low-rank visual/text interaction with a small row-catalog state."""

    def __init__(self, context_dim: int, item_text_dim: int, visual_dim: int, hidden_dim: int = 32, interaction_dim: int = 8) -> None:
        super().__init__()
        if context_dim < 0 or item_text_dim < 1 or visual_dim < 1 or hidden_dim < 1 or interaction_dim < 1:
            raise ValueError("feature dimensions must be non-negative/positive as applicable")
        self.context_dim, self.item_text_dim, self.visual_dim, self.interaction_dim = context_dim, item_text_dim, visual_dim, interaction_dim
        self.text_projection = nn.Linear(item_text_dim, interaction_dim, bias=False)
        self.visual_projection = nn.Linear(visual_dim, interaction_dim, bias=False)
        self.catalog_projection = nn.Linear(item_text_dim + visual_dim + interaction_dim, hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, text_logits: Tensor, context: Tensor, item_text: Tensor, catalog_visual: Tensor) -> Tensor:
        if text_logits.ndim != 2 or context.ndim != 2 or item_text.ndim != 2 or catalog_visual.ndim != 2:
            raise ValueError("logits/context/item_text/catalog_visual ranks are invalid")
        batch, catalog = text_logits.shape
        if context.shape != (batch, self.context_dim):
            raise ValueError("context dimensions do not match the residual head")
        if item_text.shape != (catalog, self.item_text_dim):
            raise ValueError("item_text must be [catalog,item_text_dim]")
        if catalog_visual.shape != (catalog, self.visual_dim):
            raise ValueError("catalog_visual must be [catalog,visual_dim]")
        text_features = self.text_projection(item_text)
        visual_features = self.visual_projection(catalog_visual)
        interaction = text_features * visual_features
        catalog_state = self.catalog_projection(
            torch.cat((item_text, catalog_visual, interaction), dim=-1)
        )
        hidden = F.gelu(catalog_state[None, :, :] + self.context_projection(context)[:, None, :])
        return text_logits + self.output(hidden).squeeze(-1)


def placebo_regularized_loss(text_logits: Tensor, aligned_logits: Tensor, shuffled_logits: Tensor, no_image_logits: Tensor, targets: Tensor, *, placebo_weight: float = 0.25) -> Tensor:
    if placebo_weight < 0 or any(value.ndim != 2 or value.shape != text_logits.shape for value in (text_logits, aligned_logits, shuffled_logits, no_image_logits)):
        raise ValueError("logit tensors must have identical rank-2 shapes")
    if targets.ndim != 1 or targets.shape[0] != aligned_logits.shape[0]:
        raise ValueError("targets must be one index per row")
    target_score = aligned_logits.gather(1, targets[:, None]).squeeze(1)
    negative_score = (aligned_logits.sum(dim=1) - target_score) / max(aligned_logits.shape[1] - 1, 1)
    ranking = F.softplus(negative_score - target_score).mean()
    placebo = (shuffled_logits - text_logits).square().mean() + (no_image_logits - text_logits).square().mean()
    return ranking + placebo_weight * placebo


def assert_arm_parity(row_ids: Tensor, target_ids: Tensor, text_logits: Tensor, availability: Tensor, *arms: tuple[str, Tensor, Tensor, Tensor, Tensor]) -> None:
    for name, arm_rows, arm_targets, arm_logits, arm_availability in arms:
        if not torch.equal(row_ids, arm_rows): raise ValueError(f"{name} row IDs differ from reference")
        if not torch.equal(target_ids, arm_targets): raise ValueError(f"{name} target IDs differ from reference")
        if not torch.equal(text_logits, arm_logits): raise ValueError(f"{name} text logits differ from reference")
        if not torch.equal(availability, arm_availability): raise ValueError(f"{name} availability differs from reference")
