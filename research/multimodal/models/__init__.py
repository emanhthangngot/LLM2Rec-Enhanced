"""Torch-backed visual fusion modules."""

__all__ = [
    "VisualScoreResidual",
]


def __getattr__(name: str):
    if name == "VisualScoreResidual":
        from .score_visual_fusion import VisualScoreResidual

        return VisualScoreResidual
    raise AttributeError(name)
