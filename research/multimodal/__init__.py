"""Visual-signal experiments for the LLM2Rec compatibility baseline."""

from .preflight import CoverageReport, make_coverage_report, validate_coverage, validate_manifest

__all__ = [
    "CoverageReport",
    "make_coverage_report",
    "validate_coverage",
    "VisualScoreResidual",
    "validate_manifest",
]


def __getattr__(name: str):
    if name == "VisualScoreResidual":
        from .models.score_visual_fusion import VisualScoreResidual

        return VisualScoreResidual
    raise AttributeError(name)
