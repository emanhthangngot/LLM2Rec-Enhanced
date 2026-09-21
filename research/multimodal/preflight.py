"""Pre-training contracts for the LLM2Rec visual-signal screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CoverageReport:
    catalog_items: int
    catalog_items_with_images: int
    test_targets: int
    test_targets_with_images: int
    decode_failures: int
    id_conflicts: int

    @property
    def catalog_coverage(self) -> float:
        return self.catalog_items_with_images / self.catalog_items if self.catalog_items else 0.0

    @property
    def test_target_coverage(self) -> float:
        return self.test_targets_with_images / self.test_targets if self.test_targets else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "catalog_coverage": self.catalog_coverage,
            "test_target_coverage": self.test_target_coverage,
        }


def make_coverage_report(
    catalog_ids: Iterable[object],
    image_ids: Iterable[object],
    test_target_ids: Iterable[object],
    decode_failures: Iterable[object] = (),
) -> CoverageReport:
    """Build a coverage report without dropping catalog or target rows."""

    catalog = {str(value) for value in catalog_ids}
    image_ids_set = {str(value) for value in image_ids}
    targets = {str(value) for value in test_target_ids}
    failures = {str(value) for value in decode_failures}
    conflicts = (image_ids_set | failures) - catalog
    usable_images = (image_ids_set & catalog) - failures
    return CoverageReport(
        catalog_items=len(catalog),
        catalog_items_with_images=len(usable_images),
        test_targets=len(targets),
        test_targets_with_images=len(targets & usable_images),
        decode_failures=len(failures & catalog),
        id_conflicts=len(conflicts),
    )


def validate_coverage(report: CoverageReport, minimum: float = 0.95) -> None:
    """Fail closed when either image coverage gate is below the plan threshold."""

    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum coverage must be in (0, 1]")
    if report.id_conflicts:
        raise RuntimeError(f"image manifest contains {report.id_conflicts} catalog-ID conflicts")
    if report.catalog_coverage < minimum:
        raise RuntimeError(
            f"catalog image coverage {report.catalog_coverage:.3f} is below gate {minimum:.3f}"
        )
    if report.test_target_coverage < minimum:
        raise RuntimeError(
            f"test-target image coverage {report.test_target_coverage:.3f} is below gate {minimum:.3f}"
        )


def _is_hex_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def validate_manifest(manifest: Mapping[str, object]) -> None:
    """Validate the immutable manifest required before a GPU screen can run."""

    required = {
        "source_commit",
        "clip_model_id",
        "clip_resolved_revision",
        "catalog_count",
        "test_target_count",
        "coverage",
        "feature_hashes",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"visual preflight manifest missing keys: {missing}")
    if not _is_hex_digest(manifest["clip_resolved_revision"], 40):
        raise ValueError("clip_resolved_revision must be a 40-character hex revision")
    if not _is_hex_digest(manifest["source_commit"], 40):
        raise ValueError("source_commit must be a 40-character hex commit")
    if not isinstance(manifest["feature_hashes"], Mapping):
        raise ValueError("feature_hashes must be a mapping of named SHA256 digests")
    for name in ("image_manifest_sha256", "visual_features_sha256"):
        if not _is_hex_digest(manifest["feature_hashes"].get(name), 64):
            raise ValueError(f"feature_hashes.{name} must be a 64-character hex SHA256")
    if not isinstance(manifest["coverage"], Mapping):
        raise ValueError("coverage must be a serialized CoverageReport mapping")
    report = CoverageReport(
        catalog_items=int(manifest["coverage"]["catalog_items"]),
        catalog_items_with_images=int(manifest["coverage"]["catalog_items_with_images"]),
        test_targets=int(manifest["coverage"]["test_targets"]),
        test_targets_with_images=int(manifest["coverage"]["test_targets_with_images"]),
        decode_failures=int(manifest["coverage"]["decode_failures"]),
        id_conflicts=int(manifest["coverage"]["id_conflicts"]),
    )
    if report.catalog_items != int(manifest["catalog_count"]):
        raise ValueError("manifest catalog_count disagrees with coverage report")
    if report.test_targets != int(manifest["test_target_count"]):
        raise ValueError("manifest test_target_count disagrees with coverage report")
    validate_coverage(report)
