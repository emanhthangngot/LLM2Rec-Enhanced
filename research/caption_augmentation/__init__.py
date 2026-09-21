"""Caption-augmentation experiment package for the LLM2Rec research family.

See plans/260915-0955-visual-delta-fusion-pilot/plan.md for the authoritative
scientific protocol (arms, controls, budget, and evaluation contract) and
phase-01-local-module.md for this package's Phase 1 scope. This package is
standard-library only at import time; caption/paraphrase model backends are
loaded lazily inside caption.py functions so nothing here requires torch,
transformers, or a GPU.

Naming note (deliberate): the directory is `visual_delta_fusion`, matching the
sibling `baseline/` and `multimodal/` packages' import convention
(`code/llm2rec/README.md`). It predates the plan's later "no lexical delta
filter" decision; kept for path/import stability rather than churned to match
the current arm vocabulary (title-only/null/real/shuffle/paraphrase).
"""

from .corpus import (
    ARMS,
    CUE_SEPARATOR,
    TITLE_PREFIX,
    UNAVAILABLE,
    CatalogRow,
    build_arms,
    build_catalog_rows,
    compute_common_history_suffix,
    compute_shared_history_suffix,
    frequency_bin_derangement,
    fuse_text,
    load_shard_records,
    normalize_whitespace,
    shard_records,
    truncate_to_token_cap,
    validate_shard_manifest,
    write_arm_corpora,
    write_sharded_records,
)
from .crosswalk import (
    ARCHIVE_DOMAIN_DIRS,
    CATALOG_SIZE,
    DEVELOPMENT_DOMAIN,
    DOMAIN_SIZES,
    REPLICATION_DOMAIN,
    Crosswalk,
    DomainBlock,
    bind_titles,
    boundary_titles,
    build_crosswalk,
    catalog_blocks,
    read_item_asin_pairs,
)

__all__ = [
    "ARMS",
    "CUE_SEPARATOR",
    "TITLE_PREFIX",
    "UNAVAILABLE",
    "ARCHIVE_DOMAIN_DIRS",
    "CATALOG_SIZE",
    "DEVELOPMENT_DOMAIN",
    "DOMAIN_SIZES",
    "REPLICATION_DOMAIN",
    "CatalogRow",
    "Crosswalk",
    "DomainBlock",
    "bind_titles",
    "boundary_titles",
    "build_arms",
    "build_catalog_rows",
    "build_crosswalk",
    "catalog_blocks",
    "compute_common_history_suffix",
    "compute_shared_history_suffix",
    "compute_training_frequencies",
    "frequency_bin_derangement",
    "fuse_text",
    "load_shard_records",
    "normalize_whitespace",
    "truncate_to_token_cap",
    "validate_shard_manifest",
    "write_arm_corpora",
    "write_sharded_records",
]
