"""AmazonMix-6 six-domain crosswalk with paper-verified block structure.

The mixed `AmazonMix-6` catalog is one global item list of 108,753 rows. Its
six domains and boundaries are established by the LLM2Rec paper, Table 1
(KDD'25): Arts (12,454), Electronics (20,150), Home_and_Kitchen (33,478),
Video_Games (9,517), Movies_and_TV (13,190), Tools_and_Home_Improvement
(19,964). Cumulative block boundaries were verified locally against the real
archive before this module was written (see
plans/260915-0955-visual-delta-fusion-pilot/reports/crosswalk-verification.md):
every block size matches Table 1 exactly, block-boundary titles change domain
at exactly the predicted positions, and for the three domains whose per-domain
5-core CSVs ship in the same archive (Arts, Games, Movies) the correspondence
`global_id == block_start + local_id` holds with zero violations and exact
ASIN-set identity.

Why ASINs: `plan.md` forbids matching items by title alone, and titles repeat
across domains. ASINs are stable identifiers, so domain attribution via ASIN
is evidence; title equality across domains is not.

Verification tiers per domain, recorded on the returned `Crosswalk`:
- `verified_by_csv`: domains whose per-domain 5-core CSVs exist in the same
  archive and passed exact zero-violation offset identity. Their local-index
  convention is proven, not assumed.
- The remaining blocks (Electronics, Home_and_Kitchen,
  Tools_and_Home_Improvement) are boundary-positional: block size and position
  match paper Table 1 to the row, but their local index files are absent from
  this archive, so `global_of` for them encodes the uniform-concatenation
  convention explicitly flagged as assumed.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

CATALOG_SIZE = 108753

# Domain sizes from the LLM2Rec paper, Table 1 (KDD'25), in catalog order.
# Their cumulative boundaries were verified against the real archive.
DOMAIN_SIZES: tuple[tuple[str, int], ...] = (
    ("Arts_Crafts_and_Sewing", 12454),
    ("Electronics", 20150),
    ("Home_and_Kitchen", 33478),
    ("Video_Games", 9517),
    ("Movies_and_TV", 13190),
    ("Tools_and_Home_Improvement", 19964),
)

assert sum(size for _, size in DOMAIN_SIZES) == CATALOG_SIZE

# Archive-relative per-domain CSV directories that actually ship in the
# author archive. Electronics, Home_and_Kitchen, and
# Tools_and_Home_Improvement have no per-domain files there.
ARCHIVE_DOMAIN_DIRS: Mapping[str, str] = {
    "Arts_Crafts_and_Sewing": "Arts_Crafts_and_Sewing/5-core",
    "Video_Games": "Video_Games/5-core",
    "Movies_and_TV": "Movies_and_TV/5-core",
}

DEVELOPMENT_DOMAIN = "Video_Games"
REPLICATION_DOMAIN = "Arts_Crafts_and_Sewing"


@dataclass(frozen=True)
class DomainBlock:
    """One domain's contiguous range in the global mixed catalog."""

    domain: str
    start: int
    stop: int  # exclusive

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class Crosswalk:
    """Validated mixed-catalog structure."""

    blocks: tuple[DomainBlock, ...]
    global_asin: Mapping[int, str]
    asin_less: frozenset[int] = frozenset()
    verified_by_csv: frozenset[str] = frozenset()

    def domain_of(self, global_id: int) -> str:
        for block in self.blocks:
            if block.start <= global_id < block.stop:
                return block.domain
        raise KeyError(f"global item {global_id} is outside the catalog")

    def global_of(self, domain: str, local_id: int) -> int:
        """Global catalog index of a domain-local 0-based item id.

        Proven exact for `verified_by_csv` domains. For the rest it encodes
        the uniform-concatenation convention (same layout as the three
        verified domains) and must be flagged `assumed` in any report or
        corpus that uses it. Position 75893 is inside the Movies block but
        has no ASIN anywhere in the archive, so it can never be returned as
        a caption donor and must be `unavailable` in every cue-bearing arm.
        """
        for block in self.blocks:
            if block.domain == domain:
                if not 0 <= local_id < block.size:
                    raise KeyError(f"{domain} local item {local_id} is out of range")
                return block.start + local_id
        raise KeyError(f"unknown domain {domain!r}")


def catalog_blocks(
    sizes: Iterable[tuple[str, int]] = DOMAIN_SIZES, catalog_size: int = CATALOG_SIZE
) -> tuple[DomainBlock, ...]:
    blocks: list[DomainBlock] = []
    cursor = 0
    for domain, size in sizes:
        if size <= 0:
            raise ValueError(f"domain {domain} has non-positive size {size}")
        blocks.append(DomainBlock(domain, cursor, cursor + size))
        cursor += size
    if cursor != catalog_size:
        raise ValueError(f"domain sizes sum to {cursor}, expected {catalog_size}")
    for previous, current in zip(blocks, blocks[1:]):
        if previous.stop != current.start:
            raise ValueError(f"gap between {previous.domain} and {current.domain}")
    return tuple(blocks)


def read_item_asin_pairs(csv_paths: Iterable[Path]) -> tuple[dict[int, str], set[int]]:
    """Read distinct `(item_id, item_asin)` pairs from 5-core CSVs (all splits).

    Returns the pairs plus the set of item ids that appear with a blank ASIN.
    A contradictory pair for one id raises instead of silently winning.
    """
    pairs: dict[int, str] = {}
    blank: set[int] = set()
    files = 0
    for csv_path in sorted(Path(path) for path in csv_paths):
        if not csv_path.is_file():
            raise FileNotFoundError(f"5-core CSV missing: {csv_path}")
        files += 1
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for column in ("item_id", "item_asin"):
                if column not in (reader.fieldnames or ()):
                    raise ValueError(f"{csv_path} lacks required column {column!r}")
            for row in reader:
                raw_id = (row.get("item_id") or "").strip()
                asin = (row.get("item_asin") or "").strip()
                if not raw_id or not raw_id.isdigit():
                    continue
                item_id = int(raw_id)
                if not asin:
                    blank.add(item_id)
                    continue
                previous = pairs.setdefault(item_id, asin)
                if previous != asin:
                    raise ValueError(
                        f"{csv_path} maps item_id {item_id} to both {previous!r} and {asin!r}"
                    )
    if files == 0:
        raise ValueError("no CSV files supplied")
    return pairs, blank


def build_crosswalk(
    mixed_pairs: Mapping[int, str],
    domain_pairs: Mapping[str, Mapping[int, str]],
    blank_asin_ids: Iterable[int] = (),
    sizes: Iterable[tuple[str, int]] | None = None,
) -> Crosswalk:
    """Assemble and validate the crosswalk against the paper's Table 1.

    Every ASIN-bearing mixed id must fall in the block whose domain owns
    that ASIN. Verified domains (per-domain CSVs present) additionally
    satisfy exact `global == block_start + local` offset identity with empty
    ASIN-set difference on both sides. Positions with no ASIN anywhere are
    recorded as `asin_less` rather than attributed.
    """
    resolved = list(sizes) if sizes is not None else list(DOMAIN_SIZES)
    blocks = catalog_blocks(resolved, catalog_size=sum(size for _, size in resolved))
    by_domain = {block.domain: block for block in blocks}
    known = {domain for domain in domain_pairs if domain_pairs[domain]}
    if not known:
        raise ValueError("no per-domain item/ASIN maps supplied")

    asin_owner: dict[str, str] = {}
    for domain in known:
        if domain not in by_domain:
            raise KeyError(f"domain {domain!r} is not one of the six mix domains")
        for asin in domain_pairs[domain].values():
            owner = asin_owner.setdefault(asin, domain)
            if owner != domain:
                raise ValueError(f"ASIN {asin} appears in both {owner} and {domain}")
    catalog_size = sum(block.size for block in blocks)
    global_asin: dict[int, str] = dict(mixed_pairs)
    asin_less: set[int] = {
        global_id for global_id in range(catalog_size) if global_id not in global_asin
    }
    for global_id, asin in global_asin.items():
        owner = asin_owner.get(asin)
        if owner is None:
            continue  # Electronics/Home/Tools have no per-domain CSVs in this archive
        block = next(b for b in blocks if b.start <= global_id < b.stop)
        if owner != block.domain:
            raise ValueError(
                f"global item {global_id} ASIN {asin} belongs to {owner} "
                f"but sits in the {block.domain} block"
            )

    verified: set[str] = set()
    for domain in known:
        block = by_domain[domain]
        block_asins = {global_asin[g] for g in range(block.start, block.stop) if g in global_asin}
        csv_asins = set(domain_pairs[domain].values())
        missing_from_csv = block_asins - csv_asins
        missing_from_block = csv_asins - block_asins
        if missing_from_csv or missing_from_block:
            raise ValueError(
                f"domain {domain} ASIN sets disagree: "
                f"{len(missing_from_csv)} block ASINs absent from CSV, "
                f"{len(missing_from_block)} CSV ASINs absent from block"
            )
        csv_ids = domain_pairs[domain]
        violations = [
            local_id
            for local_id in csv_ids
            if not (0 <= local_id < block.size and global_asin.get(block.start + local_id) == csv_ids[local_id])
        ]
        if violations:
            raise ValueError(
                f"domain {domain} has {len(violations)} offset violations, first: {violations[:5]}"
            )
        verified.add(domain)

    return Crosswalk(
        blocks=blocks,
        global_asin=global_asin,
        asin_less=frozenset(asin_less),
        verified_by_csv=frozenset(verified),
    )


def bind_titles(crosswalk: Crosswalk, plain_text: str) -> list[str]:
    """Attach the positional `item_titles.txt` title list to global IDs.

    The shipped `info/item_titles.txt` is plain titles, one per line, in
    catalog order. A count mismatch raises instead of silently truncating.
    """
    lines = plain_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != CATALOG_SIZE:
        raise ValueError(f"expected {CATALOG_SIZE} title lines, read {len(lines)}")
    return lines


def boundary_titles(titles: list[str], crosswalk: Crosswalk, width: int = 2) -> list[tuple[str, int, str]]:
    """Report the first/last `width` titles of every block for the audit."""
    report: list[tuple[str, int, str]] = []
    for block in crosswalk.blocks:
        for global_id in list(range(block.start, min(block.start + width, block.stop))):
            report.append((block.domain, global_id, titles[global_id]))
        for global_id in list(range(max(block.start, block.stop - width), block.stop)):
            report.append((block.domain, global_id, titles[global_id]))
    return report
