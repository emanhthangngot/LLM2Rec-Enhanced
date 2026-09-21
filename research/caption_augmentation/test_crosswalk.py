"""Stdlib-only regression tests for crosswalk.py.

Run: python -m unittest discover -s code/llm2rec/visual_delta_fusion -p "test_*.py"
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from crosswalk import (
    ARCHIVE_DOMAIN_DIRS,
    CATALOG_SIZE,
    DOMAIN_SIZES,
    DomainBlock,
    bind_titles,
    boundary_titles,
    build_crosswalk,
    catalog_blocks,
    read_item_asin_pairs,
)


def _write_csv(path: Path, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_asins", "item_asin", "history_item_id", "item_id"])
        for item_id, asin in rows:
            writer.writerow(["u", f"['{asin}']", asin, "[0]", item_id])


class CatalogBlocksTests(unittest.TestCase):
    def test_sizes_sum_to_catalog(self) -> None:
        self.assertEqual(sum(size for _, size in DOMAIN_SIZES), CATALOG_SIZE)

    def test_six_domains_no_gaps(self) -> None:
        blocks = catalog_blocks()
        self.assertEqual(len(blocks), 6)
        self.assertEqual(blocks[0].start, 0)
        self.assertEqual(blocks[-1].stop, CATALOG_SIZE)
        for previous, current in zip(blocks, blocks[1:]):
            self.assertEqual(previous.stop, current.start)

    def test_paper_table_1_order_and_sizes(self) -> None:
        self.assertEqual([b.domain for b in catalog_blocks()], [
            "Arts_Crafts_and_Sewing", "Electronics", "Home_and_Kitchen",
            "Video_Games", "Movies_and_TV", "Tools_and_Home_Improvement",
        ])
        self.assertEqual([b.size for b in catalog_blocks()], [12454, 20150, 33478, 9517, 13190, 19964])

    def test_wrong_sum_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            catalog_blocks([("A", 10), ("B", 10)])


class ReadPairsTests(unittest.TestCase):
    def test_unions_splits_and_detects_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train" / "x.csv"
            valid = Path(tmp) / "valid" / "x.csv"
            _write_csv(train, [(1, "A1"), (2, "A2")])
            _write_csv(valid, [(2, "A2"), (3, "A3")])
            pairs, blank = read_item_asin_pairs([train, valid])
            self.assertEqual(pairs, {1: "A1", 2: "A2", 3: "A3"})
            self.assertEqual(blank, set())

    def test_conflicting_pair_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.csv"
            _write_csv(bad, [(1, "A1"), (1, "A2")])
            with self.assertRaises(ValueError):
                read_item_asin_pairs([bad])

    def test_blank_asin_recorded_not_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write("user_id,item_asins,item_asin,history_item_id,item_id\n")
                handle.write("u,['A1'],A1,[0],1\n")
                handle.write("u,[],,[0],2\n")
            pairs, blank = read_item_asin_pairs([path])
            self.assertEqual(pairs, {1: "A1"})
            self.assertEqual(blank, {2})


class BuildCrosswalkTests(unittest.TestCase):

    def test_verified_domain_offset_identity(self) -> None:
        got = build_crosswalk(
            {0: "a0", 1: "a1", 2: "a2", 4: "x9"},
            {"A": {0: "a0", 1: "a1", 2: "a2"}},
            sizes=[("A", 3), ("B", 2)],
        )
        self.assertEqual(got.blocks[0], DomainBlock("A", 0, 3))
        self.assertEqual(got.verified_by_csv, frozenset({"A"}))
        self.assertEqual(got.global_of("A", 2), 2)
        self.assertEqual(got.domain_of(4), "B")
        self.assertEqual(got.asin_less, frozenset({3}))
    def test_cross_domain_asin_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_crosswalk(
                {0: "a0", 1: "a1"},
                {"A": {0: "a0"}, "B": {1: "a0"}},
                sizes=[("A", 1), ("B", 1)],
            )

    def test_offset_violation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_crosswalk(
                {0: "a0", 1: "zz"},
                {"A": {0: "a0", 1: "a1"}},
                sizes=[("A", 2), ("B", 1)],
            )

    def test_asin_less_positions_recorded(self) -> None:
        got = build_crosswalk(
            {0: "a0"},
            {"A": {0: "a0"}},
            sizes=[("A", 2), ("B", 1)],
        )
        self.assertEqual(got.asin_less, frozenset({1, 2}))

    def test_unknown_domain_lookup_raises(self) -> None:
        got = build_crosswalk({0: "a0"}, {"A": {0: "a0"}}, sizes=[("A", 1), ("B", 1)])
        with self.assertRaises(KeyError):
            got.global_of("zzz", 0)
        with self.assertRaises(KeyError):
            got.global_of("A", 7)
        with self.assertRaises(KeyError):
            got.domain_of(99)


class BindTitlesTests(unittest.TestCase):
    def test_count_mismatch_is_rejected(self) -> None:
        got = build_crosswalk({0: "a0"}, {"A": {0: "a0"}}, sizes=[("A", 1), ("B", 1)])
        with self.assertRaises(ValueError):
            bind_titles(got, "only one line\n")

    def test_boundary_report(self) -> None:
        got = build_crosswalk({0: "a0"}, {"A": {0: "a0"}}, sizes=[("A", 1), ("B", 1)])
        titles = bind_titles(got, "x\n" * CATALOG_SIZE)
        rows = boundary_titles(titles, got, 1)
        self.assertEqual(len(rows), 2 * len(got.blocks))


class ArchiveLayoutTests(unittest.TestCase):
    def test_three_csv_domains_are_the_proven_ones(self) -> None:
        self.assertEqual(set(ARCHIVE_DOMAIN_DIRS), {
            "Arts_Crafts_and_Sewing", "Video_Games", "Movies_and_TV",
        })


if __name__ == "__main__":
    unittest.main()
