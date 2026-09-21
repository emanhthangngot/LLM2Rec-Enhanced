"""Stdlib-only regression tests for corpus.py.

Run: python -m unittest discover -s code/llm2rec/visual_delta_fusion -p "test_*.py"
"""
from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from corpus import (
    ARMS,
    UNAVAILABLE,
    build_arms,
    build_catalog_rows,
    compute_common_history_suffix,
    compute_training_frequencies,
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


class WordTokenizer:
    """Deterministic stub tokenizer: one token per whitespace-separated word."""

    def encode(self, text: str) -> list[int]:
        return [hash(word) % 1000 for word in text.split()]


class FuseTextTests(unittest.TestCase):
    def test_real_cue_format(self) -> None:
        self.assertEqual(
            fuse_text("Mario Party", "a colorful video game box"),
            "Title: Mario Party; Visual cues: a colorful video game box",
        )

    def test_empty_cue_becomes_unavailable(self) -> None:
        self.assertEqual(fuse_text("Mario Party", ""), "Title: Mario Party; Visual cues: unavailable")

    def test_empty_title_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fuse_text("", "some cue")

    def test_normalize_whitespace_collapses_only(self) -> None:
        self.assertEqual(normalize_whitespace("  a   caption\twith\nwhitespace  "), "a caption with whitespace")


class BuildCatalogRowsTests(unittest.TestCase):
    def test_covers_full_id_range_and_preserves_order(self) -> None:
        titles = {"1": "Alpha", "2": "Beta", "3": "Gamma"}
        asin_map = {"items": {"1": {"parent_asin": "A1", "title": "Alpha"}, "3": {"parent_asin": "A3", "title": "Gamma"}}}
        rows = build_catalog_rows(titles, asin_map, domain="games")
        self.assertEqual([row.downstream_item_id for row in rows], [1, 2, 3])
        self.assertEqual([row.order_index for row in rows], [0, 1, 2])
        self.assertEqual(rows[0].parent_asin, "A1")
        self.assertIsNone(rows[1].parent_asin)  # unmapped row kept, not dropped
        self.assertEqual(rows[1].image_status, "unmapped")
        self.assertEqual(rows[2].parent_asin, "A3")

    def test_rejects_gap_in_id_range(self) -> None:
        with self.assertRaises(ValueError):
            build_catalog_rows({"1": "Alpha", "3": "Gamma"})

    def test_rejects_empty_titles(self) -> None:
        with self.assertRaises(ValueError):
            build_catalog_rows({})


class FrequencyBinDerangementTests(unittest.TestCase):
    def test_permutation_has_no_fixed_points_and_covers_available(self) -> None:
        available = list(range(1, 201))
        frequencies = {item_id: (item_id % 7) for item_id in available}
        mapping = frequency_bin_derangement(available, frequencies, seed=7001, bin_size=64)
        self.assertEqual(set(mapping.keys()), set(available))
        self.assertEqual(sorted(mapping.values()), sorted(mapping.keys()))
        self.assertTrue(all(donor != item_id for item_id, donor in mapping.items()))

    def test_deterministic_for_same_seed(self) -> None:
        available = list(range(1, 130))
        frequencies = {item_id: item_id for item_id in available}
        first = frequency_bin_derangement(available, frequencies, seed=7001)
        second = frequency_bin_derangement(available, frequencies, seed=7001)
        self.assertEqual(first, second)

    def test_different_seeds_can_differ(self) -> None:
        available = list(range(1, 130))
        frequencies = {item_id: item_id for item_id in available}
        first = frequency_bin_derangement(available, frequencies, seed=7001)
        second = frequency_bin_derangement(available, frequencies, seed=7002)
        self.assertNotEqual(first, second)

    def test_singleton_tail_merged_not_left_alone(self) -> None:
        # 65 items with bin_size=64 leaves a tail of exactly 1, which must merge
        # into the previous bin rather than crash or fix itself.
        available = list(range(1, 66))
        frequencies = {item_id: item_id for item_id in available}
        mapping = frequency_bin_derangement(available, frequencies, seed=7001, bin_size=64)
        self.assertEqual(len(mapping), 65)
        self.assertTrue(all(donor != item_id for item_id, donor in mapping.items()))

    def test_fewer_than_two_available_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            frequency_bin_derangement([1], {1: 1}, seed=7001)


class TruncateToTokenCapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WordTokenizer()

    def test_truncates_at_word_boundary_under_cap(self) -> None:
        text = " ".join(f"word{i}" for i in range(50))
        truncated, count = truncate_to_token_cap(text, cap=10, tokenizer=self.tokenizer)
        self.assertEqual(count, 10)
        self.assertEqual(truncated, " ".join(f"word{i}" for i in range(10)))

    def test_text_under_cap_is_unchanged(self) -> None:
        text = "short caption here"
        truncated, count = truncate_to_token_cap(text, cap=32, tokenizer=self.tokenizer)
        self.assertEqual(truncated, text)
        self.assertEqual(count, 3)

    def test_empty_text_returns_empty(self) -> None:
        self.assertEqual(truncate_to_token_cap("", cap=32, tokenizer=self.tokenizer), ("", 0))

    def test_zero_cap_returns_empty(self) -> None:
        self.assertEqual(truncate_to_token_cap("word", cap=0, tokenizer=self.tokenizer), ("", 0))


class BuildArmsTests(unittest.TestCase):
    def _rows(self):
        titles = {"1": "Alpha Widget", "2": "Beta Widget", "3": "Gamma Widget"}
        return build_catalog_rows(titles)

    def test_five_arms_identical_length_and_order(self) -> None:
        rows = self._rows()
        real_cues = {1: "a red widget", 2: "a blue widget"}  # item 3 unavailable
        paraphrase_cues = {1: "widget alpha version", 2: "widget beta version"}
        donor_map = {1: 2, 2: 1}  # item 3 has no donor: unavailable
        arms = build_arms(rows, real_cues, paraphrase_cues, donor_map)
        self.assertEqual(len(arms), 3)
        self.assertEqual([arm.downstream_item_id for arm in arms], [1, 2, 3])
        for arm in arms:
            self.assertEqual(set(arm.texts), set(ARMS))

    def test_title_only_is_raw_title_not_wrapped(self) -> None:
        rows = self._rows()
        arms = build_arms(rows, {}, {}, {})
        self.assertEqual(arms[0].texts["title-only"], "Alpha Widget")

    def test_unavailable_item_gets_unavailable_everywhere(self) -> None:
        rows = self._rows()
        cues = {1: "a red widget", 2: "a blue widget"}
        paraphrases = {1: "widget alpha version", 2: "widget beta version"}
        arms = build_arms(rows, cues, paraphrases, {1: 2, 2: 1})
        item3 = arms[2]
        for name in ("real", "shuffle", "paraphrase", "null"):
            self.assertTrue(item3.texts[name].endswith(UNAVAILABLE))

    def test_shuffle_uses_donor_real_cue(self) -> None:
        rows = self._rows()
        real_cues = {1: "a red widget", 2: "a blue widget"}
        paraphrase_cues = {1: "widget alpha", 2: "widget beta"}
        arms = build_arms(rows, real_cues, paraphrase_cues, {1: 2, 2: 1})
        self.assertTrue(arms[0].texts["shuffle"].endswith("a blue widget"))
        self.assertTrue(arms[1].texts["shuffle"].endswith("a red widget"))

    def test_title_segment_shared_across_cue_bearing_arms(self) -> None:
        rows = self._rows()
        real_cues = {1: "a red widget", 2: "a blue widget"}
        paraphrases = {1: "widget alpha version", 2: "widget beta version"}
        arms = build_arms(rows, real_cues, paraphrases, {1: 2, 2: 1})
        prefix = "Title: Alpha Widget"
        for name in ("null", "real", "shuffle", "paraphrase"):
            self.assertTrue(arms[0].texts[name].startswith(prefix), arms[0].texts[name])


class CommonHistorySuffixTests(unittest.TestCase):
    def test_retains_longest_suffix_within_budget(self) -> None:
        tokenizer = WordTokenizer()
        arm_text_by_id = {i: " ".join(["word"] * 5) for i in range(1, 21)}
        history = list(range(1, 21))
        retained, used = compute_common_history_suffix(
            history, target_title="target word", arm_text_by_id=arm_text_by_id,
            tokenizer=tokenizer, special_token_overhead=0, max_tokens=27,
        )
        # Complete rendered history + target sequence is counted.
        self.assertEqual(retained, list(range(16, 21)))
        self.assertEqual(used, 27)

    def test_oversized_target_raises_instead_of_truncating(self) -> None:
        tokenizer = WordTokenizer()
        with self.assertRaises(RuntimeError):
            compute_common_history_suffix(
                [1], target_title=" ".join(["word"] * 100), arm_text_by_id={1: "word"},
                tokenizer=tokenizer, special_token_overhead=0, max_tokens=50,
            )

    def test_empty_history_preserves_target_when_it_fits(self) -> None:
        tokenizer = WordTokenizer()
        retained, used = compute_common_history_suffix([], "target", {}, tokenizer, 0, 1)
        self.assertEqual(retained, [])
        self.assertEqual(used, 1)

    def test_shared_suffix_uses_tightest_variant(self) -> None:
        from corpus import compute_shared_history_suffix

        tokenizer = WordTokenizer()
        history = [1, 2, 3]
        variants = {
            "title": {1: "a", 2: "b", 3: "c"},
            "real": {1: "long long", 2: "long long", 3: "long long"},
        }
        suffix, budgets = compute_shared_history_suffix(history, "target", variants, tokenizer, 0, 5)
        self.assertEqual(suffix, [2, 3])
        self.assertLessEqual(max(budgets.values()), 5)

    def test_same_history_and_variant_yields_identical_suffix(self) -> None:
        tokenizer = WordTokenizer()
        arm_text_by_id = {i: "word word word" for i in range(1, 11)}
        history = list(range(1, 11))
        first, _ = compute_common_history_suffix(history, "t", arm_text_by_id, tokenizer, 0, 20)
        second, _ = compute_common_history_suffix(history, "t", arm_text_by_id, tokenizer, 0, 20)
        self.assertEqual(first, second)


class TrainingFrequencyTests(unittest.TestCase):
    def test_counts_occurrences_across_sequences(self) -> None:
        sequences = [[1, 2, 3], [2, 3, 3], [1]]
        frequencies = compute_training_frequencies(sequences)
        self.assertEqual(frequencies, {1: 2, 2: 2, 3: 3})


class WriteArmCorporaTests(unittest.TestCase):
    def test_writes_json_and_txt_with_matching_hashes(self) -> None:
        rows = build_catalog_rows({"1": "Alpha", "2": "Beta"})
        arms = build_arms(rows, {}, {}, {})
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            hashes = write_arm_corpora(arms, out_dir)
            self.assertEqual(set(hashes), set(ARMS))
            real_json = json.loads((out_dir / "real" / "item_texts.json").read_text(encoding="utf-8"))
            self.assertEqual(set(real_json), {"1", "2"})
            real_txt_lines = (out_dir / "real" / "item_texts.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(real_txt_lines), 2)
            self.assertEqual(real_txt_lines[0], real_json["1"])

    def test_newline_in_text_is_rejected(self) -> None:
        rows = build_catalog_rows({"1": "Alpha\nBroken"})
        arms = build_arms(rows, {}, {}, {})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_arm_corpora(arms, Path(tmp))

    def test_empty_arms_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_arm_corpora([], Path(tmp))


class ShardManifestTests(unittest.TestCase):
    def test_shards_preserve_order_and_validate_hashes(self) -> None:
        records = [{"id": index, "text": f"item-{index}"} for index in range(5)]
        self.assertEqual([len(chunk) for chunk in shard_records(records, 2)], [2, 2, 1])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {"protocol_hash": "p", "model_revision": "r"}
            manifest = write_sharded_records(records, root, identity, shard_size=2)
            self.assertEqual(manifest["record_count"], 5)
            checked = validate_shard_manifest(root / "shard-manifest.json", identity)
            self.assertEqual(checked["shard_count"], 3)

    def test_load_shard_records_roundtrips_catalog_order(self) -> None:
        records = [{"id": index, "text": f"item-{index}"} for index in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sharded_records(records, root, {"protocol_hash": "p"}, shard_size=2)
            self.assertEqual(load_shard_records(root), records)

    def test_load_shard_records_preserves_unicode_nel_inside_json_string(self) -> None:
        records = [{"id": 1, "text": "left\u0085right"}, {"id": 2, "text": "plain"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sharded_records(records, root, {"protocol_hash": "p"}, shard_size=2)
            self.assertEqual(load_shard_records(root, {"protocol_hash": "p"}), records)

    def test_transient_read_glitch_recovers_via_retry(self) -> None:
        # Defense in depth for a genuine transient read: the production
        # failure was caused by Unicode-aware line splitting, but a truncated
        # payload must still recover cleanly when a later read matches the
        # manifest hash.
        records = [{"id": index} for index in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {"protocol_hash": "p"}
            write_sharded_records(records, root, identity, shard_size=5)
            shard_path = root / "shard-00000.jsonl"
            real_content = shard_path.read_bytes()
            original_read_bytes = Path.read_bytes
            attempts = {"n": 0}

            def flaky_read_bytes(self, *args, **kwargs):
                if self == shard_path:
                    attempts["n"] += 1
                    if attempts["n"] < 2:
                        return real_content[:10]
                return original_read_bytes(self, *args, **kwargs)

            with unittest.mock.patch.object(Path, "read_bytes", flaky_read_bytes), \
                 unittest.mock.patch("corpus.time.sleep"):
                loaded = load_shard_records(root, identity)
            self.assertEqual(loaded, records)
            self.assertGreaterEqual(attempts["n"], 2)

    def test_persistent_read_failure_raises_with_file_and_line(self) -> None:
        records = [{"id": index} for index in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {"protocol_hash": "p"}
            write_sharded_records(records, root, identity, shard_size=5)
            shard_path = root / "shard-00000.jsonl"
            original_read_bytes = Path.read_bytes

            def always_truncated(self, *args, **kwargs):
                if self == shard_path:
                    return original_read_bytes(self, *args, **kwargs)[:10]
                return original_read_bytes(self, *args, **kwargs)

            with unittest.mock.patch.object(Path, "read_bytes", always_truncated), \
                 unittest.mock.patch("corpus.time.sleep"):
                with self.assertRaises(RuntimeError) as ctx:
                    load_shard_records(root, identity)
            self.assertIn("shard-00000.jsonl", str(ctx.exception))
            self.assertIn("3 attempts", str(ctx.exception))

    def test_merged_read_catches_hash_mismatch_without_a_second_read(self) -> None:
        # The old two-pass design (validate_shard_manifest then
        # load_shard_records) could read a file twice and observe different
        # bytes each time. The merged design reads once; verify it still
        # catches a genuinely tampered file via the identity/hash path.
        records = [{"id": index} for index in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {"protocol_hash": "p"}
            write_sharded_records(records, root, identity, shard_size=5)
            (root / "shard-00000.jsonl").write_bytes(b'{"id": 0}\n{"id": 1}\n{"id": 999}\n')
            with unittest.mock.patch("corpus.time.sleep"):
                with self.assertRaises(RuntimeError) as ctx:
                    load_shard_records(root, identity)
            self.assertIn("hash mismatch", str(ctx.exception))

    def test_resume_rejects_identity_mismatch_and_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sharded_records([{"id": 1}], root, {"protocol_hash": "p"})
            with self.assertRaises(RuntimeError):
                write_sharded_records([{"id": 1}], root, {"protocol_hash": "different"})
            (root / "shard-00000.jsonl").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_shard_manifest(root / "shard-manifest.json", {"protocol_hash": "p"})


if __name__ == "__main__":
    unittest.main()
