"""Offline tests for the paired arm kernel's pure data-preparation helpers."""
from __future__ import annotations

import ast
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "kaggle"))

import video_games_paired_arm as arm  # noqa: E402


def words(text: str) -> int:
    return len(text.split())


def _catalog(cue_for: dict[int, str | None]) -> tuple[list[dict[str, object]], dict[str, str]]:
    titles = {str(i): f"Game {i}" for i in range(1, arm.CATALOG_SIZE + 1)}
    records = [
        {"local_item_id": i, "title": titles[str(i)], "real_cue": cue_for.get(i, f"color: c{i}")}
        for i in range(1, arm.CATALOG_SIZE + 1)
    ]
    return records, titles


class ItemTextTests(unittest.TestCase):
    def test_real_text_caps_cue_and_marks_missing_cues_unavailable(self) -> None:
        records, titles = _catalog({1: "a " * 50, 2: None})
        texts = arm.build_item_texts(records, titles, words)
        self.assertEqual(texts["title-only"][1], "Game 1")
        self.assertEqual(words(texts["real"][1].split(arm.CUE_SEPARATOR)[1]), arm.CUE_CAP_TOKENS)
        self.assertEqual(texts["real"][2], "Title: Game 2; Visual cues: unavailable")

    def test_corpus_title_disagreeing_with_downstream_fails(self) -> None:
        records, titles = _catalog({})
        records[10]["title"] = "Some Other Game"
        with self.assertRaises(RuntimeError):
            arm.build_item_texts(records, titles, words)


class HistoryTests(unittest.TestCase):
    def test_suffix_keeps_most_recent_items_that_fit_real_rendering(self) -> None:
        history = ["a b c"] * 10  # ", ".join adds no whitespace-separated tokens
        keep = arm.shared_history_suffix(history, "t", words, cutoff=3 * 4 + 1 + 2)
        self.assertEqual(keep, 4)
        self.assertEqual(arm.shared_history_suffix(history, "t", words, cutoff=10_000), 10)

    def test_both_arms_keep_identical_history_ids_and_original_targets(self) -> None:
        records, titles = _catalog({i: "w " * 30 for i in range(1, 6)})
        texts = arm.build_item_texts(records, titles, words)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["user_id", "history_item_id", "history_item_title", "item_id", "item_title"])
                writer.writeheader()
                writer.writerow({"user_id": "u", "history_item_id": repr([1, 2, 3, 4, 5]),
                                 "history_item_title": repr([f"Game {i}" for i in range(1, 6)]),
                                 "item_id": 6, "item_title": "Game 6"})
            outputs = {}
            for name in arm.ARMS:
                destination = Path(tmp) / f"{name}.csv"
                arm.render_csft_csv(source, destination, name, texts, {i: i for i in range(1, 6)}, words)
                with destination.open(encoding="utf-8", newline="") as handle:
                    outputs[name] = next(csv.DictReader(handle))
        self.assertEqual(outputs["title-only"]["history_item_id"], outputs["real"]["history_item_id"])
        self.assertEqual(outputs["real"]["item_title"], "Game 6")
        real_history = ast.literal_eval(outputs["real"]["history_item_title"])
        self.assertTrue(all(text.startswith("Title: Game ") for text in real_history))

    def test_history_outside_catalog_fails(self) -> None:
        records, titles = _catalog({})
        texts = arm.build_item_texts(records, titles, words)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.csv"
            source.write_text(
                'user_id,history_item_id,history_item_title,item_id,item_title\nu,[0],[\'x\'],1,Game 1\n',
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                arm.render_csft_csv(source, Path(tmp) / "out.csv", "real", texts, {1: 1}, words)


class IdBridgeTests(unittest.TestCase):
    def _csv(self, root: Path, rows: list[tuple[int, str, str]]) -> Path:
        path = root / "split.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["history_item_id", "item_asins", "history_item_title", "item_id", "item_asin", "item_title"])
            writer.writeheader()
            for item_id, asin, name in rows:
                writer.writerow({"history_item_id": "[]", "item_asins": "[]", "history_item_title": "[]",
                                 "item_id": item_id, "item_asin": asin, "item_title": name})
        return path

    def test_permuted_csv_ids_map_by_asin_with_unique_title_fallback(self) -> None:
        records, titles = _catalog({})
        for record in records:
            record["asin"] = f"A{record['local_item_id']}"
        records[0]["asin"] = None  # falls back to its unique title
        rows = [(arm.CATALOG_SIZE - i, f"A{i}", f"Game {i}") for i in range(1, arm.CATALOG_SIZE + 1)]
        with tempfile.TemporaryDirectory() as tmp:
            mapping = arm.csv_to_downstream_ids([self._csv(Path(tmp), rows)], records, titles)
        self.assertEqual(mapping[arm.CATALOG_SIZE - 1], 1)
        self.assertEqual(mapping[0], arm.CATALOG_SIZE)

    def test_asin_pointing_at_a_different_title_fails(self) -> None:
        records, titles = _catalog({})
        for record in records:
            record["asin"] = f"A{record['local_item_id']}"
        rows = [(i - 1, f"A{i}", f"Game {i}") for i in range(1, arm.CATALOG_SIZE + 1)]
        rows[0] = (0, "A2", "Game 1")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(RuntimeError):
            arm.csv_to_downstream_ids([self._csv(Path(tmp), rows)], records, titles)


class ResultParsingTests(unittest.TestCase):
    def test_parses_three_per_seed_blocks_and_ignores_summary(self) -> None:
        text = "Final Results:\nndcg@10: (0.05, 0.001)\n\n\nResults of each experiment:\n"
        for index in range(1, 4):
            text += f"Experiment {index}:\nndcg@10: 0.0{index}\nrecall@20: 0.1{index}\n\n"
        parsed = arm.parse_sasrec_results(text)
        self.assertEqual([p["ndcg@10"] for p in parsed], [0.01, 0.02, 0.03])

    def test_missing_seed_block_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            arm.parse_sasrec_results("Experiment 1:\nndcg@10: 0.1\n")


if __name__ == "__main__":
    unittest.main()
