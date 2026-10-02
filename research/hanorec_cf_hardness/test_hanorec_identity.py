from __future__ import annotations
import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep import _identity_rows, _user_seen_items


class IdentityTests(unittest.TestCase):
    def test_sliding_windows_and_future_train_positives_are_seen(self):
        rows = [[1, 2, 3, 4], [3, 4, 5, 6], [1, 2, 3, 7]]
        seen = _user_seen_items(rows, ["same-user", "same-user", "another-user"])
        self.assertEqual(seen["same-user"], {1, 2, 3, 4, 5, 6})
        self.assertNotIn(7, seen["same-user"])
        self.assertNotIn(6, seen["another-user"])

    def test_csv_alignment_offset_and_real_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "train"
            folder.mkdir()
            path = folder / "Video_Games_5_1996-9-2023-10.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["user_id", "history_item_id", "item_id"])
                writer.writeheader()
                writer.writerow({"user_id": "alice", "history_item_id": "[0, 1]", "item_id": 2})
                writer.writerow({"user_id": "alice", "history_item_id": "[1, 2]", "item_id": 3})
                writer.writerow({"user_id": "bob", "history_item_id": "[0, 1]", "item_id": 4})
            pin = hashlib.sha256(path.read_bytes()).hexdigest()
            clusters, _ = _identity_rows(root, "train", [[1, 2, 3], [2, 3, 4], [1, 2, 5]], pin)
            self.assertEqual(clusters[0], clusters[1])
            self.assertNotEqual(clusters[0], clusters[2])
            self.assertNotIn("alice", clusters)
            with self.assertRaises(RuntimeError):
                _identity_rows(root, "train", [[1, 2, 4], [2, 3, 4], [1, 2, 5]], pin)
            with self.assertRaises(RuntimeError):
                _identity_rows(root, "train", [[1, 2, 3]], pin)
            with self.assertRaises(RuntimeError):
                _identity_rows(root, "train", [[1, 2, 3], [2, 3, 4], [1, 2, 5]], "bad-hash")


if __name__ == "__main__":
    unittest.main()
