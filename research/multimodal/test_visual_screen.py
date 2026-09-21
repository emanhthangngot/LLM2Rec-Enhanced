from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    try:
        from .visual_screen import aggregate_history, score_catalog, evaluate_visual_screen_cell
        from .dataset_preflight import compute_effective_rank, score_dataset_rubric
    except ImportError:
        from visual_screen import aggregate_history, score_catalog, evaluate_visual_screen_cell
        from dataset_preflight import compute_effective_rank, score_dataset_rubric
else:
    aggregate_history = None
    score_catalog = None
    evaluate_visual_screen_cell = None
    compute_effective_rank = None
    score_dataset_rubric = None


@unittest.skipIf(np is None, "NumPy is required for visual screen tests")
class VisualScreenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2**-0.5, 2**-0.5],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.availability = np.array([0, 1, 1, 1, 0], dtype=np.float32)

    def test_aggregate_history_linear_recency(self) -> None:
        seq = np.array([1, 2, 0])
        profile = aggregate_history(seq, self.catalog, self.availability, method="linear_recency")
        expected = np.array([1.0, 2.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        self.assertTrue(np.allclose(profile, expected, atol=1e-5))

    def test_score_catalog_standardization(self) -> None:
        user_profile = np.array([1.0, 0.0], dtype=np.float32)
        scores = score_catalog(user_profile, self.catalog, metric="cosine")
        candidates = scores[1:]
        self.assertAlmostEqual(float(candidates.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(candidates.std()), 1.0, places=4)
        self.assertEqual(scores[0], -1e9)

    def test_evaluate_visual_screen_cell(self) -> None:
        val_seqs = [np.array([1, 2]), np.array([2, 3])]
        val_targets = [1, 2]
        res = evaluate_visual_screen_cell(val_seqs, val_targets, self.catalog, self.availability, "linear_recency", "cosine")
        self.assertIn("ndcg@10", res)
        self.assertIn("recall@10", res)
        self.assertGreaterEqual(res["recall@10"], 0.0)

    def test_effective_rank_calculation(self) -> None:
        diag_matrix = np.eye(8, dtype=np.float32)
        rank = compute_effective_rank(diag_matrix)
        self.assertAlmostEqual(rank, 8.0, delta=0.5)

    def test_score_dataset_rubric_disqualification(self) -> None:
        score = score_dataset_rubric(0.90, 0.98, 64.0, 0.2, 5000, 4.0)
        self.assertEqual(score, -1.0)


if __name__ == "__main__":
    unittest.main()
