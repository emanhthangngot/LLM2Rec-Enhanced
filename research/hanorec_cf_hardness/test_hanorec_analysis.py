from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import full_cohort_values, holm_adjust, paired_bootstrap, retriever_verdict, summarize  # noqa: E402


def _row(user, target, ndcg=0.0, recall=0.0, cand=1.0):
    return {"user_id": user, "target": target, "ndcg@10": ndcg, "recall@10": recall, "candidate_recall@20": cand}


def _vals(deltas):
    return [{"ndcg@10": d, "recall@10": 0.0, "candidate_recall@20": 0.0} for d in deltas]


class AnalysisTests(unittest.TestCase):
    def test_uninformative_users_count_as_zero_in_the_full_cohort(self) -> None:
        cohort = [_row("a", 1, cand=0.0), _row("b", 2), _row("c", 3, cand=0.0), _row("d", 4, cand=0.0)]
        values = full_cohort_values(cohort, [_row("b", 2, ndcg=1.0, recall=1.0)])
        self.assertEqual(summarize(values)["ndcg@10"], 0.25)
        with self.assertRaises(ValueError):
            full_cohort_values(cohort, [_row("z", 9, ndcg=1.0)])

    def test_dropping_an_informative_user_is_an_error_not_a_silent_zero(self) -> None:
        cohort = [_row("a", 1), _row("b", 2)]
        with self.assertRaises(ValueError):
            full_cohort_values(cohort, [_row("b", 2, ndcg=1.0)])

    def test_bootstrap_ci_brackets_the_true_mean_and_is_deterministic(self) -> None:
        rng = random.Random(7)
        deltas = [rng.choice([-0.2, 0.0, 0.0, 0.3, 0.5]) for _ in range(200)]
        true_mean = sum(deltas) / len(deltas)
        result = paired_bootstrap(_vals(deltas), _vals([0.0] * 200), n=2000)
        self.assertEqual(result, paired_bootstrap(_vals(deltas), _vals([0.0] * 200), n=2000))
        low, high = result["ci95"]
        self.assertLess(low, true_mean)
        self.assertGreater(high, true_mean)
        self.assertLess(high - low, 0.15)
        self.assertAlmostEqual(result["mean_delta"], true_mean)

    def test_consistent_gain_is_detected_and_p_value_is_never_zero(self) -> None:
        result = paired_bootstrap(_vals([0.5] * 40), _vals([0.0] * 40), n=500)
        self.assertGreater(result["ci95"][0], 0)
        self.assertGreater(result["p_two_sided"], 0.0)
        self.assertEqual(retriever_verdict(result), "BEATS_RETRIEVER")
        self.assertEqual(result["changed_events"], 40)

    def test_no_effect_ci_contains_zero(self) -> None:
        rng = random.Random(3)
        deltas = [rng.choice([-0.3, 0.3]) for _ in range(300)]
        result = paired_bootstrap(_vals(deltas), _vals([0.0] * 300), n=2000)
        self.assertLess(result["ci95"][0], 0.0)
        self.assertGreater(result["ci95"][1], 0.0)
        self.assertGreater(result["p_two_sided"], 0.05)

    def test_verdict_does_not_hide_degradation_noise_or_trivial_gain(self) -> None:
        self.assertEqual(retriever_verdict({"ci95": [-0.05, -0.01]}), "WORSE_THAN_RETRIEVER")
        self.assertEqual(retriever_verdict({"ci95": [-0.05, 0.001]}), "NO_MEANINGFUL_GAIN")
        self.assertEqual(retriever_verdict({"ci95": [-0.02, 0.03]}), "INCONCLUSIVE")
        self.assertEqual(retriever_verdict({"ci95": [0.0001, 0.0005]}), "GAIN_BELOW_MEANINGFUL")
        self.assertEqual(
            retriever_verdict({"ci95": [0.0001, 0.003]}),
            "POSITIVE_GAIN_MEANINGFULNESS_UNCERTAIN",
        )

    def test_meaningful_ci_cannot_override_unrejected_null(self) -> None:
        contrast = {"ci95": [0.003, 0.02], "p_two_sided": 0.0625}
        self.assertEqual(retriever_verdict(contrast), "POSITIVE_CI_NULL_NOT_REJECTED")
        contrast["p_two_sided"] = 0.02
        self.assertEqual(retriever_verdict(contrast), "BEATS_RETRIEVER")
        contrast["ci95"] = [0.0001, 0.02]
        self.assertEqual(retriever_verdict(contrast), "POSITIVE_GAIN_MEANINGFULNESS_UNCERTAIN")

    def test_duplicate_events_are_rejected(self) -> None:
        row = _row("a", 1, ndcg=0.2)
        with self.assertRaises(ValueError):
            full_cohort_values([row], [row, row])
        with self.assertRaises(ValueError):
            full_cohort_values([row, row], [row])

    def test_repeated_events_do_not_create_independent_users(self) -> None:
        better = [{**_vals([0.3])[0], "cluster_id": str(i // 10)} for i in range(20)]
        base = [{**row, "ndcg@10": 0.0} for row in better]
        result = paired_bootstrap(better, base, n=1000)
        self.assertEqual(result["independent_clusters"], 2)
        self.assertEqual(result["events"], 20)
        self.assertEqual(result["resampling_unit"], "source_user")
        # Four equally likely null label assignments; both equal-sign assignments are extreme.
        self.assertEqual(result["p_two_sided"], 0.5)

    def test_partial_or_mismatched_cluster_metadata_fails(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap([{"ndcg@10": 0.3, "cluster_id": "a"}], [{"ndcg@10": 0.0}])
        with self.assertRaises(ValueError):
            paired_bootstrap(
                [{"ndcg@10": 0.3, "cluster_id": "a"}],
                [{"ndcg@10": 0.0, "cluster_id": "b"}],
            )

    def test_holm_is_monotone_and_capped(self) -> None:
        adjusted = holm_adjust({"x": 0.01, "y": 0.04})
        self.assertAlmostEqual(adjusted["x"], 0.02)
        self.assertAlmostEqual(adjusted["y"], 0.04)
        self.assertEqual(holm_adjust({"x": 0.9, "y": 0.95}), {"x": 1.0, "y": 1.0})


if __name__ == "__main__":
    unittest.main()
