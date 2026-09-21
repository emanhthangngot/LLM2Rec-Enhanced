from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    try:
        from .models.score_visual_fusion import VisualScoreResidual
    except ImportError:
        from models.score_visual_fusion import VisualScoreResidual
else:
    VisualScoreResidual = None


@unittest.skipIf(torch is None, "PyTorch is required for score-fusion tests")
class VisualScoreResidualTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2**-0.5, 2**-0.5],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.availability = torch.tensor([0, 1, 1, 1, 0], dtype=torch.float32)
        self.module = VisualScoreResidual(self.catalog, self.availability)

    def test_zero_alpha_is_exact_text_control(self) -> None:
        text_logits = torch.randn(2, 5)
        item_seqs = torch.tensor([[1, 2, 0], [3, 1, 2]], dtype=torch.long)
        observed = self.module(text_logits, item_seqs)
        self.assertTrue(torch.equal(observed, text_logits))

    def test_history_uses_recency_weights_and_ignores_padding(self) -> None:
        history = self.module.history_visual(torch.tensor([[1, 2, 0]], dtype=torch.long))
        expected = torch.tensor([[1.0, 2.0]])
        expected = expected / expected.norm(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(history, expected))

    def test_missing_history_rows_are_ignored(self) -> None:
        observed = self.module.history_visual(torch.tensor([[4, 1, 0]], dtype=torch.long))
        self.assertTrue(torch.allclose(observed, torch.tensor([[1.0, 0.0]])))
        empty = self.module.history_visual(torch.tensor([[4, 0, 0]], dtype=torch.long))
        self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))

    def test_visual_logits_are_standardized_over_non_padding_catalog(self) -> None:
        logits = self.module.standardized_visual_logits(torch.tensor([[1, 2, 0]], dtype=torch.long))
        candidates = logits[:, 1:]
        self.assertTrue(torch.allclose(candidates.mean(dim=1), torch.zeros(1), atol=1e-6))
        self.assertTrue(torch.allclose(candidates.std(dim=1, unbiased=False), torch.ones(1), atol=1e-6))
        self.assertEqual(float(logits[0, 0]), 0.0)

    def test_alpha_receives_gradient_without_changing_text_path(self) -> None:
        text_logits = torch.randn(1, 5, requires_grad=True)
        item_seqs = torch.tensor([[1, 2, 0]], dtype=torch.long)
        self.module(text_logits, item_seqs)[0, 1].backward()
        self.assertIsNotNone(self.module.alpha.grad)
        self.assertGreater(abs(float(self.module.alpha.grad)), 0)
        expected_text_grad = torch.zeros_like(text_logits)
        expected_text_grad[0, 1] = 1
        self.assertTrue(torch.equal(text_logits.grad, expected_text_grad))

    def test_unavailable_visual_rows_must_be_zero(self) -> None:
        invalid = self.catalog.clone()
        invalid[4] = 1
        with self.assertRaises(ValueError):
            VisualScoreResidual(invalid, self.availability)


if __name__ == "__main__":
    unittest.main()
