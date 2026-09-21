from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    try:
        from .models.interventional_visual_residual import InterventionalVisualResidual, assert_arm_parity, placebo_regularized_loss
    except ImportError:
        from models.interventional_visual_residual import InterventionalVisualResidual, assert_arm_parity, placebo_regularized_loss


@unittest.skipIf(torch is None, "PyTorch is required for intervention tests")
class InterventionalVisualResidualTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.context = torch.randn(2, 3)
        self.item_text = torch.randn(4, 5)
        self.visual = torch.randn(4, 2)
        self.text = torch.randn(2, 4)
        self.model = InterventionalVisualResidual(3, 5, 2, hidden_dim=8, interaction_dim=4)

    def test_zero_init_is_exact_text_control(self) -> None:
        self.assertTrue(torch.equal(self.model(self.text, self.context, self.item_text, self.visual), self.text))

    def test_aligned_gradient_is_live(self) -> None:
        targets = torch.tensor([1, 2])
        aligned = self.model(self.text, self.context, self.item_text, self.visual)
        shuffled = self.model(self.text, self.context, self.item_text, self.visual.flip(0))
        no_image = self.model(self.text, self.context, self.item_text, torch.zeros_like(self.visual))
        placebo_regularized_loss(self.text, aligned, shuffled, no_image, targets).backward()
        self.assertGreater(float(self.model.output.weight.grad.abs().sum()), 0.0)

    def test_parity_rejects_changed_availability(self) -> None:
        rows = torch.tensor([10, 11]); targets = torch.tensor([1, 2]); text = self.text.clone(); availability = torch.ones(4); changed = availability.clone(); changed[0] = 0
        with self.assertRaisesRegex(ValueError, "availability"):
            assert_arm_parity(rows, targets, text, availability, ("shuffle", rows, targets, text, changed))

    def test_placebo_penalty_is_zero_for_null_arms(self) -> None:
        targets = torch.tensor([1, 2]); text = torch.zeros(2, 4); aligned = torch.tensor([[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]])
        value = placebo_regularized_loss(text, aligned, text, text, targets)
        self.assertLess(float(value), 0.25)


if __name__ == "__main__": unittest.main()
