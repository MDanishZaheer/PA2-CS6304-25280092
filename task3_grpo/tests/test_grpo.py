# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Tests for within-group advantages and both GRPO normalizations."""

import unittest

import torch

from task3_grpo.grpo import (
    gradient_allocation_proxy,
    grpo_policy_loss,
    group_relative_advantages,
    mask_truncated_sequences,
)


class GrpoObjectiveTests(unittest.TestCase):
    def test_advantages_are_computed_inside_each_prompt_group(self):
        rewards = torch.tensor([1.0, 3.0, 100.0, 100.0])
        group_ids = torch.tensor([0, 0, 1, 1])
        actual = group_relative_advantages(rewards, group_ids)
        torch.testing.assert_close(actual, torch.tensor([-1.0, 1.0, 0.0, 0.0]))

    def test_each_informative_group_has_standardized_advantages(self):
        rewards = torch.tensor([1.0, 2.0, 6.0, -2.0, 4.0, 8.0])
        group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        advantages = group_relative_advantages(rewards, group_ids)
        for group_id in (0, 1):
            values = advantages[group_ids == group_id]
            self.assertAlmostEqual(float(values.mean()), 0.0, places=6)
            self.assertAlmostEqual(float(values.var(unbiased=False)), 1.0, places=6)

    def test_canonical_and_dr_use_expected_denominators(self):
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        zeros = torch.zeros_like(mask)
        advantages = torch.ones(2)
        canonical, _ = grpo_policy_loss(
            zeros, zeros, advantages, mask, zeros, 0.2, 0.0, "grpo", 4
        )
        dr_grpo, _ = grpo_policy_loss(
            zeros, zeros, advantages, mask, zeros, 0.2, 0.0, "dr_grpo", 4
        )
        self.assertAlmostEqual(float(canonical), -1.0)
        self.assertAlmostEqual(float(dr_grpo), -0.75)

    def test_gradient_proxy_exposes_length_allocation(self):
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        advantages = torch.ones(2)
        _, canonical = gradient_allocation_proxy(advantages, mask, "grpo", 4)
        _, dr_grpo = gradient_allocation_proxy(advantages, mask, "dr_grpo", 4)
        torch.testing.assert_close(canonical, torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(dr_grpo, torch.tensor([0.5, 1.0]))

    def test_truncated_completions_are_fully_masked(self):
        actual = mask_truncated_sequences(torch.ones(3, 4), [False, True, False])
        self.assertEqual(actual.sum(dim=1).tolist(), [4.0, 0.0, 4.0])
        per_token, per_sequence = gradient_allocation_proxy(
            torch.ones(3), actual, "grpo", 4
        )
        self.assertEqual(float(per_token[1]), 0.0)
        self.assertEqual(float(per_sequence[1]), 0.0)

    def test_invalid_group_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.tensor([1.0, 2.0]), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
