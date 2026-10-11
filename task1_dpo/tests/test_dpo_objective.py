# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check the DPO equation and reference-relative preference accuracy."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as functional

from task1_dpo.dpo import dpo_loss


class DpoObjectiveTests(unittest.TestCase):
    """Cover the starter defect and the zero-drift boundary case."""

    def test_identical_policy_and_reference_have_zero_margin(self):
        chosen = torch.tensor([-2.0, -5.0])
        rejected = torch.tensor([-4.0, -3.0])
        loss, details = dpo_loss(chosen, rejected, chosen, rejected, 0.1)
        self.assertAlmostEqual(float(loss), math.log(2), places=6)
        self.assertEqual(float(details["preference_accuracy"]), 0.0)
        self.assertEqual(float(details["preference_margin_mean"]), 0.0)

    def test_reference_margin_is_subtracted(self):
        policy_chosen = torch.tensor([-1.0], requires_grad=True)
        policy_rejected = torch.tensor([-4.0], requires_grad=True)
        loss, details = dpo_loss(
            policy_chosen, policy_rejected,
            torch.tensor([-2.0]), torch.tensor([-4.0]), 0.1,
        )
        expected = -functional.logsigmoid(torch.tensor(0.1))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(float(details["preference_accuracy"]), 1.0)
        loss.backward()
        self.assertLess(float(policy_chosen.grad), 0.0)
        self.assertGreater(float(policy_rejected.grad), 0.0)

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            dpo_loss(torch.zeros(2), torch.zeros(1), torch.zeros(2), torch.zeros(2), 0.1)


if __name__ == "__main__":
    unittest.main()
