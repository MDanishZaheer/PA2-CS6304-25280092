# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check the corrected PPO surrogate, GAE, shaping, and masking."""

from __future__ import annotations

import unittest

import torch

from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
)


class PpoObjectiveTests(unittest.TestCase):
    """Protect the assignment's deliberately incorrect maximum objective."""

    def test_positive_advantage_uses_clipped_minimum(self):
        old_logprobs = torch.zeros(1, 1)
        new_logprobs = torch.log(torch.tensor([[1.5]]))
        advantages = torch.ones(1, 1)
        mask = torch.ones(1, 1)
        loss, _, clip_fraction, affected_fraction = ppo_policy_loss(
            new_logprobs, old_logprobs, advantages, mask, eps=0.2
        )
        self.assertAlmostEqual(float(loss), -1.2, places=6)
        self.assertEqual(float(clip_fraction), 1.0)
        self.assertEqual(float(affected_fraction), 1.0)

    def test_negative_advantage_clips_a_large_probability_drop(self):
        old_logprobs = torch.zeros(1, 1)
        new_logprobs = torch.log(torch.tensor([[0.5]]))
        advantages = -torch.ones(1, 1)
        mask = torch.ones(1, 1)
        loss, _, _, affected_fraction = ppo_policy_loss(
            new_logprobs, old_logprobs, advantages, mask, eps=0.2
        )
        self.assertAlmostEqual(float(loss), 0.8, places=6)
        self.assertEqual(float(affected_fraction), 1.0)

    def test_gae_propagates_terminal_reward_backwards(self):
        rewards = torch.tensor([[0.0, 1.0, 9.0]])
        values = torch.zeros_like(rewards)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        advantages, returns = compute_gae(
            rewards, values, mask, gamma=1.0, lam=1.0
        )
        expected = torch.tensor([[1.0, 1.0, 0.0]])
        self.assertTrue(torch.allclose(advantages, expected))
        self.assertTrue(torch.allclose(returns, expected))

    def test_reward_is_added_only_to_final_valid_token(self):
        policy_logprobs = torch.tensor([[2.0, 3.0, 5.0]])
        reference_logprobs = torch.tensor([[1.0, 1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        rewards = shaped_rewards(
            torch.tensor([4.0]),
            policy_logprobs,
            reference_logprobs,
            mask,
            beta_kl=0.5,
        )
        self.assertTrue(torch.allclose(rewards, torch.tensor([[-0.5, 3.0, 0.0]])))

    def test_advantage_normalization_ignores_padding(self):
        advantages = torch.tensor([[1.0, 3.0, 100.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        normalized = normalize_advantages(advantages, mask)
        self.assertTrue(torch.allclose(normalized, torch.tensor([[-1.0, 1.0, 0.0]])))


if __name__ == "__main__":
    unittest.main()
