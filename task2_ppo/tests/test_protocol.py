# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check that controlled PPO forks cannot silently change extra settings."""

from __future__ import annotations

from copy import deepcopy
import unittest

from common.data import load_yaml
from task2_ppo.core.trainer import validate_training_protocol
from task2_ppo.core.protocol import validate_shared_protocol


class ProtocolTests(unittest.TestCase):
    """Protect equal-budget, one-variable-at-a-time comparisons."""

    def setUp(self):
        self.config = load_yaml("configs/ppo.yaml")

    def test_released_conditions_are_valid(self):
        validate_training_protocol(self.config, "standard", 20, 0.20, 0.10)
        validate_training_protocol(self.config, "clip_eps_0.05", 8, 0.05, 0.10)
        validate_training_protocol(self.config, "kl_beta_0.00", 8, 0.20, 0.00)

    def test_clipping_fork_cannot_change_kl_beta(self):
        with self.assertRaises(ValueError):
            validate_training_protocol(
                self.config, "clip_eps_0.05", 8, 0.05, 0.20
            )

    def test_kl_fork_cannot_change_clip_epsilon(self):
        with self.assertRaises(ValueError):
            validate_training_protocol(
                self.config, "kl_beta_0.20", 8, 0.50, 0.20
            )

    def test_accidental_release_config_change_is_rejected(self):
        changed = deepcopy(self.config)
        changed["generation"]["top_p"] = 0.8
        with self.assertRaises(ValueError):
            validate_shared_protocol(changed)


if __name__ == "__main__":
    unittest.main()
