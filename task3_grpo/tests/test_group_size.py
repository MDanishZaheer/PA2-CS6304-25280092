# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Tests for the fixed-cache equal-budget group-size study."""

import unittest

from common.data import load_yaml
from task3_grpo.analyze_group_size import (
    assign_difficulty_bins,
    load_k8_cache,
    regroup_equal_generation_budget,
)


class GroupSizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_yaml("configs/grpo.yaml")
        cls.prompts = load_k8_cache(config["group_cache"])

    def test_released_cache_has_complete_k8_groups(self):
        self.assertEqual(len(self.prompts), 24)
        self.assertEqual({len(rows) for rows in self.prompts.values()}, {8})

    def test_every_k_uses_same_192_completions(self):
        for k, expected_group_count in {2: 96, 4: 48, 8: 24}.items():
            groups = regroup_equal_generation_budget(self.prompts, k)
            self.assertEqual(len(groups), expected_group_count)
            self.assertEqual(sum(len(group) for group in groups), 192)
            self.assertEqual({len(group) for group in groups}, {k})

    def test_difficulty_rule_has_three_nonempty_bins(self):
        mapping, definition = assign_difficulty_bins(self.prompts)
        self.assertEqual(set(mapping.values()), {"hard", "medium", "easy"})
        self.assertEqual(len(mapping), 24)
        self.assertEqual(len(definition["prompt_assignments"]), 24)


if __name__ == "__main__":
    unittest.main()
