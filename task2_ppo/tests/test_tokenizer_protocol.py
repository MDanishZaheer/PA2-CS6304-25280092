# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check that over-budget causal prompts preserve their most recent tokens."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from common.models import load_tokenizer


class TokenizerProtocolTests(unittest.TestCase):
    """Protect the left-truncation setting used by every Task 2 stage."""

    def test_policy_tokenizer_uses_left_truncation(self):
        tokenizer = SimpleNamespace(
            pad_token=None,
            eos_token="<eos>",
            padding_side="right",
            truncation_side="right",
        )
        with patch(
            "common.models.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ):
            loaded = load_tokenizer("offline-model")
        self.assertEqual(loaded.pad_token, "<eos>")
        self.assertEqual(loaded.padding_side, "left")
        self.assertEqual(loaded.truncation_side, "left")


if __name__ == "__main__":
    unittest.main()
