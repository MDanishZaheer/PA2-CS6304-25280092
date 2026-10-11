# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check atomic, versioned PPO update checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from task2_ppo.core.checkpoints import (
    directory_sha256,
    latest_update_checkpoint,
    save_update_checkpoint,
)


class TinySavedModel:
    """Write the minimum files needed by the checkpoint hash validator."""

    def __init__(self, content: bytes):
        self.content = content

    def save_pretrained(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.safetensors").write_bytes(self.content)


class CheckpointTests(unittest.TestCase):
    """Ensure resume selects only the highest fully published update."""

    def test_latest_complete_update_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = TinySavedModel(b"policy")
            critic = TinySavedModel(b"critic")
            save_update_checkpoint(root, 1, model, critic, {"next_update": 1})
            save_update_checkpoint(root, 2, model, critic, {"next_update": 2})
            checkpoint, state = latest_update_checkpoint(root)
            self.assertEqual(checkpoint.name, "update_002")
            self.assertEqual(state["next_update"], 2)

    def test_checkpoint_hash_includes_configuration_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_model.safetensors").write_bytes(b"weights")
            (root / "adapter_config.json").write_text("one", encoding="utf-8")
            first = directory_sha256(root)
            (root / "adapter_config.json").write_text("two", encoding="utf-8")
            self.assertNotEqual(first, directory_sha256(root))


if __name__ == "__main__":
    unittest.main()
