"""Bounded serialization checks; no exploit or untrusted checkpoint is executed."""
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import checkpoint_io as io


class CheckpointIOTests(unittest.TestCase):
    def test_plain_weights_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            expected = {"weight": torch.arange(4, dtype=torch.float32)}
            torch.save(expected, path)
            actual = io.load_checkpoint(path)
            torch.testing.assert_close(actual["weight"], expected["weight"])

    def test_default_is_restricted_and_cpu_only(self):
        with patch.object(torch, "load", return_value={}) as load:
            io.load_checkpoint("unused.pth")
            load.assert_called_once_with("unused.pth", map_location="cpu", weights_only=True)

    def test_restricted_rejection_never_falls_back(self):
        with patch.object(torch, "load", side_effect=pickle.UnpicklingError("rejected")) as load:
            with self.assertRaises(pickle.UnpicklingError):
                io.load_checkpoint("unused.pth")
            self.assertEqual(load.call_count, 1)
            self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_trusted_full_state_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "own_resume.pth"
            torch.save({"weights":torch.ones(2), "rng":np.random.RandomState(3).get_state()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                io.load_checkpoint(path)
            with self.assertWarns(RuntimeWarning):
                result = io.load_checkpoint(path, trusted=True)
            np.testing.assert_array_equal(result["rng"][1],np.random.RandomState(3).get_state()[1])

    def test_resume_denied_without_trust_flag(self):
        with self.assertRaisesRegex(ValueError, "trust-checkpoint"):
            io.require_trusted_resume(False)

    def test_unsupported_torch_refused_before_loading(self):
        with patch.object(torch, "__version__", "2.5.1+cu121"), patch.object(torch, "load") as load:
            with self.assertRaisesRegex(RuntimeError, "supported PyTorch"):
                io.load_checkpoint("unused.pth", trusted=True)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
