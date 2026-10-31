"""Small CPU checks for the public R4 entry points, not new paper experiments."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from prepare_public_inputs import prepare
from train_public import input_scales, normalized_loss
import train_pinn_balanced as worker


class PublicReproductionTests(unittest.TestCase):
    def test_rms_uses_only_inputs(self):
        values = np.ones((3, 2801), dtype=np.float64)
        values[:, :200] *= 2
        self.assertEqual(input_scales(values), {"g_mean_square": 4.0,
                                               "f_mean_square": 1.0})
        with self.assertRaises(ValueError):
            input_scales(np.ones((3, 5402)))
        values[0, 0] = np.nan
        with self.assertRaises(ValueError):
            input_scales(values)

    def test_objective_matches_frozen_formula(self):
        generator = torch.Generator().manual_seed(17)
        value, lap, f = [torch.randn(2, 13, generator=generator) for _ in range(3)]
        bv, g = [torch.randn(2, 7, generator=generator) for _ in range(2)]
        scales = {"f_mean_square": 0.123, "g_mean_square": 0.00008}
        for k in (0, 1, 10, 100):
            actual = normalized_loss(value, lap, bv, g, f, k, scales)
            expected = (0.9 * torch.mean((lap + k * value - f) ** 2) / 0.123
                        + 0.1 * torch.mean((bv - g) ** 2) / 0.00008)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_coordinate_jets_against_autograd(self):
        torch.manual_seed(11)
        trunk = torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.Tanh(),
                                    torch.nn.Linear(5, 3)).double()
        points = torch.rand(4, 2, dtype=torch.float64, requires_grad=True)
        value, dx, dy, lap = worker.trunk_jets(trunk, points)
        truth = trunk(points)
        torch.testing.assert_close(value, truth)
        for j in range(3):
            first = torch.autograd.grad(truth[:, j].sum(), points, create_graph=True)[0]
            second_x = torch.autograd.grad(first[:, 0].sum(), points, retain_graph=True)[0][:, 0]
            second_y = torch.autograd.grad(first[:, 1].sum(), points, retain_graph=True)[0][:, 1]
            torch.testing.assert_close(dx[:, j], first[:, 0])
            torch.testing.assert_close(dy[:, j], first[:, 1])
            torch.testing.assert_close(lap[:, j], second_x + second_y)

    def test_public_tuple_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated"
            generated.mkdir()
            tuples = np.random.default_rng(4).normal(size=(2, 5402))
            np.save(generated / "data.npy", tuples)
            np.savetxt(root / "mad.txt", tuples)
            prepare(generated, root / "mad.txt", root / "paired")
            inputs = np.load(root / "paired/inputs.npy")
            supervised = np.load(root / "paired/supervised.npy")
            self.assertEqual(inputs.shape, (2, 2801))
            np.testing.assert_array_equal(inputs, supervised[:, :2801])
            np.testing.assert_allclose(np.max(np.abs(inputs), axis=1), 1)
            with self.assertRaises(ValueError):
                worker.PINNDataset(root / "paired/supervised.npy", 2)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
