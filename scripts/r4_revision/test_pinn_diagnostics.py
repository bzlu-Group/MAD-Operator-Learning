"""Check interpolation, component summation, and common-scale invariance."""
import unittest

import numpy as np

from diagnose_pinn_inputs import components, source_interpolant
from generate_balanced import norms


class PINNDiagnosticTests(unittest.TestCase):
    def setUp(self):
        axis = np.linspace(0, 1, 51)
        self.y, self.x = np.meshgrid(axis, axis, indexing="ij")
        self.u = self.x**2+self.y**2
        self.g = np.concatenate((self.u[0, :-1], self.u[:-1, -1],
                                 self.u[-1, ::-1][:-1], self.u[::-1, 0][:-1]))[None]

    def test_source_interpolation_retains_input_nodes_and_orientation(self):
        f = (self.x+3*self.y+self.x*self.y)[None]
        fine = source_interpolant(f, 101)
        np.testing.assert_allclose(fine[:, ::2, ::2], f, atol=1e-13)

    def test_component_sum_matches_discrete_exact_polynomial(self):
        f = (4+3*self.u)[None]
        ug, uf, residual = components(self.g, f, 101, 3, workers=1, residual=True)
        np.testing.assert_allclose(ug+uf, self.u.reshape(1, -1), atol=2e-12)
        self.assertLess(float(residual[0]), 1e-10)

    def test_component_ratios_do_not_change_with_common_scale(self):
        f = (4+3*self.u)[None]
        ug, uf, _ = components(self.g, f, 101, 3, workers=1)
        scaled_g, scaled_f, _ = components(.1*self.g, .1*f, 101, 3, workers=1)
        np.testing.assert_allclose(norms(uf)/norms(ug+uf),
                                   norms(scaled_f)/norms(scaled_g+scaled_f), atol=1e-13)


if __name__ == "__main__":
    unittest.main()
