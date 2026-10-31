"""Checks for the continuum field and response-balancing contract."""
import unittest
import numpy as np
from generate_balanced import basis, field, norms


class BalancedFieldTests(unittest.TestCase):
    def test_same_field_on_nested_grids(self):
        coefficients = np.random.default_rng(42).standard_normal((2, 49, 49))
        np.testing.assert_allclose(field(coefficients, basis(101))[:, ::2, ::2],
                                   field(coefficients, basis(51)), rtol=1e-12, atol=1e-12)

    def test_response_scaling(self):
        rng = np.random.default_rng(43)
        uf, ug = rng.normal(size=(5, 51, 51)), rng.normal(size=(5, 51, 51))
        ratio = np.exp(rng.uniform(np.log(0.5), np.log(2), 5))
        alpha = ratio*norms(ug)/norms(uf)
        np.testing.assert_allclose(norms(uf*alpha[:, None, None])/norms(ug), ratio)

    def test_constant_pointwise_prior_variance(self):
        np.testing.assert_allclose(np.sum(basis(51)**2, axis=1), 1, atol=1e-14)

    def test_high_modes_attenuated(self):
        self.assertLess(np.exp(-0.5*0.1**2*(np.pi*24)**2), 1e-12)


if __name__ == "__main__":
    unittest.main()
