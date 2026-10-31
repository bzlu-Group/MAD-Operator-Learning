"""Diagnose archived PINN inputs using explicitly interpolated reference fields.

These files contain g and f, not solution labels. No model is trained or
selected; the reference decomposition is a numerical input diagnostic.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from threadpoolctl import threadpool_limits

from diagnose_archived_tuples import boundary_interpolant, FastHelmholtzSolver
from generate_balanced import norms, sha, stats, write_json


def source_interpolant(source, points, degree=3):
    axis = np.linspace(0.0, 1.0, 51)
    fine = np.linspace(0.0, 1.0, points)
    return np.array([RectBivariateSpline(axis, axis, row, kx=degree, ky=degree)(fine, fine)
                     for row in source.reshape(-1, 51, 51)])


def components(boundary, source, points, k, workers=4,
               boundary_degree=3, source_degree=3, residual=False):
    if (points-1) % 50:
        raise ValueError("Reference grids must contain the retained 51-point grid")
    solver = FastHelmholtzSolver(points, k, workers)
    fine_g = boundary_interpolant(boundary, points, boundary_degree)
    zeros = np.zeros((len(boundary), points, points))
    stride = (points-1)//50
    ug = solver.solve(fine_g, zeros)[:, ::stride, ::stride].reshape(len(boundary), -1)
    fine_f = source_interpolant(source, points, source_degree)
    full_uf = solver.solve(np.zeros_like(fine_g), fine_f)
    uf = full_uf[:, ::stride, ::stride].reshape(len(boundary), -1)
    relative_residual = None
    if residual:
        center = full_uf[:, 1:-1, 1:-1]
        laplacian = (full_uf[:, 2:, 1:-1]+full_uf[:, :-2, 1:-1]
                     +full_uf[:, 1:-1, 2:]+full_uf[:, 1:-1, :-2]-4*center)/solver.spacing**2
        rhs = fine_f[:, 1:-1, 1:-1]
        relative_residual = norms(laplacian+k*center-rhs)/np.maximum(norms(rhs), np.finfo(float).tiny)
    return ug, uf, relative_residual


def diagnose(original, output, k, points=1001, batch=4, workers=4, expected_count=2000):
    if output.exists():
        raise FileExistsError(output)
    rows = np.loadtxt(original)
    if rows.shape != (expected_count, 2801) or not np.isfinite(rows).all():
        raise ValueError("Expected finite [g(200), f(2601)] archived PINN inputs")
    if not np.allclose(np.max(np.abs(rows), axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Archived input maximum normalization differs from the stated rule")
    boundary, source = rows[:, :200], rows[:, 200:]
    check_count = min(10, len(rows))
    observations = {key: [] for key in ("uf_over_u", "ug_over_u", "input_scaled_max_u")}
    checks = {key: [] for key in ("501_to_1001_total_over_u", "1001_to_2001_total_over_u",
                                  "1001_to_2001_source_over_uf",
                                  "cubic_to_quintic_boundary_total_over_u",
                                  "cubic_to_quintic_source_total_over_u", "source_relative_residual")}
    for first in range(0, len(rows), batch):
        last = min(first+batch, len(rows))
        g, f = boundary[first:last], source[first:last]
        ug, uf, residual = components(g, f, points, k, workers, residual=first < check_count)
        total = ug+uf
        denominator = norms(total)
        if np.any(denominator == 0):
            raise ValueError("The reconstructed total solution has zero norm")
        observations["uf_over_u"].extend((norms(uf)/denominator).tolist())
        observations["ug_over_u"].extend((norms(ug)/denominator).tolist())
        observations["input_scaled_max_u"].extend(np.max(np.abs(total), axis=1).tolist())
        if first < check_count:
            count = min(last, check_count)-first
            checks["source_relative_residual"].extend(residual[:count].tolist())
            for name, fine, gd, fd in (("501_to_1001_total_over_u", 501, 3, 3),
                                       ("1001_to_2001_total_over_u", 2001, 3, 3),
                                       ("cubic_to_quintic_boundary_total_over_u", 1001, 5, 3),
                                       ("cubic_to_quintic_source_total_over_u", 1001, 3, 5)):
                other_g, other_f, _ = components(g[:count], f[:count], fine, k, workers, gd, fd)
                checks[name].extend((norms(other_g+other_f-total[:count])/denominator[:count]).tolist())
                if fine == 2001:
                    checks["1001_to_2001_source_over_uf"].extend(
                        (norms(other_f-uf[:count])/norms(uf[:count])).tolist())
        if first % 100 == 0 or last == len(rows):
            print(f"PINN k={k:g}: diagnosed {last}/{len(rows)}", flush=True)
    report = {
        "original": str(original), "original_sha256": sha(original), "samples": len(rows), "k": k,
        "schema": "MAD-R4-archived-PINN-input-diagnostic-v1",
        "provenance": "archived PINN input file at the repository Table 1 training path; no new random ensemble",
        "normalization": "original shared max over g and f; reconstructed u is not renormalized or used for training",
        "reference": {"points": points, "dtype": "float64", "operator": "Delta u + k u = f",
                      "solver": "five-point finite differences diagonalized by DST-I",
                      "boundary_interpolation": "edgewise cubic spline through 51 values per edge",
                      "source_interpolation": "tensor-product cubic spline through the 51-by-51 source values"},
        "limitations": "diagnostics of the stated interpolated input fields, not exact solution labels or replay of unavailable continuous fields; file names alone do not verify every historical run's input provenance",
        "reference_check_count": check_count,
        "reference_checks": {key: stats(np.array(value)) for key, value in checks.items()},
        "metrics": {key: stats(np.array(value)) for key, value in observations.items()},
        "per_sample": observations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        for k, name in ((0, "PINNpoisson2D_(2000, 51).txt"),
                        (1, "PINN1helmholtz2D_(2000, 51).txt"),
                        (10, "PINN10helmholtz2D_(2000, 51).txt"),
                        (100, "PINN100helmholtz2D_(2000, 51).txt")):
            diagnose(args.data_root/name, args.output/f"pinn_k{k}.json", k, workers=args.workers)
