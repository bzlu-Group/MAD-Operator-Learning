"""Generate resolved, response-balanced source/boundary data for MAD R4."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

R3 = Path(__file__).resolve().parents[1] / "r3_revision"
sys.path.insert(0, str(R3))


def basis(points: int, length: float = 0.1, modes: int = 24) -> np.ndarray:
    x = np.linspace(0.0, 1.0, points)
    frequency = np.pi * np.arange(1, modes + 1)
    weights = np.exp(-0.25 * length**2 * frequency**2)
    result = np.column_stack((np.ones(points),
                              np.cos(x[:, None] * frequency) * weights,
                              np.sin(x[:, None] * frequency) * weights))
    return result / np.sqrt(1.0 + np.sum(weights**2))


def field(coefficients: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ym,bmn,xn->byx", b, coefficients, b, optimize=True)


def norms(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values.reshape(len(values), -1), axis=1)


def stats(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "median": float(np.median(values)),
            "max": float(np.max(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95))}


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def generate(args: argparse.Namespace) -> None:
    from generate_helmholtz_datasets import FastHelmholtzSolver, sample_boundary_paths

    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    n, fine, coarse = args.samples, args.fine, 51
    if (fine - 1) % (coarse - 1):
        raise ValueError("The reference grid must contain the retained 51-point grid.")
    stride = (fine - 1) // (coarse - 1)
    rng = np.random.default_rng(args.seed)
    boundary_start = time.perf_counter()
    boundary = sample_boundary_paths(rng, n, fine, 0.1)
    boundary_seconds = time.perf_counter() - boundary_start
    coefficients = rng.standard_normal((n, 49, 49))
    requested_ratio = np.exp(rng.uniform(np.log(0.5), np.log(2.0), n))
    b = basis(fine)
    solver = FastHelmholtzSolver(fine, args.k, args.workers)
    arrays = {}
    for name, shape in (("data", (n, 5402)), ("uf", (n, 2601)),
                        ("ug", (n, 2601))):
        arrays[name] = np.lib.format.open_memmap(root / f"{name}.npy", mode="w+",
                                               dtype=np.float64, shape=shape)
    scales = np.empty(n)
    alpha = np.empty(n)
    observations = {key: [] for key in ("uf_over_u", "ug_over_u", "uf_over_ug",
                    "raw_max_u", "normalized_max_u", "source_lag1_x",
                    "source_lag1_y", "direct_recombination_relative")}
    field_seconds = solve_seconds = 0.0
    for first in range(0, n, args.batch):
        last = min(first + args.batch, n)
        tic = time.perf_counter()
        raw_source = field(coefficients[first:last], b)
        field_seconds += time.perf_counter() - tic
        tic = time.perf_counter()
        ug = solver.solve(boundary[first:last], np.zeros_like(raw_source))
        uf0 = solver.solve(np.zeros_like(boundary[first:last]), raw_source)
        coarse_ug, coarse_uf0 = ug[:, ::stride, ::stride], uf0[:, ::stride, ::stride]
        alpha[first:last] = requested_ratio[first:last] * norms(coarse_ug) / norms(coarse_uf0)
        source = raw_source * alpha[first:last, None, None]
        uf = uf0 * alpha[first:last, None, None]
        u = ug + uf
        # The diagnostic confirms that the two-part construction solves the full problem.
        direct = solver.solve(boundary[first:last], source)
        solve_seconds += time.perf_counter() - tic
        cf, cg = uf[:, ::stride, ::stride], ug[:, ::stride, ::stride]
        cu, cs = u[:, ::stride, ::stride], source[:, ::stride, ::stride]
        cb = boundary[first:last, ::stride]
        scale = np.maximum.reduce((np.max(np.abs(cb), axis=1),
                 np.max(np.abs(cs.reshape(last-first, -1)), axis=1),
                 np.max(np.abs(cu.reshape(last-first, -1)), axis=1)))
        scales[first:last] = scale
        arrays["data"][first:last] = np.concatenate(
            (cb, cs.reshape(last-first, -1), cu.reshape(last-first, -1)), axis=1) / scale[:, None]
        arrays["uf"][first:last] = cf.reshape(last-first, -1) / scale[:, None]
        arrays["ug"][first:last] = cg.reshape(last-first, -1) / scale[:, None]
        nu, nf, ng = norms(cu), norms(cf), norms(cg)
        for key, value in (("uf_over_u", nf/nu), ("ug_over_u", ng/nu),
                           ("uf_over_ug", nf/ng),
                           ("raw_max_u", np.max(np.abs(cu.reshape(last-first, -1)), axis=1)),
                           ("normalized_max_u", np.max(np.abs(cu.reshape(last-first, -1)), axis=1)/scale),
                           ("direct_recombination_relative", norms(direct-u)/norms(u))):
            observations[key].extend(value.tolist())
        for sample in cs:
            observations["source_lag1_x"].append(float(np.corrcoef(sample[:, :-1].ravel(), sample[:, 1:].ravel())[0, 1]))
            observations["source_lag1_y"].append(float(np.corrcoef(sample[:-1].ravel(), sample[1:].ravel())[0, 1]))
        for array in arrays.values():
            array.flush()
        if first % 100 == 0:
            print(f"k={args.k:g} generated {last}/{n}", flush=True)
    np.savez(root / "latent.npz", coefficients=coefficients, alpha=alpha,
             boundary=boundary, scales=scales, requested_ratio=requested_ratio)
    summary = {"schema": "MAD-R4-balanced-source-v1", "samples": n, "seed": args.seed,
               "k": args.k, "fine_points": fine, "coarse_points": coarse,
               "source": {"family": "real Gaussian Fourier field on period-2 square",
                          "physical_length": 0.1, "max_mode_each_axis": 24,
                          "amplitude_weights": "exp(-ell^2*(pi*m)^2/4)",
                          "coefficient_law": "independent N(0,1), 49x49 real coefficients"},
               "response_balance": {"ratio_uf_over_ug": "log-uniform[0.5,2]",
                                    "scale_grid": "retained 51x51",
                                    "input_shapes_independent": True,
                                    "input_amplitudes_independent": False},
               "normalization": "shared maximum of retained (g,f,u) tuple, unchanged R3 rule",
               "timing": {"complete_generation_seconds": time.perf_counter()-start,
                          "boundary_sampling_seconds": boundary_seconds,
                          "field_evaluation_seconds": field_seconds,
                          "three_component_and_full_solves_seconds": solve_seconds,
                          "includes_component_diagnostics": True},
               "metrics": {key: stats(np.asarray(value)) for key, value in observations.items()},
               "runtime": {"numpy": np.__version__, "pid": os.getpid()},
               "files": {p.name: sha(p) for p in root.iterdir() if p.is_file()}}
    if not np.allclose(np.asarray(observations["uf_over_ug"]), requested_ratio, rtol=1e-10, atol=1e-12):
        raise AssertionError("Response balance failed")
    if summary["metrics"]["direct_recombination_relative"]["max"] > 1e-9:
        raise AssertionError("Direct solve differs from component sum")
    write_json(root / "summary.json", summary)
    print(json.dumps({"output": str(root), "samples": n, "k": args.k,
                      "uf_over_u": summary["metrics"]["uf_over_u"]}), flush=True)


def refine(root: Path, count: int, workers: int) -> None:
    from generate_helmholtz_datasets import FastHelmholtzSolver

    metadata = json.loads((root / "summary.json").read_text())
    latent = np.load(root / "latent.npz")
    fine, k = metadata["fine_points"], metadata["k"]
    native = np.load(root / "data.npy", mmap_mode="r")[:, -2601:]
    reference_errors, reconstruction_errors = [], []
    for index in range(min(count, metadata["samples"])):
        boundary = latent["boundary"][index]
        boundary_t = np.linspace(0.0, 1.0, len(boundary)+1)
        interpolant = CubicSpline(boundary_t, np.append(boundary, boundary[0]))
        coefficients = latent["coefficients"][index:index+1]
        alpha, scale = latent["alpha"][index], latent["scales"][index]
        grids = {}
        for points in (501, 1001, 2001):
            values = field(coefficients, basis(points)) * alpha
            gp = interpolant(np.arange(4*(points-1))/(4*(points-1)))[None, :]
            solver = FastHelmholtzSolver(points, k, workers)
            u = solver.solve(gp, values)
            grids[points] = u[:, ::(points-1)//50, ::(points-1)//50].reshape(-1)/scale
        reference_errors.append({"sample": index,
            "501_to_1001": float(np.linalg.norm(grids[501]-grids[1001])/np.linalg.norm(grids[1001])),
            "1001_to_2001": float(np.linalg.norm(grids[1001]-grids[2001])/np.linalg.norm(grids[2001])),
            "stored_to_recomputed_1001": float(np.linalg.norm(grids[1001]-native[index])/np.linalg.norm(native[index]))})
        exact_source = field(coefficients, basis(fine)) * alpha
        coarse_source = field(coefficients, basis(51))[0] * alpha
        axis = np.linspace(0, 1, 51)
        interpolated = RectBivariateSpline(axis, axis, coarse_source)(
            np.linspace(0, 1, fine), np.linspace(0, 1, fine))[None]
        solver = FastHelmholtzSolver(fine, k, workers)
        zero_boundary = np.zeros((1, 4*(fine-1)))
        exact_uf = solver.solve(zero_boundary, exact_source)[:, ::(fine-1)//50, ::(fine-1)//50]
        interpolated_uf = solver.solve(zero_boundary, interpolated)[:, ::(fine-1)//50, ::(fine-1)//50]
        reconstruction_errors.append(float(norms(interpolated_uf-exact_uf)[0]/norms(exact_uf)[0]))
    report = {"samples": len(reference_errors), "rows": reference_errors,
              "boundary_definition": "fixed cubic interpolant through original fine-grid GP samples",
              "source_coarse_reconstruction_relative": stats(np.asarray(reconstruction_errors)),
              "reference_1001_to_2001": stats(np.asarray([r["1001_to_2001"] for r in reference_errors])),
              "gate_limit": 0.001}
    report["passed"] = (report["reference_1001_to_2001"]["max"] < 0.001
                         and report["source_coarse_reconstruction_relative"]["max"] < 0.001)
    write_json(root / "refinement.json", report)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Numerical reference/input resolution gate failed; preserve results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=float, default=1)
    parser.add_argument("--fine", type=int, default=1001)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refine-count", type=int, default=10)
    args = parser.parse_args()
    refine(args.output, args.refine_count, args.workers) if args.refine else generate(args)
