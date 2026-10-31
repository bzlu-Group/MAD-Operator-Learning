"""Estimate components of archived tuples without inventing generator seeds.

Only retained boundary values are available for these historical files. The
boundary-driven component therefore uses an explicitly documented interpolant;
the source-driven component is the archived solution minus that component.
"""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import make_interp_spline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_revision"))
from generate_helmholtz_datasets import FastHelmholtzSolver, perimeter_edges
from generate_balanced import norms, stats, write_json


def boundary_interpolant(boundary, fine_points, degree=3):
    axis = np.linspace(0.0, 1.0, 51)
    fine_axis = np.linspace(0.0, 1.0, fine_points)
    edges = [make_interp_spline(axis, edge, k=degree, axis=-1)(fine_axis)
             for edge in perimeter_edges(boundary, 51)]
    bottom, right, top, left = edges
    return np.concatenate((bottom[:, :-1], right[:, :-1], top[:, ::-1][:, :-1],
                           left[:, ::-1][:, :-1]), axis=1)


def boundary_response(boundary, fine_points, k, degree=3):
    if (fine_points-1) % 50:
        raise ValueError("The fine grid must contain the original 51-by-51 points")
    solver = FastHelmholtzSolver(fine_points, k, 4)
    fine_g = boundary_interpolant(boundary, fine_points, degree)
    zero = np.zeros((len(boundary), fine_points, fine_points))
    stride = (fine_points-1)//50
    return solver.solve(fine_g, zero)[:, ::stride, ::stride].reshape(len(boundary), -1)


def run(args):
    rows = np.load(args.original) if args.original.suffix == ".npy" else np.loadtxt(args.original)
    if rows.ndim != 2 or rows.shape[1] != 5402 or not np.isfinite(rows).all():
        raise ValueError("Expected finite [g(200), f(2601), u(2601)] tuples")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    boundary, truth = rows[:, :200], rows[:, -2601:]
    denominator = norms(truth)
    if np.any(denominator == 0):
        raise ValueError("A relative component ratio is undefined for a zero solution")
    observations = {key: [] for key in ("uf_over_u", "ug_over_u", "normalized_max_u")}
    check_count = min(10, len(rows))
    checks = {key: [] for key in ("501_to_1001_over_u", "1001_to_2001_over_u",
                                  "cubic_to_quintic_boundary_over_u")}
    for first in range(0, len(rows), 4):
        last = min(first+4, len(rows))
        ug = boundary_response(boundary[first:last], 1001, args.k)
        uf = truth[first:last]-ug
        observations["uf_over_u"].extend((norms(uf)/denominator[first:last]).tolist())
        observations["ug_over_u"].extend((norms(ug)/denominator[first:last]).tolist())
        observations["normalized_max_u"].extend(np.max(np.abs(truth[first:last]), axis=1).tolist())
        if first < check_count:
            count = min(last, check_count)-first
            g = boundary[first:first+count]
            base = ug[:count]
            norm = denominator[first:first+count]
            for name, points, degree in (("501_to_1001_over_u", 501, 3),
                                         ("1001_to_2001_over_u", 2001, 3),
                                         ("cubic_to_quintic_boundary_over_u", 1001, 5)):
                other = boundary_response(g, points, args.k, degree)
                checks[name].extend((norms(other-base)/norm).tolist())
        if first % 100 == 0:
            print(f"diagnosed archived tuples {last}/{len(rows)}", flush=True)
    report = {
        "original": str(args.original),
        "original_sha256": hashlib.sha256(args.original.read_bytes()).hexdigest(),
        "samples": len(rows), "k": args.k,
        "provenance": "original archived tuples; not a reconstructed random ensemble",
        "decomposition": "archived u minus zero-source 1001-by-1001 DST boundary response",
        "boundary_interpolation": "edgewise cubic spline through the retained 51 values per edge",
        "limitations": "original fine boundary and generator parameters are unavailable; refinement and cubic-versus-quintic checks are diagnostics, not rigorous bounds on interpolation or text quantization error",
        "reference_check_count": check_count,
        "reference_checks": {key: stats(np.asarray(value)) for key, value in checks.items()},
        "metrics": {key: stats(np.asarray(value)) for key, value in observations.items()},
        "per_sample": observations,
    }
    write_json(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=float, required=True)
    run(parser.parse_args())
