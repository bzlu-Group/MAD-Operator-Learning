"""Reproduce R3 data before measuring their linear solution components."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_revision"))
from generate_helmholtz_datasets import (
    FastHelmholtzSolver, SineGenerator, normalize_row,
    sample_boundary_paths, set_seed, square_boundary, square_grid,
)
from relabel_mad0_solver import fields
from generate_balanced import norms, stats, write_json


def source_replay_scale(raw_source, archived_source):
    denominator = float(archived_source @ archived_source)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Archived source must be finite and nonzero")
    scale = float(raw_source @ archived_source / denominator)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Source replay requires a finite positive shared tuple scale")
    replay = raw_source / scale
    np.testing.assert_allclose(replay, archived_source, rtol=1e-9, atol=1e-11)
    return scale, float(np.max(np.abs(replay - archived_source)))


def run(args):
    torch.set_num_threads(4)
    original = np.load(args.original, mmap_mode="r") if args.original else None
    total = len(original) if original is not None else (args.samples or 200)
    n = min(total, args.samples) if args.samples else total
    solver = FastHelmholtzSolver(1001, args.k, 4)
    observations = {key: [] for key in ("uf_over_u", "ug_over_u", "normalized_max_u")}
    source_scales, source_replay_gaps, source_residuals = [], [], []
    maximum_replay_gap = 0.0
    if args.kind == "mad0":
        if original is None:
            raise ValueError("MAD0 diagnostics require the original tuples for replay verification")
        set_seed(args.seed)
        coarse = square_grid(51, torch.float64)
        coarse_boundary = square_boundary(51, torch.float64)
        fine_boundary = square_boundary(1001, torch.float64)
        with torch.no_grad():
            for index in range(n):
                generator = SineGenerator().eval()
                u, f = fields(generator, coarse, args.k)
                g = generator(coarse_boundary)[:, 0]
                raw = torch.cat((g, f, u)).numpy()
                scaled = raw/np.max(np.abs(raw))
                np.testing.assert_allclose(scaled, original[index], rtol=2e-10, atol=2e-12)
                maximum_replay_gap = max(maximum_replay_gap, float(np.max(np.abs(scaled-original[index]))))
                fine_g = generator(fine_boundary)[:, 0].numpy()
                ug = solver.solve(fine_g[None], np.zeros((1, 1001, 1001)))[:, ::20, ::20].reshape(1, -1)
                truth = u.numpy()[None]
                uf = truth-ug
                observations["uf_over_u"].append(float(norms(uf)[0]/norms(truth)[0]))
                observations["ug_over_u"].append(float(norms(ug)[0]/norms(truth)[0]))
                observations["normalized_max_u"].append(float(np.max(np.abs(scaled[-2601:]))))
                if index % 100 == 0:
                    print(f"verified MAD0 {index+1}/{n}", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        if original is not None:
            # Preserve source RNG position without recomputing the ill-conditioned GP factorization.
            rng.standard_normal((4001, total))
            boundaries = None
        else:
            boundaries = sample_boundary_paths(rng, total, 1001, 0.1)
        for first in range(0, n, 4):
            last = min(first+4, n)
            source = gaussian_filter(rng.standard_normal((last-first, 1001, 1001)),
                                     sigma=(0, 5, 5), mode="reflect")
            zero_boundary = np.zeros((last-first, 4000))
            fine_uf = solver.solve(zero_boundary, source)
            if original is not None:
                if np.any(fine_uf[:, 0, :]) or np.any(fine_uf[:, -1, :]) or np.any(fine_uf[:, :, 0]) or np.any(fine_uf[:, :, -1]):
                    raise ValueError("Source-only solve has nonzero boundary values")
                for offset in range(min(last-first, max(10-first, 0))):
                    v = fine_uf[offset]
                    residual = ((v[1:-1, :-2] + v[1:-1, 2:] + v[:-2, 1:-1] + v[2:, 1:-1]
                                 - 4*v[1:-1, 1:-1]) / solver.spacing**2
                                + args.k*v[1:-1, 1:-1] - source[offset, 1:-1, 1:-1])
                    gap = float(np.linalg.norm(residual) / np.linalg.norm(source[offset, 1:-1, 1:-1]))
                    if not np.isfinite(gap) or gap > 1e-8:
                        raise ValueError(f"Source-only discrete residual check failed: {first+offset}: {gap}")
                    source_residuals.append(gap)
            uf = fine_uf[:, ::20, ::20].copy()
            if original is not None:
                scales = []
                for offset, index in enumerate(range(first, last)):
                    scale, gap = source_replay_scale(source[offset, ::20, ::20].ravel(),
                                                     original[index, 200:2801])
                    scales.append(scale)
                    source_scales.append(scale)
                    source_replay_gaps.append(gap)
                    maximum_replay_gap = max(maximum_replay_gap, gap)
                uf /= np.asarray(scales)[:, None, None]
                truth = np.asarray(original[first:last, -2601:]).reshape(-1, 51, 51)
                trace = np.concatenate((truth[:, 0, :-1], truth[:, :-1, -1],
                                        truth[:, -1, ::-1][:, :-1], truth[:, ::-1, 0][:, :-1]), axis=1)
                np.testing.assert_allclose(trace, original[first:last, :200], rtol=1e-12, atol=1e-13)
                ug = truth-uf
            else:
                ug = solver.solve(boundaries[first:last], np.zeros_like(source))[:, ::20, ::20]
                truth = ug+uf
            for offset, index in enumerate(range(first, last)):
                if original is not None:
                    row = original[index]
                else:
                    row = normalize_row(boundaries[index, ::20], source[offset, ::20, ::20].ravel(),
                                        truth[offset].ravel(), "all")
                observations["normalized_max_u"].append(float(np.max(np.abs(row[-2601:]))))
            observations["uf_over_u"].extend((norms(uf)/norms(truth)).tolist())
            observations["ug_over_u"].extend((norms(ug)/norms(truth)).tolist())
            if first % 100 == 0:
                print(f"diagnosed legacy solver {last}/{n}", flush=True)
    solver_source_replay = original is not None and args.kind == "solver"
    report = {"kind": args.kind, "k": args.k, "seed": args.seed, "samples": n,
              "provenance": ("verified source replay of archived solver tuples" if solver_source_replay
                             else "verified replay of archived tuples") if original is not None
                            else "fresh reconstruction of R3 sampling law, not the archived TS2 realization",
              "original": str(args.original) if args.original else None,
              "original_sha256": hashlib.sha256(args.original.read_bytes()).hexdigest() if args.original else None,
              "maximum_replay_absolute_gap": maximum_replay_gap if original is not None else None,
              "replay_scope": "all 2601 retained source values per tuple" if solver_source_replay
                              else "full normalized tuple" if original is not None else "not an archived replay",
              "decomposition": "fine-grid zero-boundary source solve from verified source replay; boundary component is archived u minus uf"
                               if solver_source_replay else "fine-grid zero-source boundary solve; MAD0 source component is analytic u minus this solve"
                               if args.kind == "mad0" else "separate fine-grid zero-boundary source and zero-source boundary solves",
              "runtime": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
              "source_replay": {"shared_scales": source_scales, "coarse_source_maximum_gaps": source_replay_gaps,
                                "source_only_residual_first_ten": source_residuals,
                                "boundary_trace_checked_all": True, "original_draw_shape": [4001, total],
                                "source_batch_size": 4,
                                "limitations": "Coarse source values are verified against every archived tuple; original fine-grid GP boundary is not replayed, and the complement component retains any archived-solution error."}
                               if solver_source_replay else None,
              "metrics": {key: stats(np.asarray(value)) for key, value in observations.items()},
              "per_sample": observations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("mad0", "solver"), required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=float, default=1)
    run(parser.parse_args())
