"""Portable entry point for the R4 MAD0, raw PINN and input-RMS PINN workers.

The archived coordinate-jet/Adam/checkpoint implementation is reused unchanged.
This wrapper replaces only the author-specific cluster launcher. It is not a
bitwise-compatible restart entry point for the authors' historical jobs.
"""
import argparse
import sys

import numpy as np
import torch

import train_pinn_balanced as worker
from checkpoint_io import require_supported_runtime, require_trusted_resume


def input_scales(inputs):
    if (inputs.ndim != 2 or inputs.shape[1] != 2801
            or not np.issubdtype(inputs.dtype, np.floating)
            or not np.isfinite(inputs).all()):
        raise ValueError("RMS calibration accepts finite floating [g,f] inputs only")
    g2 = float(np.mean(np.asarray(inputs[:, :200], dtype=np.float64) ** 2))
    f2 = float(np.mean(np.asarray(inputs[:, 200:2801], dtype=np.float64) ** 2))
    if min(g2, f2) <= 0:
        raise ValueError("Both training input populations must have positive RMS")
    return {"g_mean_square": g2, "f_mean_square": f2}


def normalized_loss(value, lap, boundary_value, g, f, k, scales):
    return (0.9 * (lap + k * value - f).square().mean() / scales["f_mean_square"]
            + 0.1 * (boundary_value - g).square().mean() / scales["g_mean_square"])


def install_rms(scales):
    original_config = worker.training_config
    original_loss = worker.pinn_loss

    def config(args, dataset):
        result = original_config(args, dataset)
        result.update({"loss_mode": "input_rms", "loss_scales": scales,
                       "calibration": "all training g/f only; no solution or test labels"})
        return result

    def loss(model, inputs, k, grid, boundary):
        g, f = inputs[:, :200], inputs[:, 200:2801]
        value, lap, boundary_value = worker.pinn_predictions(model, g, f, grid, boundary)
        return normalized_loss(value, lap, boundary_value, g, f, k, scales)

    worker.training_config, worker.pinn_loss = config, loss
    return original_config, original_loss


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--loss-mode", choices=("input_rms", "raw", "supervised"),
                        default="input_rms")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Permit pickle loading only for your own trusted resume checkpoint")
    options, rest = parser.parse_known_args(argv)
    if "--help" in rest or "-h" in rest:
        parser.print_help()
        worker.parse_args(["--help"])
    args = worker.parse_args(rest)
    require_supported_runtime()
    if args.resume:
        require_trusted_resume(options.trust_checkpoint)
    if min(options.threads, options.interop_threads) < 1:
        raise ValueError("Thread counts must be positive")
    if args.objective != "pinn":
        raise ValueError("Select the objective with --loss-mode, not --objective")
    args.objective = "supervised" if options.loss_mode == "supervised" else "pinn"
    torch.set_num_threads(options.threads)
    torch.set_num_interop_threads(options.interop_threads)
    original = None
    if options.loss_mode == "input_rms":
        inputs = np.load(args.data, mmap_mode="r", allow_pickle=False)
        if len(inputs) != args.num_functions:
            raise ValueError("Calibration must use the entire specified training set")
        original = install_rms(input_scales(inputs))
    try:
        return worker.run(args, device=options.device)
    finally:
        if original is not None:
            worker.training_config, worker.pinn_loss = original


if __name__ == "__main__":
    main(sys.argv[1:])
