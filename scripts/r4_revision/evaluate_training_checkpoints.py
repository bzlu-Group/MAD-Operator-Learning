"""Report full-training errors of retained and final models without selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from checkpoint_io import load_checkpoint


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(prediction, truth):
    difference = prediction.astype(np.float64) - truth.astype(np.float64)
    reference = truth.astype(np.float64)
    denominators = np.linalg.norm(reference, axis=1)
    if np.any(denominators == 0):
        raise ValueError("Relative errors require nonzero reference functions")
    errors = np.linalg.norm(difference, axis=1) / denominators
    return {
        "full_training_mse": float(np.mean(difference ** 2)),
        "aggregate_relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "per_function_mean": float(errors.mean()),
        "per_function_std": float(errors.std(ddof=1)),
        "per_function_median": float(np.median(errors)),
        "per_function_max": float(errors.max()),
    }


def main(args):
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_revision"))
    from train_original_protocol import PoissonONet
    from evaluate_original_protocol import predict_grid

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    data = np.load(args.data).astype(np.float32)
    if data.shape != (2000, 5402) or not np.isfinite(data).all():
        raise ValueError("Expected 2000 finite original-protocol 51x51 training tuples")
    checkpoint = load_checkpoint(args.final, trusted=getattr(args, "trust_checkpoint", False))
    if checkpoint["epoch"] != args.epochs or len(checkpoint["loss_records"]) != args.epochs:
        raise ValueError("Checkpoint does not match the prespecified diagnostic budget")
    axis = np.linspace(0, 1, 51)
    xx, yy = np.meshgrid(axis, axis)
    coordinates = torch.tensor(np.column_stack((xx.ravel(), yy.ravel())), dtype=torch.float32)
    report = {
        "purpose": "post-hoc training diagnostic, not model selection",
        "data": str(args.data), "data_sha256": digest(args.data),
        "samples": len(data), "epoch_budget": args.epochs,
        "target_representation": "float32 normalized tuples used by the training loss",
        "error_accumulation": "float64", "device": "cpu", "CPU_threads": 4,
        "retained_rule": "minimum observed training-minibatch MSE, before optimizer update",
        "torch": torch.__version__, "numpy": np.__version__,
    }
    with torch.inference_mode():
        for name, path in (("retained", args.retained), ("final", args.final)):
            saved = load_checkpoint(path, trusted=getattr(args, "trust_checkpoint", False))
            state = saved.get("model_state_dict", saved)
            state = {key.removeprefix("module."): value for key, value in state.items()}
            model = PoissonONet(200, 2601, 2, 90, 110)
            model.load_state_dict(state)
            model.eval()
            predictions = []
            for start in range(0, len(data), 25):
                inputs = torch.from_numpy(data[start:start + 25])
                predictions.append(predict_grid(model, inputs[:, :200], inputs[:, 200:2801],
                                                coordinates).numpy())
            report[name] = {"path": str(path), "sha256": digest(path),
                            **metrics(np.concatenate(predictions), data[:, -2601:])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, choices=(50000, 100000, 200000), default=50000)
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow unrestricted pickle only for independently trusted files")
    main(parser.parse_args())
