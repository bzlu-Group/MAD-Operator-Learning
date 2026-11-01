"""Evaluate frozen R3/R4 dual-branch models and their source response."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from checkpoint_io import load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_revision"))
from train_original_protocol import PoissonONet, TextDataset
from evaluate_original_protocol import predict_grid, evaluate


def metric(prediction, truth, scales=None):
    difference = prediction-truth
    errors = np.linalg.norm(difference, axis=1)/np.maximum(np.linalg.norm(truth, axis=1), 1e-30)
    result = {"aggregate_relative_l2": float(np.linalg.norm(difference)/np.linalg.norm(truth)),
              "per_sample_mean": float(errors.mean()), "per_sample_std": float(errors.std(ddof=1)),
              "median": float(np.median(errors)), "max": float(errors.max())}
    if scales is not None:
        result["unscaled_aggregate_relative_l2"] = float(np.linalg.norm(difference*scales[:, None])/np.linalg.norm(truth*scales[:, None]))
    return result


@torch.inference_mode()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoissonONet(200, 2601, 2, 90, 110).to(device)
    saved = load_checkpoint(args.model, trusted=getattr(args, "trust_checkpoint", False))
    if "model_state_dict" in saved:
        saved = saved["model_state_dict"]
    saved = {key.removeprefix("module."): value for key, value in saved.items()}
    model.load_state_dict(saved)
    model.eval()
    axis = np.linspace(0, 1, 51)
    xx, yy = np.meshgrid(axis, axis)
    coordinates = torch.tensor(np.column_stack((xx.ravel(), yy.ravel())), dtype=torch.float32, device=device)
    data = np.load(args.dataset / "data.npy")
    predictions, source_predictions, boundary_predictions = [], [], []
    homogeneity_gap = 0.0
    for start in range(0, len(data), 25):
        inputs = torch.tensor(data[start:start+25], dtype=torch.float32, device=device)
        g, f = inputs[:, :200], inputs[:, 200:2801]
        predictions.append(predict_grid(model, g, f, coordinates).double().cpu().numpy())
        source_predictions.append(predict_grid(model, torch.zeros_like(g), f, coordinates).double().cpu().numpy())
        boundary_predictions.append(predict_grid(model, g, torch.zeros_like(f), coordinates).double().cpu().numpy())
        if start == 0:
            for scale in (0.1, 10.0):
                original = predict_grid(model, g, f, coordinates)
                transformed = predict_grid(model, scale*g, scale*f, coordinates)/scale
                gap = torch.linalg.vector_norm(transformed-original)/torch.linalg.vector_norm(original)
                homogeneity_gap = max(homogeneity_gap, float(gap))
    prediction = np.concatenate(predictions)
    uf, ug = np.load(args.dataset / "uf.npy"), np.load(args.dataset / "ug.npy")
    scales = np.load(args.dataset / "latent.npz")["scales"]
    report = {"model": str(args.model), "dataset": str(args.dataset), "samples": len(data),
              "total": metric(prediction, data[:, -2601:], scales),
              "source": metric(np.concatenate(source_predictions), uf, scales),
              "boundary": metric(np.concatenate(boundary_predictions), ug, scales),
              "ignore_source_reference": metric(ug, data[:, -2601:], scales),
              "input_homogeneity_relative_gap": homogeneity_gap}
    if args.test1:
        report["original_Test_Set_1"] = evaluate(model, args.test1, coordinates, 51, 25, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--test1", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow unrestricted pickle only for independently trusted files")
    main(parser.parse_args())
