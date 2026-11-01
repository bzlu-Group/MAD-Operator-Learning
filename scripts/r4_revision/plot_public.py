"""Plot fixed analytical examples or full last-minibatch histories after training."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from checkpoint_io import load_checkpoint

from evaluate_pinn_control import prediction
from train_pinn_balanced import PoissonONet, make_coordinates
from publication_figure_style import figure_style


def analytical(points, case, k):
    x, y = np.asarray(points, dtype=np.float64).T
    if case == "quadratic":
        u, lap = x * x + y * y - x * y, np.full_like(x, 4.0)
    else:
        u = np.exp(x * x + y * y)
        lap = (4.0 + 4.0 * (x * x + y * y)) * u
    return u, lap + k * u


def load_model(path, *, trust_checkpoint=False):
    state = load_checkpoint(path, trusted=trust_checkpoint)
    model = PoissonONet(200, 2601, 2, 90, 110)
    model.load_state_dict(state.get("model_state_dict", state))
    return model.eval()


@torch.inference_mode()
def example(args):
    grid, boundary = make_coordinates()
    g, _ = analytical(boundary.numpy(), args.case, args.k)
    _, f = analytical(grid.numpy(), args.case, args.k)
    inputs = torch.tensor(np.r_[g, f][None], dtype=torch.float32)
    axis = np.linspace(0, 1, 201)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    truth, _ = analytical(points, args.case, args.k)
    predicted, errors = {}, {}
    for kind, path in (("MAD0", args.mad0), ("PINN", args.pinn)):
        model = load_model(path, trust_checkpoint=getattr(args, "trust_checkpoint", False))
        parts = []
        for start in range(0, len(points), 1024):
            coords = torch.tensor(points[start:start + 1024], dtype=torch.float32)
            parts.append(prediction(model, inputs, coords)[0].double().numpy().ravel())
        predicted[kind] = np.concatenate(parts)
        errors[kind] = float(np.linalg.norm(predicted[kind] - truth) / np.linalg.norm(truth))
    with figure_style():
        fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), layout="constrained")
        fields = [truth] + list(predicted.values())
        levels = np.linspace(min(a.min() for a in fields), max(a.max() for a in fields), 41)
        for row, kind in enumerate(("MAD0", "PINN")):
            panels = ((truth, "True solution", "viridis", levels),
                      (predicted[kind], f"{kind} prediction\nRelative $L^2$: {errors[kind]:.2e}", "viridis", levels),
                      (np.abs(predicted[kind] - truth), f"{kind} absolute error", "inferno", 40))
            for ax, (field, title, cmap, contour) in zip(axes[row], panels):
                artist = ax.contourf(xx, yy, field.reshape(xx.shape), levels=contour, cmap=cmap)
                artist.set_rasterized(True)
                fig.colorbar(artist, ax=ax, pad=0.02)
                ax.set(title=title, xlabel="$x$", ylabel="$y$", aspect="equal")
        fig.savefig(args.output, dpi=220)
        plt.close(fig)
    np.savez(args.output.with_suffix(".npz"), points=points, truth=truth, **predicted)
    print(json.dumps({"relative_l2": errors}))


def history(args):
    with figure_style("loss"):
        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        for label, path in (("MAD0", args.mad0), ("PINN", args.pinn)):
            record = json.loads(path.read_text(encoding="utf-8"))
            values = np.asarray(record["last_batch_losses"], dtype=np.float64)
            if len(values) != record["epoch"] or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError("Incomplete or invalid history")
            indices = np.unique(np.r_[np.arange(0, len(values), 100), len(values) - 1])
            ax.semilogy(indices + 1, values[indices], label=label, linewidth=0.8)
        ax.set(xlabel="Epoch", ylabel="Last-minibatch objective")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.savefig(args.output)
        plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("example", "history"))
    parser.add_argument("--mad0", type=Path, required=True)
    parser.add_argument("--pinn", type=Path, required=True)
    parser.add_argument("--k", type=int, choices=(0, 1, 10, 100), default=1)
    parser.add_argument("--case", choices=("quadratic", "exponential"), default="quadratic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow unrestricted pickle only for independently trusted files")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("Use a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (example if args.kind == "example" else history)(args)
