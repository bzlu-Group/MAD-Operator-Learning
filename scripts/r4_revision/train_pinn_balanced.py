"""Original PINN objective with shared branches and exact tanh coordinate jets.

Scientific protocol: PoissonONet(200, 2601, 2, 90, 110), all 51**2 points
(including the boundary), 0.9*PDE MSE + 0.1*boundary MSE, Adam, shuffled
DataLoader, an unstepped ReduceLROnPlateau, and best PRE-UPDATE batch weights.
Differences from trainPINNhelmholtz2D.py: explicit k (the original residual
hard-coded +u), minimum loss starts at infinity instead of 1, and the seeded
R3 loader/checkpoint protocol. No normalization, labels, FD, AMP, or TF32.
Coordinate derivatives use the chain rule; parameter gradients still use AD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r3_revision"))
from train_original_protocol import PoissonONet, set_seed
from train_original_protocol_vectorized import predict_grid


BOUNDARY_COUNT = 200
POINT_COUNT = 2601
INPUT_COUNT = BOUNDARY_COUNT + POINT_COUNT
MILESTONES = (50000, 100000, 200000)
SCHEMA = "MAD-R4-PINN-balanced-v1"
TIMING_SCHEMA = "MAD-R4-training-timing-v1"
TIMING_SCOPE = (
    "cumulative training including best/latest checkpoint, history, milestone IO and hashing; "
    "excluding initialization, resume setup, evaluation and timing-receipt publication"
)
DIFFERENCES = [
    "k is explicit; the source PINN residual hard-coded k=1",
    "min_loss starts at infinity, not 1 (strong sources still save a best)",
    "R3 set_seed and independent DataLoader generator fix init and batch order",
    "exact tanh coordinate jets and matrix products reuse branch/trunk features",
    "strict epoch-boundary resume and atomic checkpoints/milestone hashes",
    "implementation only: finite flags are stacked for one scalar synchronization per device",
]


class PINNDataset(Dataset):
    """Accept only numeric [g, f] rows; reject label-bearing files before access."""

    def __init__(self, path: Path, num_functions: int):
        self.data = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.data.ndim != 2 or self.data.shape[1] != INPUT_COUNT:
            self.data._mmap.close()
            raise ValueError("Expected a g/f-only .npy matrix with exactly 2801 columns; labels are forbidden")
        if not np.issubdtype(self.data.dtype, np.floating):
            raise ValueError("Expected floating-point g/f data")
        if len(self.data) != num_functions:
            raise ValueError(f"Expected {num_functions} samples, found {len(self.data)}")
        for first in range(0, len(self.data), 256):
            if not np.isfinite(self.data[first:first + 256, :INPUT_COUNT]).all():
                raise FloatingPointError("Non-finite g/f data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index, :INPUT_COUNT], dtype=torch.float32)


class SupervisedDataset(Dataset):
    """Separate explicit label-bearing objective; PINNDataset stays input-only."""

    def __init__(self, path: Path, num_functions: int):
        self.data = np.load(path, mmap_mode="r", allow_pickle=False)
        if (self.data.shape != (num_functions, INPUT_COUNT + POINT_COUNT)
                or not np.issubdtype(self.data.dtype, np.floating)):
            raise ValueError("Expected a floating [g,f,u] matrix with 5402 columns")
        for first in range(0, len(self.data), 256):
            if not np.isfinite(self.data[first:first + 256]).all():
                raise FloatingPointError("Non-finite supervised tuples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)


def make_coordinates(*, dtype=torch.float32, device="cpu"):
    spacing = 1.0 / 50
    axis = np.arange(0.0, 1.0 + spacing, spacing)
    x, y = np.meshgrid(axis, axis)
    grid = torch.tensor(np.vstack((x.ravel(), y.ravel())).T, dtype=dtype, device=device)
    # Keep the original linspace calls and clockwise ordering, without corners twice.
    axis_up = torch.linspace(0, 1, 51, dtype=dtype)[:-1]
    axis_down = torch.linspace(1, 0, 51, dtype=dtype)[:-1]
    zeros = torch.zeros(50, dtype=dtype)
    ones = torch.ones(50, dtype=dtype)
    boundary = torch.cat((torch.stack((axis_up, zeros), 1),
                          torch.stack((ones, axis_up), 1),
                          torch.stack((axis_down, ones), 1),
                          torch.stack((zeros, axis_down), 1)))
    return grid, boundary.to(device)


def trunk_jets(trunk: nn.Sequential, coordinates: torch.Tensor):
    """Propagate (value, dx, dy, laplacian), preserving the parameter AD graph."""
    value = coordinates
    dx = torch.zeros_like(value)
    dy = torch.zeros_like(value)
    dx[:, 0] = 1
    dy[:, 1] = 1
    lap = torch.zeros_like(value)
    for layer in trunk:
        if isinstance(layer, nn.Linear):
            value = F.linear(value, layer.weight, layer.bias)
            dx = F.linear(dx, layer.weight)
            dy = F.linear(dy, layer.weight)
            lap = F.linear(lap, layer.weight)
        elif isinstance(layer, nn.Tanh):
            value = torch.tanh(value)
            slope = 1 - value.square()
            lap = slope * lap - 2 * value * slope * (dx.square() + dy.square())
            dx = slope * dx
            dy = slope * dy
        else:
            raise TypeError(f"Unsupported trunk layer: {type(layer).__name__}")
    return value, dx, dy, lap


def pinn_predictions(model: PoissonONet, g, f, grid, boundary):
    # Separate boundary coordinates retain the original linspace rounding exactly.
    coordinates = torch.cat((grid, boundary))
    values, laps = [], []
    for net, branch_input in ((model.net1, g), (model.net2, f)):
        branch = net.branch_net(branch_input) * net.last_layer_weights
        value, _, _, lap = trunk_jets(net.trunk_net, coordinates)
        values.append(branch @ value.T)
        laps.append(branch @ lap[:len(grid)].T)
    prediction = values[0] + values[1]
    return prediction[:, :len(grid)], laps[0] + laps[1], prediction[:, len(grid):]


def pinn_loss(model, inputs, k, grid, boundary):
    g, f = inputs[:, :BOUNDARY_COUNT], inputs[:, BOUNDARY_COUNT:INPUT_COUNT]
    value, lap, boundary_value = pinn_predictions(model, g, f, grid, boundary)
    return 0.9 * (lap + k * value - f).square().mean() + 0.1 * (boundary_value - g).square().mean()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, write):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_save(path: Path, state):
    atomic_write(path, lambda handle: torch.save(state, handle))


def atomic_json(path: Path, state):
    payload = json.dumps(state, indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda handle: handle.write(payload))


def timing_receipt_path(checkpoint_path: Path):
    return checkpoint_path.with_suffix(".timing.json")


def timing_receipt_digest(receipt):
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_timing_receipt(checkpoint_path: Path, checkpoint):
    """Validate a latest or milestone receipt; never fall back to pre-save timing."""
    try:
        receipt = json.loads(timing_receipt_path(checkpoint_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Missing or invalid timing receipt; exact elapsed time is unavailable") from error
    if not isinstance(receipt, dict):
        raise ValueError("Invalid timing receipt")
    elapsed = receipt.get("elapsed_time")
    presave = checkpoint.get("elapsed_time")
    for value in (elapsed, presave, receipt.get("checkpoint_elapsed_time")):
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid timing receipt elapsed time")
    if (receipt.get("schema") != TIMING_SCHEMA or receipt.get("timing_scope") != TIMING_SCOPE or
            receipt.get("checkpoint_file") != checkpoint_path.name or
            receipt.get("data_sha256") != checkpoint["data_sha256"] or
            type(receipt.get("epoch")) is not int or receipt["epoch"] != checkpoint["epoch"] or
            type(receipt.get("global_step")) is not int or
            receipt["global_step"] != checkpoint["global_step"] or
            receipt["checkpoint_elapsed_time"] != presave or elapsed < presave or
            (checkpoint["epoch"] == 0 and (elapsed != 0 or presave != 0))):
        raise ValueError("Inconsistent timing receipt/checkpoint")
    try:
        digest = timing_receipt_digest(receipt)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid timing receipt payload") from error
    if receipt.get("receipt_sha256") != digest:
        raise ValueError("Corrupt timing receipt checksum")
    if receipt.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("Timing receipt checkpoint hash mismatch")
    return receipt


def assert_finite(name, tensors):
    checks = {}
    for tensor in tensors:
        if tensor is not None and (tensor.is_floating_point() or tensor.is_complex()):
            checks.setdefault(tensor.device, []).append(tensor.reshape(-1))
    if any(not torch.isfinite(torch.cat(group)).all().item() for group in checks.values()):
        raise FloatingPointError(f"Non-finite {name}")


def state_tensors(state):
    if isinstance(state, torch.Tensor):
        yield state
    elif isinstance(state, dict):
        for value in state.values():
            yield from state_tensors(value)
    elif isinstance(state, (list, tuple)):
        for value in state:
            yield from state_tensors(value)


def cpu_model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    if set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Incomplete RNG states")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count changed")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def runtime_info(device):
    cuda = device.type == "cuda"
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "platform": platform.platform(),
            "hardware": torch.cuda.get_device_name(device) if cuda else platform.processor(),
            "device": str(device), "cuda_runtime": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count() if cuda else 0,
            "precision": "float32", "amp": False,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads()}


def training_config(args, dataset):
    # Epoch target and reporting/checkpoint cadence may change during continuation.
    return {"schema": SCHEMA, "run_name": args.run_name, "k": args.k,
            "objective": args.objective,
            "num_functions": args.num_functions, "seed": args.seed,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "data_shape": list(dataset.data.shape), "data_dtype": str(dataset.data.dtype),
            "architecture": [200, 2601, 2, 90, 110, 1000, 4],
            "collocation": "all 51x51 including boundary; original meshgrid order",
            "boundary": "200 original clockwise square points",
            "loss_weights": [0.9, 0.1] if args.objective == "pinn" else None,
            "normalization": "none; frozen g/f unchanged",
            "best_rule": "minimum pre-update training batch loss; initial inf",
            "optimizer": "Adam defaults", "scheduler": "ReduceLROnPlateau(min,50,0.95); no step",
            "loader": "R3 DataLoader; shuffle=True; workers=0; drop_last=False",
            "milestones": list(MILESTONES)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--objective", choices=("pinn", "supervised"), default="pinn")
    parser.add_argument("--epochs", type=int, default=200000)
    parser.add_argument("--num-functions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--benchmark-steps", type=int, default=0,
                        help="1..50 actual optimizer steps, benchmark.json only, then exit")
    return parser.parse_args(argv)


def validate_args(args):
    if (not args.run_name or args.run_name in (".", "..") or
            any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for c in args.run_name)):
        raise ValueError("run-name must be a simple ASCII filename")
    if not math.isfinite(args.k) or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("k must be finite and learning-rate finite and positive")
    if args.epochs < 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid epochs or seed")
    if min(args.num_functions, args.batch_size, args.print_every, args.checkpoint_every) < 1:
        raise ValueError("Counts/cadences must be positive")
    if not 0 <= args.benchmark_steps <= 50 or (args.resume and args.benchmark_steps):
        raise ValueError("benchmark-steps must be 0..50; benchmark cannot resume")


def run(args, *, device=None):
    validate_args(args)
    root = args.output_dir
    if not args.resume and root.exists():
        raise FileExistsError(f"Fresh run refuses existing output directory: {root}")
    checkpoint_path = root / f"{args.run_name}_checkpoint.pth"
    best_path = root / f"{args.run_name}_best.pth"
    if args.resume and not checkpoint_path.is_file():
        raise FileNotFoundError("Resume requires the latest checkpoint")
    dataset_class = PINNDataset if args.objective == "pinn" else SupervisedDataset
    dataset = dataset_class(args.data, args.num_functions)
    data_hash = sha256(args.data)
    config = training_config(args, dataset)
    device = torch.device(device if device is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    runtime = runtime_info(device)
    set_seed(args.seed)
    model = PoissonONet(200, 2601, 2, 90, 110).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=50, factor=0.95)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    grid, boundary = make_coordinates(device=device)
    epoch, global_step, minimum = 0, 0, float("inf")
    best_state, best_record = None, None
    losses, elapsed = [], 0.0

    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["schema"] != SCHEMA or checkpoint["data_sha256"] != data_hash:
            raise ValueError("Resume data hash/schema mismatch")
        if checkpoint["config"] != config or checkpoint["runtime"] != runtime:
            raise ValueError("Resume config/runtime mismatch")
        epoch = checkpoint["epoch"]
        if not isinstance(epoch, int) or not 0 <= epoch <= args.epochs:
            raise ValueError("Resume target precedes the saved epoch")
        global_step = checkpoint["global_step"]
        losses = checkpoint["loss_records"]
        minimum = checkpoint["min_loss"]
        best_state, best_record = checkpoint["best_model_state_dict"], checkpoint["best_record"]
        if (len(losses) != epoch or global_step != epoch * len(loader) or
                not all(math.isfinite(value) for value in losses) or
                (epoch > 0 and (best_state is None or not math.isfinite(minimum)))):
            raise ValueError("Inconsistent resume progress/best")
        elapsed = read_timing_receipt(checkpoint_path, checkpoint)["elapsed_time"]
        assert_finite("checkpoint", state_tensors(checkpoint))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        generator.set_state(checkpoint["loader_generator_state"].cpu())
        restore_rng(checkpoint["rng_state"])
        if best_state is not None:
            # A crash mid-epoch can leave best.pth ahead of the last completed epoch.
            # Roll it back to the embedded best along with model/optimizer/RNG states.
            if not best_path.is_file() or sha256(best_path) != checkpoint["best_sha256"]:
                atomic_save(best_path, best_state)
                print("Restored retained best from the completed-epoch checkpoint", flush=True)
    else:
        root.mkdir(parents=True, exist_ok=False)

    model.train()
    started = None

    def training_elapsed():
        return elapsed if started is None else elapsed + time.perf_counter() - started

    def save_checkpoint(completed_epoch):
        nonlocal elapsed, started
        state = {"schema": SCHEMA, "epoch": completed_epoch, "global_step": global_step,
                 "model_state_dict": cpu_model_state(model),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(), "min_loss": minimum,
                 "best_model_state_dict": best_state, "best_record": best_record,
                 "best_sha256": sha256(best_path) if best_state is not None else None,
                 "loss_records": losses, "elapsed_time": training_elapsed(),
                 "loader_generator_state": generator.get_state(), "rng_state": rng_state(),
                 "data_sha256": data_hash, "config": config, "runtime": runtime,
                 "differences": DIFFERENCES}
        atomic_save(checkpoint_path, state)
        atomic_json(root / f"{args.run_name}_history.json",
                    {"epoch": completed_epoch, "global_step": global_step,
                     "minimum_batch_loss": minimum if math.isfinite(minimum) else None,
                     "last_batch_losses": losses, "data_sha256": data_hash,
                     "config": config, "runtime": runtime, "differences": DIFFERENCES})
        snapshot = None
        if completed_epoch == 0 or completed_epoch in MILESTONES:
            snapshot = root / "milestones" / f"epoch_{completed_epoch:06d}"
            snapshot.mkdir(parents=True, exist_ok=False)
            names = [checkpoint_path.name]
            if best_state is not None:
                names.append(best_path.name)
            for name in names:
                with (root / name).open("rb") as source:
                    atomic_write(snapshot / name, lambda handle: shutil.copyfileobj(source, handle))
            atomic_json(snapshot / "hashes.json",
                        {"epoch": completed_epoch, "data_sha256": data_hash,
                         "config": config, "best_record": best_record,
                         "files": {name: sha256(snapshot / name) for name in names}})
        receipt = {"schema": TIMING_SCHEMA, "timing_scope": TIMING_SCOPE,
                   "epoch": completed_epoch, "global_step": global_step,
                   "checkpoint_file": checkpoint_path.name,
                   "checkpoint_sha256": sha256(checkpoint_path), "data_sha256": data_hash,
                   "checkpoint_elapsed_time": state["elapsed_time"]}
        # Seal only after all checkpoint/history/milestone IO and hashes finish.
        if started is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = training_elapsed()
            started = None
        receipt["elapsed_time"] = elapsed
        receipt["receipt_sha256"] = timing_receipt_digest(receipt)
        if snapshot is not None:
            atomic_json(timing_receipt_path(snapshot / checkpoint_path.name), receipt)
        atomic_json(timing_receipt_path(checkpoint_path), receipt)
        if completed_epoch > 0 and completed_epoch < args.epochs:
            started = time.perf_counter()

    def step(inputs):
        assert_finite("batch g/f", [inputs])
        optimizer.zero_grad()
        if args.objective == "pinn":
            loss = pinn_loss(model, inputs, args.k, grid, boundary)
        else:
            value = predict_grid(model, inputs[:, :BOUNDARY_COUNT],
                                 inputs[:, BOUNDARY_COUNT:INPUT_COUNT], grid)
            loss = (value - inputs[:, INPUT_COUNT:]).square().mean()
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if any(parameter.grad is None for parameter in model.parameters()):
            raise FloatingPointError("Missing parameter gradient")
        assert_finite("parameter gradients", [p.grad for p in model.parameters()])
        return value

    def update():
        optimizer.step()
        assert_finite("updated parameters/optimizer", state_tensors(
            [model.state_dict(), optimizer.state_dict()]))

    if args.benchmark_steps:
        timings = []
        while len(timings) < args.benchmark_steps:
            for inputs in loader:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                step(inputs.to(device))
                update()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - tick)
                if len(timings) == args.benchmark_steps:
                    break
        report = {"kind": "performance_benchmark_not_experiment", "steps": len(timings),
                  "warmup_steps": 0, "step_seconds": timings,
                  "mean_step_seconds": sum(timings) / len(timings),
                  "timing_scope": "g/f transfer, forward, backward, Adam and finite checks; no checkpoint IO",
                  "cuda_synchronized": device.type == "cuda", "config": config,
                  "data_sha256": data_hash, "runtime": runtime, "differences": DIFFERENCES}
        atomic_json(root / "benchmark.json", report)
        print(f"Benchmark only: {len(timings)} steps, {report['mean_step_seconds']:.6f} s/step", flush=True)
        return report

    if not args.resume:
        save_checkpoint(0)
    if epoch < args.epochs:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
    for current_epoch in range(epoch, args.epochs):
        for batch_index, inputs in enumerate(loader):
            value = step(inputs.to(device))
            if value < minimum:
                best_state = cpu_model_state(model)
                best_record = {"epoch": current_epoch + 1, "batch_index": batch_index,
                               "global_step": global_step, "loss": value,
                               "parameter_timing": "pre_update"}
                atomic_save(best_path, best_state)
                minimum = value
            update()
            global_step += 1
        completed = current_epoch + 1
        losses.append(value)
        if completed % args.print_every == 0 or current_epoch == epoch:
            print(f"Epoch [{completed}], Minimum Loss: {minimum:.6e}, Last-batch Loss: {value:.6e}", flush=True)
        if (completed % args.checkpoint_every == 0 or completed in MILESTONES or
                completed == args.epochs):
            save_checkpoint(completed)
    print(f"Completed epoch {args.epochs}; elapsed {elapsed:.2f} seconds", flush=True)
    return {"epoch": args.epochs, "global_step": global_step, "min_loss": minimum}


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
