"""Small, single-thread CPU checks; optional CUDA algebra checks, never a run queue."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_pinn_balanced as worker
from train_original_protocol import TextDataset

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def coordinate_ad(model, g, f, points):
    """The complete nn.forward with repeated branches, as in the source PINN."""
    count = len(points)
    x = points.repeat(len(g), 1)[:, :1].clone().requires_grad_(True)
    y = points.repeat(len(g), 1)[:, 1:].clone().requires_grad_(True)
    value = model(g.repeat_interleave(count, 0), f.repeat_interleave(count, 0),
                  torch.cat((x, y), 1))

    def grad(value, coordinate):
        return torch.autograd.grad(value, coordinate, torch.ones_like(value), create_graph=True)[0]

    lap = grad(grad(value, x), x) + grad(grad(value, y), y)
    return value.reshape(len(g), count), lap.reshape(len(g), count)


def direct_boundary(model, g, f, points):
    return model(g.repeat_interleave(len(points), 0), f.repeat_interleave(len(points), 0),
                 points.repeat(len(g), 1)).reshape(len(g), len(points))


class AlgebraTests(unittest.TestCase):
    def check_equivalence(self, dtype, device, *, rtol, atol):
        worker.set_seed(71)
        template = worker.PoissonONet(200, 2601, 2, 90, 110).to(device=device, dtype=dtype)
        g = torch.randn(2, 200, dtype=dtype, device=device) * 0.1
        f = torch.randn(2, 2601, dtype=dtype, device=device) * 0.1
        points = torch.tensor([[0, 0], [.17, .31], [.44, .89], [1, 1]], dtype=dtype, device=device)
        _, boundary = worker.make_coordinates(dtype=dtype, device=device)
        for k in (0, 1, 10, 100):
            with self.subTest(k=k, dtype=dtype, device=device):
                fast, direct = copy.deepcopy(template), copy.deepcopy(template)
                calls = []
                hooks = [net.branch_net.register_forward_hook(
                    lambda *_: calls.append(1)) for net in (fast.net1, fast.net2)]
                with patch("torch.autograd.grad", side_effect=AssertionError("reverse coordinate AD in worker")):
                    value, lap, ub = worker.pinn_predictions(fast, g, f, points, boundary)
                for hook in hooks:
                    hook.remove()
                self.assertEqual(len(calls), 2)
                reference_value, reference_lap = coordinate_ad(direct, g, f, points)
                reference_ub = direct_boundary(direct, g, f, boundary)
                torch.testing.assert_close(value, reference_value, rtol=rtol, atol=atol)
                torch.testing.assert_close(lap, reference_lap, rtol=rtol, atol=atol)
                torch.testing.assert_close(ub, reference_ub, rtol=rtol, atol=atol)
                target = f[:, :len(points)]
                loss = .9 * (lap + k * value - target).square().mean() + .1 * (ub - g).square().mean()
                reference_loss = (.9 * (reference_lap + k * reference_value - target).square().mean()
                                  + .1 * (reference_ub - g).square().mean())
                torch.testing.assert_close(loss, reference_loss, rtol=rtol, atol=atol)
                loss.backward()
                reference_loss.backward()
                for (name, parameter), (ref_name, reference) in zip(
                        fast.named_parameters(), direct.named_parameters()):
                    self.assertEqual(name, ref_name)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertIsNotNone(reference.grad, name)
                    torch.testing.assert_close(parameter.grad, reference.grad,
                                               rtol=rtol, atol=atol, msg=name)

    def test_double_complete_forward_coordinate_ad_and_all_parameter_gradients(self):
        self.check_equivalence(torch.float64, "cpu", rtol=2e-9, atol=2e-10)

    @unittest.skipUnless(os.environ.get("PINN_TEST_CUDA") == "1" and torch.cuda.is_available(),
                         "optional CUDA algebra test (PINN_TEST_CUDA=1); no GPU training")
    def test_optional_float32_cuda_equivalence(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.check_equivalence(torch.float32, "cuda", rtol=3e-3, atol=2e-3)

    def test_boundary_order_and_grid_match_source_exactly(self):
        grid, boundary = worker.make_coordinates()
        n = 51
        original = torch.cat((
            torch.stack((torch.linspace(0, 1, n)[:-1], torch.zeros(n - 1)), 1),
            torch.stack((torch.ones(n - 1), torch.linspace(0, 1, n)[:-1]), 1),
            torch.stack((torch.linspace(1, 0, n)[:-1], torch.ones(n - 1)), 1),
            torch.stack((torch.zeros(n - 1), torch.linspace(1, 0, n)[:-1]), 1)))
        self.assertTrue(torch.equal(boundary, original))
        self.assertEqual(len(torch.unique(boundary, dim=0)), 200)
        self.assertEqual(boundary[[0, 50, 100, 150]].tolist(), [[0, 0], [1, 0], [1, 1], [0, 1]])
        axis = np.arange(0, 1 + 1 / 50, 1 / 50)
        x, y = np.meshgrid(axis, axis)
        expected = torch.tensor(np.vstack((x.ravel(), y.ravel())).T, dtype=torch.float32)
        self.assertTrue(torch.equal(grid, expected))
        self.assertEqual(grid.shape, (2601, 2))
        self.assertEqual(int(((grid == 0) | (grid == 1)).any(1).sum()), 200)

    def test_trunk_dx_dy_laplacian_chain_rule(self):
        worker.set_seed(5)
        trunk = worker.PoissonONet(200, 2601, 2, 90, 110).double().net1.trunk_net
        points = torch.tensor([[.2, .4], [.8, .7]], dtype=torch.float64, requires_grad=True)
        value, dx, dy, lap = worker.trunk_jets(trunk, points)
        weights = torch.linspace(-1, 1, 1000, dtype=torch.float64)
        reference = trunk(points)
        first = torch.autograd.grad((reference * weights).sum(), points, create_graph=True)[0]
        dxx = torch.autograd.grad(first[:, 0].sum(), points, retain_graph=True)[0][:, 0]
        dyy = torch.autograd.grad(first[:, 1].sum(), points)[0][:, 1]
        torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(dx @ weights, first[:, 0], rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(dy @ weights, first[:, 1], rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(lap @ weights, dxx + dyy, rtol=1e-10, atol=1e-11)

    def test_loss_weights_all_collocation_and_only_gf(self):
        batch = torch.arange(2 * 5402, dtype=torch.float64).reshape(2, 5402) / 1000
        grid, boundary = worker.make_coordinates(dtype=torch.float64)
        value = torch.linspace(-1, 1, 2601, dtype=torch.float64).repeat(2, 1)
        lap = value.square() + .3
        ub = torch.linspace(1, -1, 200, dtype=torch.float64).repeat(2, 1)
        for k in (0, 1, 10, 100):
            with patch.object(worker, "pinn_predictions", return_value=(value, lap, ub)) as predict:
                actual = worker.pinn_loss(None, batch, k, grid, boundary)
            g, f = batch[:, :200], batch[:, 200:2801]
            expected = .9 * torch.mean((lap + k * value - f)**2) + .1 * torch.mean((ub - g)**2)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(predict.call_args.args[1], g))
            self.assertTrue(torch.equal(predict.call_args.args[2], f))

    def test_finite_checks_reduce_once_per_device_and_reject_every_nonfinite_tensor(self):
        devices = [torch.device("cpu")]
        if os.environ.get("PINN_TEST_CUDA") == "1" and torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))
        tensors = [None, torch.tensor([1]), torch.tensor([True])]
        for device in devices:
            tensors.extend((torch.tensor([1., 2.], device=device),
                            torch.tensor([3.], dtype=torch.float64, device=device),
                            torch.tensor([1 + 2j], device=device)))
        real_item = torch.Tensor.item
        synchronizations = []

        def item(tensor):
            synchronizations.append(tensor.device)
            return real_item(tensor)

        with patch.object(torch.Tensor, "item", item):
            worker.assert_finite("test tensors", iter(tensors))
            self.assertEqual(synchronizations, devices)
            for index in range(3, len(tensors)):
                tensor = tensors[index]
                for value in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(index=index, value=value):
                        corrupted = list(tensors)
                        value = complex(1, value) if tensor.is_complex() else value
                        corrupted[index] = torch.tensor([value], dtype=tensor.dtype, device=tensor.device)
                        synchronizations.clear()
                        with self.assertRaisesRegex(FloatingPointError, "Non-finite test tensors"):
                            worker.assert_finite("test tensors", iter(corrupted))
                        self.assertEqual(len(synchronizations), len(set(synchronizations)))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "gf.npy"
        self.rows = np.random.default_rng(17).normal(0, .02, (3, 2801)).astype(np.float32)
        np.save(self.data, self.rows)

    def args(self, output="run", **changes):
        args = worker.parse_args(["--data", str(self.data), "--output-dir", str(self.root / output),
                                  "--run-name", "pinn", "--num-functions", "3", "--epochs", "1",
                                  "--seed", "19", "--batch-size", "2", "--print-every", "100",
                                  "--checkpoint-every", "100"])
        for name, value in changes.items():
            setattr(args, name, value)
        return args

    def run_worker(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return worker.run(args, device="cpu")

    def dataset(self, path):
        dataset = worker.PINNDataset(path, 3)
        self.addCleanup(dataset.data._mmap.close)
        return dataset

    def load(self, output="run", suffix="checkpoint"):
        return torch.load(self.root / output / f"pinn_{suffix}.pth", map_location="cpu", weights_only=False)

    def receipt(self, output="run"):
        checkpoint_path = self.root / output / "pinn_checkpoint.pth"
        return worker.read_timing_receipt(checkpoint_path, self.load(output))

    @contextlib.contextmanager
    def timing_clock(self):
        clock = {"now": 1000., "expected": 0., "active": False}
        real_model, real_loss = worker.PoissonONet, worker.pinn_loss
        real_write, real_hash = worker.atomic_write, worker.sha256

        def advance(seconds):
            clock["now"] += seconds
            if clock["active"]:
                clock["expected"] += seconds

        def initialize(*args, **kwargs):
            clock["active"] = False
            advance(100)
            return real_model(*args, **kwargs)

        def loss(*args, **kwargs):
            clock["active"] = True
            advance(2)
            return real_loss(*args, **kwargs)

        def write(path, callback):
            if path.name.endswith(".timing.json"):
                clock["active"] = False
                seconds = 19
            elif path.parent.name.startswith("epoch_"):
                seconds = 17 if path.name == "hashes.json" else (13 if "best" in path.name else 11)
            else:
                seconds = 7 if path.suffix == ".json" else (3 if "best" in path.name else 5)
            result = real_write(path, callback)
            advance(seconds)
            return result

        def file_hash(path):
            result = real_hash(path)
            advance(1)
            return result

        with patch.object(worker.time, "perf_counter", side_effect=lambda: clock["now"]), patch.object(
                worker, "PoissonONet", side_effect=initialize), patch.object(
                worker, "pinn_loss", side_effect=loss), patch.object(
                worker, "atomic_write", side_effect=write), patch.object(
                worker, "sha256", side_effect=file_hash):
            yield clock

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_defaults_have_no_seed_queue_scope(self):
        args = worker.parse_args(["--data", "gf.npy", "--output-dir", "new", "--run-name", "pinn"])
        self.assertEqual((args.epochs, args.batch_size, args.learning_rate), (200000, 25, 1e-4))
        self.assertEqual(worker.MILESTONES, (50000, 100000, 200000))

    def test_init_and_loader_batches_match_supervised_protocol(self):
        args = self.args(epochs=0)
        self.run_worker(args)
        state = self.load()
        worker.set_seed(args.seed)
        reference = worker.PoissonONet(200, 2601, 2, 90, 110)
        self.assert_tree_equal(state["model_state_dict"], reference.state_dict())
        pinndata = self.dataset(self.data)
        labeled = self.root / "tuple.npy"
        np.save(labeled, np.concatenate((self.rows, np.zeros((3, 2601))), 1))
        reference_data = TextDataset(labeled)
        self.addCleanup(reference_data.data._mmap.close)
        g1 = torch.Generator().manual_seed(args.seed)
        g2 = torch.Generator().manual_seed(args.seed)
        g1.set_state(state["loader_generator_state"])
        first = DataLoader(pinndata, batch_size=2, shuffle=True, generator=g1)
        second = DataLoader(reference_data, batch_size=2, shuffle=True, generator=g2)
        for _ in range(3):
            for a, b in zip(first, second):
                self.assertTrue(torch.equal(a, b[:, :2801]))
            self.assertTrue(torch.equal(g1.get_state(), g2.get_state()))

    def test_loss_ignores_tail_labels_without_loading_label_file(self):
        plain = self.dataset(self.data)
        worker.set_seed(13)
        model = worker.PoissonONet(200, 2601, 2, 90, 110)
        grid, boundary = worker.make_coordinates()
        batch = plain[0][None]
        with torch.no_grad():
            loss = worker.pinn_loss(model, batch, 10, grid, boundary)
            for label in (float("nan"), 1e20):
                extended = torch.cat((batch, torch.full((1, 2601), label)), 1)
                changed = worker.pinn_loss(model, extended, 10, grid, boundary)
                self.assertTrue(torch.equal(loss, changed))

    def test_dataset_training_and_benchmark_reject_label_files_before_access(self):
        labeled = self.root / "forbidden_tuple.npy"
        np.save(labeled, np.concatenate((self.rows, np.full((3, 2601), np.nan)), 1))
        with patch.object(np.memmap, "__getitem__", side_effect=AssertionError("dataset payload read")):
            with self.assertRaisesRegex(ValueError, "exactly 2801"):
                worker.PINNDataset(labeled, 3)
            for steps in (0, 2):
                output = f"forbidden_{steps}"
                args = self.args(output, data=labeled, benchmark_steps=steps)
                with self.subTest(benchmark_steps=steps), patch.object(
                        worker, "sha256") as data_hash, patch.object(worker, "PoissonONet") as model:
                    with self.assertRaisesRegex(ValueError, "exactly 2801"):
                        self.run_worker(args)
                    data_hash.assert_not_called()
                    model.assert_not_called()
                    self.assertFalse(args.output_dir.exists())

    def test_resume_is_bitwise_equal_to_uninterrupted(self):
        self.run_worker(self.args("whole", epochs=3))
        self.run_worker(self.args("split", epochs=1))
        random.random()
        np.random.random(5)
        torch.rand(5)
        self.run_worker(self.args("split", epochs=3, resume=True, checkpoint_every=1))
        whole, split = self.load("whole"), self.load("split")
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                    "epoch", "global_step", "min_loss", "best_model_state_dict", "best_record",
                    "loader_generator_state", "rng_state", "loss_records", "config", "data_sha256"):
            with self.subTest(key=key):
                self.assert_tree_equal(whole[key], split[key])
        self.assert_tree_equal(self.load("whole", "best"), self.load("split", "best"))
        self.assertEqual(split["scheduler_state_dict"]["last_epoch"], 0)
        self.assertEqual(split["optimizer_state_dict"]["param_groups"][0]["lr"], 1e-4)

    def test_timing_excludes_init_and_sealing_includes_io_and_resumes_consistently(self):
        with self.timing_clock() as clock:
            self.run_worker(self.args("timed_terminal", epochs=1))
            terminal = self.receipt("timed_terminal")
            self.assertEqual(terminal["elapsed_time"], clock["expected"])
            self.assertEqual(terminal["elapsed_time"] - terminal["checkpoint_elapsed_time"], 13)
        receipt_path = worker.timing_receipt_path(self.root / "timed_terminal" / "pinn_checkpoint.pth")
        for elapsed, message in ((terminal["elapsed_time"] + 1, "checksum"),
                                 (terminal["checkpoint_elapsed_time"] - 1, "receipt/checkpoint")):
            worker.atomic_json(receipt_path, dict(terminal, elapsed_time=elapsed))
            with self.assertRaisesRegex(ValueError, message):
                self.run_worker(self.args("timed_terminal", epochs=1, resume=True))
        with patch.object(worker, "MILESTONES", (1, 2)):
            with self.timing_clock() as clock:
                self.run_worker(self.args("timed_whole", epochs=2, checkpoint_every=1))
                whole = self.receipt("timed_whole")
                self.assertEqual(whole["elapsed_time"], clock["expected"])
                # Checkpoint + history + two copies + manifest + three file hashes.
                self.assertEqual(whole["elapsed_time"] - whole["checkpoint_elapsed_time"], 56)
            with self.timing_clock() as clock:
                self.run_worker(self.args("timed_split", epochs=1, checkpoint_every=1))
                first = self.receipt("timed_split")
                self.assertEqual(first["elapsed_time"], clock["expected"])
                self.assertGreater(first["elapsed_time"], self.load("timed_split")["elapsed_time"])
                clock["now"] += 10000  # Downtime/evaluation between segments is not training.
                self.run_worker(self.args("timed_split", epochs=2, resume=True, checkpoint_every=1))
                split = self.receipt("timed_split")
                self.assertEqual(split["elapsed_time"], clock["expected"])
                self.assertEqual(split["elapsed_time"], whole["elapsed_time"])
                self.assertGreater(split["elapsed_time"], first["elapsed_time"])
                receipt_path = worker.timing_receipt_path(self.root / "timed_split" / "pinn_checkpoint.pth")
                sealed = receipt_path.read_bytes()
                self.run_worker(self.args("timed_split", epochs=2, resume=True, checkpoint_every=1))
                self.assertEqual(receipt_path.read_bytes(), sealed)
                self.assertEqual(self.receipt("timed_split")["elapsed_time"], split["elapsed_time"])
        for output in ("timed_whole", "timed_split"):
            snapshot = self.root / output / "milestones" / "epoch_000000" / "pinn_checkpoint.pth"
            initial = torch.load(snapshot, weights_only=False)
            self.assertEqual(initial["elapsed_time"], 0)
            self.assertEqual(worker.read_timing_receipt(snapshot, initial)["elapsed_time"], 0)

    def test_timing_receipt_corruption_and_missing_receipt_fail_closed(self):
        self.run_worker(self.args(epochs=0))
        checkpoint_path = self.root / "run" / "pinn_checkpoint.pth"
        receipt_path = worker.timing_receipt_path(checkpoint_path)
        checkpoint = self.load()
        original = self.receipt()
        checkpoint_hash = worker.sha256(checkpoint_path)
        changes = (("schema", "wrong"), ("timing_scope", "wrong"),
                   ("checkpoint_file", "other.pth"), ("checkpoint_sha256", "0" * 64),
                   ("data_sha256", "0" * 64), ("epoch", 1), ("epoch", False),
                   ("global_step", 1), ("global_step", False),
                   ("checkpoint_elapsed_time", 1), ("elapsed_time", 1),
                   ("elapsed_time", -1), ("elapsed_time", True), ("elapsed_time", "0"),
                   ("elapsed_time", float("nan")), ("elapsed_time", float("inf")),
                   ("receipt_sha256", "0" * 64))
        for name, value in changes:
            with self.subTest(field=name, value=value):
                corrupted = dict(original, **{name: value})
                if name != "receipt_sha256" and not (
                        isinstance(value, float) and not np.isfinite(value)):
                    corrupted["receipt_sha256"] = worker.timing_receipt_digest(corrupted)
                payload = json.dumps(corrupted).encode("utf-8")
                worker.atomic_write(receipt_path, lambda handle: handle.write(payload))
                with self.assertRaisesRegex(ValueError, "[Tt]iming receipt"):
                    worker.read_timing_receipt(checkpoint_path, checkpoint)
        for payload in (b"{", b"[]"):
            worker.atomic_write(receipt_path, lambda handle: handle.write(payload))
            with self.assertRaisesRegex(ValueError, "[Tt]iming receipt"):
                worker.read_timing_receipt(checkpoint_path, checkpoint)
        receipt_path.unlink()
        with self.assertRaisesRegex(ValueError, "[Tt]iming receipt"):
            self.run_worker(self.args(resume=True))
        worker.atomic_json(receipt_path, dict(original, receipt_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "[Tt]iming receipt"):
            self.run_worker(self.args(resume=True))
        worker.atomic_json(receipt_path, original)
        payload = checkpoint_path.read_bytes() + b"corrupt"
        worker.atomic_write(checkpoint_path, lambda handle: handle.write(payload))
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            self.run_worker(self.args(resume=True))
        self.assertEqual(original["checkpoint_sha256"], checkpoint_hash)

    def test_failed_history_save_cannot_resume_with_previous_timing_receipt(self):
        self.run_worker(self.args(epochs=1))
        sealed = self.receipt()
        real_json = worker.atomic_json

        def fail_history(path, state):
            if path.name == "pinn_history.json":
                raise OSError("injected history failure")
            return real_json(path, state)

        with patch.object(worker, "atomic_json", side_effect=fail_history):
            with self.assertRaisesRegex(OSError, "injected history failure"):
                self.run_worker(self.args(epochs=2, resume=True))
        self.assertEqual(self.load()["epoch"], 2)
        self.assertEqual(json.loads(worker.timing_receipt_path(
            self.root / "run" / "pinn_checkpoint.pth").read_text()), sealed)
        with self.assertRaisesRegex(ValueError, "timing receipt/checkpoint"):
            self.run_worker(self.args(epochs=2, resume=True))

    def test_best_is_preupdate_and_saves_loss_above_one(self):
        strong = self.root / "strong.npy"
        np.save(strong, np.full((1, 2801), 4, dtype=np.float32))
        args = self.args(data=strong, num_functions=1, batch_size=25)
        worker.set_seed(args.seed)
        initial = worker.PoissonONet(200, 2601, 2, 90, 110)
        initial_state = worker.cpu_model_state(initial)
        grid, boundary = worker.make_coordinates()
        with torch.no_grad():
            expected = worker.pinn_loss(initial, torch.full((1, 2801), 4.), args.k, grid, boundary).item()
        self.assertGreater(expected, 1)
        self.run_worker(args)
        checkpoint = self.load()
        self.assert_tree_equal(initial_state, self.load(suffix="best"))
        self.assertEqual(checkpoint["min_loss"], expected)
        self.assertEqual(checkpoint["best_record"]["global_step"], 0)
        self.assertEqual(checkpoint["best_record"]["parameter_timing"], "pre_update")
        self.assertTrue(any(not torch.equal(value, checkpoint["model_state_dict"][name])
                            for name, value in initial_state.items()))

    def test_fresh_directory_and_resume_mismatch_fail_closed(self):
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_worker(self.args("existing", epochs=0))
        with self.assertRaises(FileNotFoundError):
            self.run_worker(self.args("absent", resume=True))
        self.run_worker(self.args(epochs=0))
        digest = worker.sha256(self.root / "run" / "pinn_checkpoint.pth")
        for name, value in (("k", 10), ("seed", 20), ("learning_rate", 2e-4), ("batch_size", 1)):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "config/runtime"):
                self.run_worker(self.args(resume=True, **{name: value}))
        np.save(self.data, self.rows + .001)
        with self.assertRaisesRegex(ValueError, "data hash"):
            self.run_worker(self.args(resume=True))
        self.assertEqual(worker.sha256(self.root / "run" / "pinn_checkpoint.pth"), digest)

    def test_epoch_zero_and_milestone_best_latest_hashes(self):
        # Scale only the checkpoint trigger for this two-epoch CPU test.
        args = self.args(num_functions=3, batch_size=3, epochs=2)
        with patch.object(worker, "MILESTONES", (1, 2)):
            self.run_worker(args)
        for epoch in (0, 1, 2):
            snapshot = self.root / "run" / "milestones" / f"epoch_{epoch:06d}"
            manifest = json.loads((snapshot / "hashes.json").read_text())
            self.assertEqual(manifest["epoch"], epoch)
            self.assertEqual(manifest["data_sha256"], worker.sha256(self.data))
            for name, digest in manifest["files"].items():
                self.assertEqual(worker.sha256(snapshot / name), digest)
            state = torch.load(snapshot / "pinn_checkpoint.pth", weights_only=False)
            self.assertEqual(state["epoch"], epoch)
            self.assertEqual("pinn_best.pth" in manifest["files"], epoch > 0)
            receipt = worker.read_timing_receipt(snapshot / "pinn_checkpoint.pth", state)
            self.assertEqual(receipt["epoch"], epoch)
            self.assertGreaterEqual(receipt["elapsed_time"], state["elapsed_time"])
            if epoch == 2:
                self.assertEqual(receipt, self.receipt())
        self.assertFalse(list((self.root / "run").rglob("*.tmp")))

    def test_benchmark_exits_without_experiment_checkpoints(self):
        report = self.run_worker(self.args(epochs=200000, benchmark_steps=2))
        self.assertEqual(report["kind"], "performance_benchmark_not_experiment")
        self.assertEqual(report["steps"], 2)
        self.assertTrue(all(value > 0 for value in report["step_seconds"]))
        self.assertFalse(report["cuda_synchronized"])
        self.assertEqual(sorted(p.name for p in (self.root / "run").iterdir()), ["benchmark.json"])
        with self.assertRaises(ValueError):
            self.run_worker(self.args("too_long", benchmark_steps=51))

    def test_nonfinite_inputs_loss_gradients_and_updates_are_rejected(self):
        np.save(self.data, np.full((3, 2801), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(FloatingPointError, "g/f"):
            self.run_worker(self.args("bad_data"))
        self.assertFalse((self.root / "bad_data").exists())
        np.save(self.data, self.rows)

        def bad_loss(model, *_):
            return next(model.parameters()).sum() * float("nan")

        with patch.object(worker, "pinn_loss", side_effect=bad_loss):
            with self.assertRaisesRegex(FloatingPointError, "loss"):
                self.run_worker(self.args("bad_loss"))

        real_backward = torch.Tensor.backward

        def bad_backward(tensor, *args, **kwargs):
            real_backward(tensor, *args, **kwargs)
            # The forward/backward graph is valid; inject a non-finite AD result.
            for leaf in captured:
                leaf.grad.fill_(float("inf"))

        captured = []
        real_loss = worker.pinn_loss

        def capture(model, *args):
            captured[:] = list(model.parameters())
            return real_loss(model, *args)

        with patch.object(worker, "pinn_loss", side_effect=capture), patch.object(
                torch.Tensor, "backward", bad_backward):
            with self.assertRaisesRegex(FloatingPointError, "gradients"):
                self.run_worker(self.args("bad_grad"))
        self.assertFalse((self.root / "bad_grad" / "pinn_best.pth").exists())

        real_step = torch.optim.Adam.step

        def bad_step(optimizer, *args, **kwargs):
            result = real_step(optimizer, *args, **kwargs)
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("inf"))
            return result

        with patch.object(torch.optim.Adam, "step", bad_step):
            with self.assertRaisesRegex(FloatingPointError, "updated"):
                self.run_worker(self.args("bad_update"))
        self.assertEqual(self.load("bad_update")["epoch"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
