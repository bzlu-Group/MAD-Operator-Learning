"""Evaluate paired objectives with input-only inference normalization.

References retain their frozen R4 scale. Predictions are made using a scale
computed from g/f alone, then returned to the stored reference scale.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from checkpoint_io import load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'r3_revision'))
from train_original_protocol import PoissonONet, TextDataset
from train_original_protocol_vectorized import predict_grid
from evaluate_components import metric


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def prediction(model, inputs, coordinates):
    g, f = inputs[:, :200], inputs[:, 200:2801]
    scale = torch.amax(torch.abs(inputs[:, :2801]), dim=1).clamp_min(1e-30)
    gn, fn = g / scale[:, None], f / scale[:, None]
    full = predict_grid(model, gn, fn, coordinates) * scale[:, None]
    source = predict_grid(model, torch.zeros_like(gn), fn, coordinates) * scale[:, None]
    boundary = predict_grid(model, gn, torch.zeros_like(fn), coordinates) * scale[:, None]
    return full, source, boundary, scale


@torch.inference_mode()
def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoissonONet(200, 2601, 2, 90, 110).to(device)
    state = load_checkpoint(args.model, trusted=getattr(args, 'trust_checkpoint', False))
    model.load_state_dict(state.get('model_state_dict', state))
    model.eval()
    axis = np.linspace(0, 1, 51)
    xx, yy = np.meshgrid(axis, axis)
    coordinates = torch.tensor(np.column_stack((xx.ravel(), yy.ravel())), dtype=torch.float32, device=device)
    data = np.load(args.dataset / 'data.npy', mmap_mode='r')
    full, source, boundary = [], [], []
    gap = 0.0
    input_scales = []
    for first in range(0, len(data), 25):
        inputs = torch.tensor(np.asarray(data[first:first + 25, :2801]), dtype=torch.float32, device=device)
        p, pf, pg, scale = prediction(model, inputs, coordinates)
        full.append(p.double().cpu().numpy())
        source.append(pf.double().cpu().numpy())
        boundary.append(pg.double().cpu().numpy())
        input_scales.extend(scale.cpu().tolist())
        direct = predict_grid(model, inputs[:, :200], inputs[:, 200:], coordinates)
        relative_gap = float(torch.linalg.vector_norm(p - direct) / torch.linalg.vector_norm(direct).clamp_min(1e-30))
        if not np.isfinite(relative_gap):
            raise ValueError('Non-finite input homogeneity ratio')
        gap = max(gap, relative_gap)
    scales = np.load(args.dataset / 'latent.npz')['scales']
    ug, uf = np.load(args.dataset / 'ug.npy'), np.load(args.dataset / 'uf.npy')
    result = {'model': str(args.model), 'model_sha256': digest(args.model),
              'test2_sha256': digest(args.dataset / 'data.npy'), 'samples': len(data),
              'test2_files_sha256': {name: digest(args.dataset / name) for name in
                                    ('data.npy', 'ug.npy', 'uf.npy', 'latent.npz', 'summary.json', 'refinement.json')},
              'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__,
                          'device': str(device), 'GPU': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
              'inference_normalization': 'max(abs(g),abs(f)); references are read only after prediction',
              'input_renormalization_nonunit_count': int(np.count_nonzero(np.asarray(input_scales) != 1)),
              'input_homogeneity_relative_gap': gap,
              'total': metric(np.concatenate(full), data[:, -2601:], scales),
              'source': metric(np.concatenate(source), uf, scales),
              'boundary': metric(np.concatenate(boundary), ug, scales),
              'ignore_source_reference': metric(ug, data[:, -2601:], scales)}
    if args.test1:
        dataset = TextDataset(args.test1)
        test_data = np.asarray(dataset.data)
        predictions = []
        for first in range(0, len(test_data), 25):
            inputs = torch.tensor(test_data[first:first + 25, :2801], dtype=torch.float32, device=device)
            p, _, _, _ = prediction(model, inputs, coordinates)
            predictions.append(p.double().cpu().numpy())
        result['original_Test_Set_1'] = metric(np.concatenate(predictions), test_data[:, -2601:].astype(np.float64))
        result['test1_sha256'] = digest(args.test1)
    if args.training_data:
        from train_pinn_balanced import make_coordinates, pinn_predictions, read_timing_receipt
        train = np.load(args.training_data, mmap_mode='r')
        checkpoint = load_checkpoint(args.checkpoint, trusted=getattr(args, 'trust_checkpoint', False))
        timing = read_timing_receipt(args.checkpoint, checkpoint)
        if train.shape != (2000, 5402) or checkpoint['epoch'] != args.epochs:
            raise ValueError('Training diagnostic does not match the fixed data/budget')
        grid, bc = make_coordinates(device=device)
        diagnostics = {'purpose': 'post-training diagnostic, never selection',
            'data_sha256': digest(args.training_data), 'checkpoint_sha256': digest(args.checkpoint),
            'epoch_budget': args.epochs, 'global_step': checkpoint['global_step'],
            'best_record': checkpoint['best_record'], 'elapsed_time': timing['elapsed_time'],
            'timing_receipt_sha256': timing['receipt_sha256'], 'timing_scope': timing['timing_scope']}
        for kind in ('retained', 'final'):
            if kind == 'final':
                model.load_state_dict(checkpoint['model_state_dict'])
            predictions = []
            pde_squared = boundary_squared = 0.0
            for first in range(0, len(train), 25):
                batch = torch.tensor(np.asarray(train[first:first + 25]), dtype=torch.float32, device=device)
                p, lap, pb = pinn_predictions(model, batch[:, :200], batch[:, 200:2801], grid, bc)
                predictions.append(p.double().cpu().numpy())
                pde_squared += float((lap.double() + args.k * p.double() - batch[:, 200:2801].double()).square().sum())
                boundary_squared += float((pb.double() - batch[:, :200].double()).square().sum())
            pde = pde_squared / (len(train) * 2601)
            boundary_mse = boundary_squared / (len(train) * 200)
            pred = np.concatenate(predictions)
            diagnostics[kind] = {'full_training_MSE': float(np.mean((pred - train[:, -2601:])**2)),
                'relative_l2': metric(pred, train[:, -2601:]),
                'PDE_MSE': pde, 'boundary_MSE': boundary_mse,
                'full_training_PINN_objective': .9 * pde + .1 * boundary_mse}
        result['training_diagnostics'] = diagnostics
    if not np.isfinite(gap) or gap > 2e-5:
        raise ValueError('Input-only renormalization does not reproduce homogeneous predictions')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--test1', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-data', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--k', type=float)
    parser.add_argument('--trust-checkpoint', action='store_true',
                        help='Allow unrestricted pickle only for independently trusted files')
    main(parser.parse_args())
