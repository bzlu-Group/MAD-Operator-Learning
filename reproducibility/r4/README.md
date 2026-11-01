# Final source-term experiments

This is the reproduction entry point for Tables 1 and 6 and Section S5 of
*Mathematical artificial data for operator learning*. The earlier
[geometry and source-free experiment guide](../../scripts/r3_revision/README.md) preserves historical
experiments; its weak-source solver samples and source-term Test Set 2 must not
be substituted for the final experiments documented here.

## What is provided

- The exact response-balanced source generator, DST reference solver, numerical
  refinement checks, original coordinate-jet training worker, and evaluators.
- Portable input-preparation and training entry points. These replace private
  cluster launch paths, not the PDE, network, gradients or checkpoint rule.
- Frozen protocol parameters, hashes and machine-readable result summaries.
- Original public MAD0 data and analytical Test Set 1 in `data/`.
- The existing PB, flower-domain, molecular PQR-to-mesh and Laplace workflows
  in `scripts/r3_revision/`; their experiments are not replaced here.

Large generated datasets and trained checkpoints are not bundled. Commands below
generate the data and train new models; full training is computationally
substantial. Result JSONs record the published runs, not a substitute for
running the code. Floating-point training trajectories and time depend on the
runtime and hardware. Private cluster credentials, dispatchers and raw job logs
are not required and are not included.

## Environment and quick checks

Run commands from the repository root in a fresh Python 3.11+ environment:

```sh
python -m pip install -r requirements-r4.txt
python scripts/r4_revision/test_balanced.py
python scripts/r4_revision/test_public_reproduction.py
python scripts/r4_revision/test_checkpoint_io.py
```

The recorded A800 training used Python 3.9.19, NumPy 1.26.3 and PyTorch
2.5.1+cu121, float32, without AMP or TF32. These are historical experiment
records, not a recommendation to install an affected version. The supported
public environment now uses PyTorch 2.15.0 for checkpoint security; see
[`SECURITY.md`](../../SECURITY.md). It does not promise bitwise-identical
training on the newer runtime. The CPU checks do not require a GPU.
Optional geometry workflows use additional dependencies listed in
`requirements-r3.txt`, installed separately as appropriate.

## 1. Generate training data and Test Set 2

For each `K` in `0`, `1`, `10`, `100`, substitute the numeric value in the
commands. Each invocation needs a new output directory:

```sh
python scripts/r4_revision/generate_balanced.py --output data/r4/train_kK --samples 2000 --seed 2026100511 --k K
python scripts/r4_revision/generate_balanced.py --output data/r4/train_kK --seed 2026100511 --refine --refine-count 10
python scripts/r4_revision/generate_balanced.py --output data/r4/test_kK --samples 200 --seed 2026100520 --k K
python scripts/r4_revision/generate_balanced.py --output data/r4/test_kK --seed 2026100520 --refine --refine-count 10
```

The reference grid is 1001 x 1001 and the retained grid is 51 x 51. The
source is a continuous Gaussian Fourier field with physical length parameter
0.1 and 24 modes per axis. Independent source/boundary shapes are scaled so
`norm(uf)/norm(ug)` is log-uniform on `[0.5, 2]`; amplitudes are therefore
coupled. The generator stores `u`, `uf`, `ug`, latent variables and shared
tuple scales. Source/boundary norm ratios are not energy fractions.

## 2. Prepare paired inputs without solution-label access by PINN

Choose the public MAD0 training text for each coefficient:

| K | MAD0 training text in `data/` | Analytical Test Set 1 in `data/` |
| --- | --- | --- |
| 0 | `MADpoisson2D_(2000, 51).txt` | `MADpoisson2D_(200, 51).txt` |
| 1 | `MAD1helmholtz2D_(2000, 51).txt` | `MAD1helmholtz2D_(200, 51).txt` |
| 10 | `MAD10helmholtz2D_(2000, 51).txt` | `MAD10helmholtz2D_(200, 51).txt` |
| 100 | `MAD100helmholtz2D_(2000, 51).txt` | `MAD100helmholtz2D_(200, 51).txt` |

For example, for K=1:

```sh
python scripts/r4_revision/prepare_public_inputs.py --generated data/r4/train_k1 --mad0-text "data/MAD1helmholtz2D_(2000, 51).txt" --output data/r4/paired_k1
```

The conversion produces `inputs.npy` with only 2801 g/f columns,
`supervised.npy` with the solver labels, and float32 `mad0.npy`. The scale is
`max(abs(g),abs(f))` per tuple. The PINN loader rejects label-bearing files.
The common linear input branches make inference homogeneous; evaluators
compute the input scale without reading solution labels, then undo that scale.

## 3. Table 1: MAD0 and input-RMS PINN

Run one seed per equation, N=2000, Adam lr=1e-4, 200000 epochs. Use batch 100
for K=0 and batch 25 for K=1,10,100. For K=1:

```sh
python scripts/r4_revision/train_public.py --loss-mode supervised --data data/r4/paired_k1/mad0.npy --output-dir results/r4/mad0_k1 --run-name model --k 1 --batch-size 25 --checkpoint-every 100 --interop-threads 56
python scripts/r4_revision/train_public.py --loss-mode input_rms --data data/r4/paired_k1/inputs.npy --output-dir results/r4/rms_k1 --run-name model --k 1 --batch-size 25 --checkpoint-every 10
```

Both commands default to seed 20260904 and 200000 epochs. The architecture is
the original `PoissonONet(200,2601,2,90,110)`, not a newly tuned network.
The RMS objective is

`0.9 * mean((Delta u + K*u - f)^2) / mean_train(f^2)`
`+ 0.1 * mean((u_boundary - g)^2) / mean_train(g^2)`.

Scales use all training g/f inputs only, not test data or solution labels.
The retained checkpoint is the minimum observed **pre-update training
minibatch objective**, not the smallest test error or final model. The worker
also saves final state, Adam/RNG/loader state, histories, milestones and timing
receipts. Use `--resume --trust-checkpoint` with the same command and output
to continue an interrupted portable run, only for a checkpoint you created
or whose source you independently trust. Full resume uses Python pickle;
never use that flag for an unknown downloaded file. Evaluation and plotting
use restricted loading by default and have no automatic pickle fallback.
Historical author-job checkpoints have different
launcher metadata and are not promised compatible with this portable wrapper.

The original raw-loss baseline in Table S.17 is reproduced with
`--loss-mode raw`, `--checkpoint-every 100` and `--interop-threads 56`, using
the **same corrected inputs**, not old weak-source data. RMS uses 2 interop
threads and checkpoint cadence 10; all workers use 2 intraop threads.
Recorded fixed-budget times include checkpoint/history I/O and hashing but
exclude queue, setup and final evaluation. Do not interpret the times as
equal-error timings or attribute their difference solely to RMS scaling.

## 4. Evaluate every test function and both components

For K=1 and either model's retained checkpoint:

```sh
python scripts/r4_revision/evaluate_pinn_control.py --model results/r4/rms_k1/model_best.pth --dataset data/r4/test_k1 --test1 "data/MAD1helmholtz2D_(200, 51).txt" --output results/r4/rms_k1/test.json
```

Repeat for MAD0, raw PINN and the other three equations. The report uses all
200 test functions, aggregate discrete relative L2, per-function statistics,
source-driven and boundary-driven field errors, a boundary-only reference,
and physical-scale aggregate errors. Boundary-driven **field** error is not
the boundary-condition error. Final-state diagnostics are separate from
retained-model results; the test set never changes the selection rule.

## 5. Table 6: paired training-data comparison

The source-term row is K=1, **50000 epochs**, batch 25, three paired model
seeds 20260904/5/6. Generate corrected solver tuples with data seeds
2026100511/12/13. Train each with
`scripts/r3_revision/train_original_protocol_vectorized.py`, specifying
`--epochs 50000 --batch-size 25 --num-functions 2000 --seed MODEL_SEED`.
Generate the corresponding MAD0 datasets with
`scripts/r3_revision/generate_helmholtz_datasets.py --samples 2000
--coefficient 1 --seed MODEL_SEED --output-dir NEW_DIRECTORY` and use only
its `mad0_k1_n2000_p51_seedMODEL_SEED.npy` output; its historical solver
output is **not** the corrected source distribution.

Use `evaluate_components.py` on the frozen corrected test directory and the
same analytical Test Set 1. The 100000/200000-epoch continuations in Table
S.15 are separate, complete three-seed groups, not a test-selected budget.
The source-free Laplace row and its R3 scripts remain unchanged. No matched-
hardware timing comparison is made for Table 6.

## Results, figures and provenance

[`results.json`](results.json) records Tables 1, 6, S.16 and S.17 and related
source/budget diagnostics, extracted from archived results without rounding.
The manifest records SHA-256 hashes of released code and result files.
The final source-term prediction plots and loss curves use the RMS PINN models/histories
for the primary comparison, not the old `models/` checkpoints or raw losses.
Loss curves are diagnostic objectives with different units, not accuracy.

Portable plotting examples (the output paths must be new):

```sh
python scripts/r4_revision/plot_public.py example --mad0 results/r4/mad0_k1/model_best.pth --pinn results/r4/rms_k1/model_best.pth --k 1 --case quadratic --output results/r4/quadratic.pdf
python scripts/r4_revision/plot_public.py history --mad0 results/r4/mad0_k1/model_history.json --pinn results/r4/rms_k1/model_history.json --output results/r4/loss_k1.pdf
```

Use `--case exponential` for the second fixed example. Plotting reads the
retained model and complete last-minibatch history; it does not select models.

The earlier source-term guide and summaries are retained for provenance; where the
source-term results differ, this guide and the final manuscript take
precedence. Original source-free, geometry and PB code/data are unchanged.

The `r4` path identifies the implementation stage, not a requirement to use
an earlier manuscript. Later code-availability and clean-format submissions
did not change the scientific configuration or numerical results here.
