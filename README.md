
# MAD-Operator-Learning

Operator learning with MAD: Training datasets and scripts for PDE solutions.

Permanent code archive:
[doi:10.5281/zenodo.23836330](https://doi.org/10.5281/zenodo.23836330).
The Zenodo record is published and openly accessible.

Security update: use the supported installation environment and read
[`SECURITY.md`](SECURITY.md) before loading model files. Historical runtime
records are retained for provenance, not recommended for new installations.

## Overview
This repository contains the datasets and code for implementing the **Mathematical Artificial Data (MAD)** paradigm, designed for solving partial differential equations (PDEs) such as Poisson, Helmholtz, and Laplace equations. The MAD framework provides a **modular, scalable, and efficient approach** to operator learning.

## Features
- Analytical dataset generation for various PDEs.
- Training and evaluation of MAD and PINN-based models.
- Reproduction instructions, small geometry inputs, and recorded result summaries.
- Historical pretrained models in the GitHub repository; these are not included
  in the Zenodo archive or substitutes for the final source-term models.

## File Structure
- **`data/`**: Contains training and test datasets.
- **`models/`**: Historical pretrained models in GitHub, not in the Zenodo archive.
- **`reproducibility/`**: Experiment guides, fixed protocols, checksums and numerical summaries.
- **`scripts/`**: Contains scripts for different functionalities:
  - **`data_generation/`**: Scripts for generating datasets for various PDEs (e.g., Laplace, Poisson, Helmholtz equations).
  - **`model_training/`**: Scripts for training MAD and PINN models on the generated datasets.
  - **`model_testing/`**: Scripts for testing trained models and visualizing performance.

## Reproduce the paper

| Paper location | Experiment | Reproduction entry point |
| --- | --- | --- |
| Table 1; Figs. 3--4; Sections S5.5--S5.7 | Source-term MAD0 and input-RMS PINN comparison | [Source-term experiment guide](reproducibility/r4/README.md) |
| Table 6, Helmholtz; Tables S.5 and S.15 | Paired MAD0 and response-balanced solver-data training | [Source-term experiment guide, Section 5](reproducibility/r4/README.md#5-table-6-paired-training-data-comparison) |
| Table 6, Laplace; Table S.6 | Source-free MAD1 and solver-data training | [Geometry and source-free guide, Section 2](scripts/r3_revision/README.md#2-mad1-versus-grf-boundary-solver-generated-training-data) |
| Fig. 6; Section S5.3 | Fixed flower-shaped domain with three holes | [Geometry and source-free guide, Section 4](scripts/r3_revision/README.md#4-fixed-flower-shaped-three-hole-domain) |
| Section 3.7; Section S4 | Nonlinear Poisson--Boltzmann prediction | [PB generation, training and evaluation](scripts/r3_revision/PB_FROM_SCRATCH.md) |
| Fig. 7; Section S5.4 | Fixed seven-atom molecular domain | [Molecular-domain generation, training and evaluation](scripts/r3_revision/MOLECULAR_FROM_SCRATCH.md) |

Directory names containing `r3` or `r4` identify development stages. They are
retained so script paths and recorded provenance remain consistent. Use the
experiment links above for the final paper; the code does not need to be
renamed to match a manuscript revision number.

For the final source-term experiments, use
[the source-term experiment guide](reproducibility/r4/README.md). It documents the
response-balanced Test Set 2 and solver inputs, the input-RMS PINN used in
Table 1, the raw-loss sensitivity baseline, the Table 6 paired comparison,
and the exact model-selection and evaluation rules. The original `models/`
checkpoints and historical source-term solver files are not substitutes for
these final experiments.

The additional experiments are indexed in
[the geometry and source-free experiment guide](scripts/r3_revision/README.md), including fast DST
data generation, paired training-data comparisons, fixed complex domains,
and source-free nonlinear Poisson--Boltzmann prediction with FNO and LNF-NO.
The guide includes commands, reviewed small inputs, and result summaries,
and provides from-scratch PB and molecular reproduction instructions.
Large generated arrays and trained checkpoints are not bundled in Zenodo;
the guides provide fresh data-generation, training and evaluation commands.
Original benchmark files are unchanged.

The LNF-NO architecture reference is H. Wu, J. Wang and B. Lu,
"Linear-Nonlinear Fusion Neural Operator for Partial Differential Equations,"
*Computer Modeling in Engineering & Sciences* 148(2), 29 (2026),
[doi:10.32604/cmes.2026.084608](https://doi.org/10.32604/cmes.2026.084608).
The additional MAD0-trained source-free comparison is a separate experiment
documented in the PB guide, not a reuse of that article's numerical results.

---

## Quick Start

Start with the source-term reproduction environment and bounded checks;
these checks do not require trained weights or reproduce the full paper runs.

1. Clone this repository:
   ```bash
   git clone https://github.com/bzlu-Group/MAD-Operator-Learning.git
   cd MAD-Operator-Learning
   ```

2. Install dependencies:
   ```bash
   python -m pip install -r requirements-r4.txt
   ```

3. Run the lightweight checks:
   ```bash
   python scripts/r4_revision/test_balanced.py
   python scripts/r4_revision/test_public_reproduction.py
   python scripts/r4_revision/test_checkpoint_io.py
   ```

Then follow the experiment guide for data generation, training, evaluation
and plotting. Full training is computationally substantial. Distinguish a
fresh reproduction of the procedure from an exact replay of archived weights.

---

## Historical benchmark scripts

The examples below describe the original benchmark scripts. They are not
the final source-term workflow in Table 1. Read `SECURITY.md` before loading
model files; legacy loaders must only be used with independently trusted files.

If you'd like to regenerate the datasets, retrain the models, or train them from scratch, follow the instructions below:

### Step 1: Generate Dataset
Use the scripts in the `data_generation` folder to generate new training and testing datasets. For example, to generate data for the Poisson equation:
```bash
python scripts/data_generation/gMADpoisson2D.py
```
The generated datasets will be saved in the `data/` directory.

### Step 2: Train Models
For a fresh run, use a new output location rather than deleting archived
models or resuming an old run. For example:
```bash
python scripts/model_training/trainMADpoisson2D.py
```
**Note:** The paper's batch sizes and budgets are specified in each experiment
guide. Changing them defines a different training configuration and should
not be described as an exact reproduction of the reported setting.

Similarly, to train a PINN-based model:
```bash
python scripts/model_training/trainPINNlaplace2D.py
```

### Step 3: Test Models
After training, you can test the models using the `model_testing` scripts:
```bash
python scripts/model_testing/testlaplace2D.py
```
This script will evaluate the trained models on the test datasets and generate visualizations for the predictions and errors.

---

## Historical pretrained models
The GitHub repository contains historical models in `models/`. They are not
included in the Zenodo archive. Examples are:

- **`MAD1helmholtz2D_(2000,51).pth`**: Pre-trained MAD model for solving Helmholtz equations with 2000 functions and 51x51 grid resolution.
- **`PINN1helmholtz2D_(2000,51).pth`**: PINN-based model trained on the same dataset for comparison.

Use legacy evaluation scripts only with trusted matching checkpoints and
inputs. These weights are not the final input-RMS PINN models in Table 1.
For the final comparisons, follow the source-term experiment guide.

---

## Requirements

The following Python libraries are required:
- Python 3.11 or newer and `torch>=2.15.0,<2.16`
- `numpy>=1.26.3`
- `matplotlib>=3.9.2`
- `scipy>=1.14.1`
- `scikit-learn>=1.5.1`

Install dependencies using:
```bash
pip install -r requirements.txt
```

### Notes
- These are current public installation requirements. The recorded experiment
  environment is documented separately in the source-term guide; it is not replaced
  retrospectively. Do not use unknown checkpoint files with legacy scripts.
- Standard Python libraries like `time`, `os`, `random`, and `math` are used but do not require additional installation.

---

## Notes on System Requirements

- **GPU Memory:** Training MAD and PINN models with default parameters requires significant GPU memory. If running on a personal laptop, consider reducing the `batch_size` parameter to avoid out-of-memory errors.
- **Pretrained models:** Evaluating an archived model requires its matching
  inputs and an independently trusted checkpoint. The Zenodo archive instead
  provides fresh generation, training and testing workflows.

---

## FAQ

1. **What if I encounter GPU out-of-memory errors during training?**
   - Reduce the `batch_size` parameter in the training scripts.
   - Use a smaller dataset by reducing the `num_samples` parameter during dataset generation.

2. **Can I use this repository for other PDEs?**
   - Yes, the framework is modular and can be adapted to other PDEs by modifying the dataset generation and model training scripts.

3. **Why does the model testing script fail to run?**
   - Legacy evaluation requires a matching trusted checkpoint at the expected
     path. Zenodo does not bundle pretrained weights; use the generation,
     training and evaluation sequence in the relevant experiment guide.

4. **What if I want to use a custom dataset?**
   - You can modify the dataset generation scripts in the `data_generation` folder to create datasets tailored to your problem.

---

## License
This project is licensed under the MIT License.
