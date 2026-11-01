
# MAD-Operator-Learning

Operator learning with MAD: Training datasets and scripts for PDE solutions.

Code archive for version 1.0.1:
[doi:10.5281/zenodo.23836330](https://doi.org/10.5281/zenodo.23836330).
The DOI is registered when the corresponding Zenodo record is published.

Security update: use the supported installation environment and read
[`SECURITY.md`](SECURITY.md) before loading model files. Historical runtime
records are retained for provenance, not recommended for new installations.

## Overview
This repository contains the datasets and code for implementing the **Mathematical Artificial Data (MAD)** paradigm, designed for solving partial differential equations (PDEs) such as Poisson, Helmholtz, and Laplace equations. The MAD framework provides a **modular, scalable, and efficient approach** to operator learning.

## Features
- Analytical dataset generation for various PDEs.
- Training and evaluation of MAD and PINN-based models.
- Pre-trained models for quick evaluation and reproducibility.

## File Structure
- **`data/`**: Contains training and test datasets.
- **`models/`**: Stores pre-trained models for reproducibility. Users can delete these models to train from scratch.
- **`scripts/`**: Contains scripts for different functionalities:
  - **`data_generation/`**: Scripts for generating datasets for various PDEs (e.g., Laplace, Poisson, Helmholtz equations).
  - **`model_training/`**: Scripts for training MAD and PINN models on the generated datasets.
  - **`model_testing/`**: Scripts for testing trained models and visualizing performance.

## Revised Experiments

For the final source-term experiments, use
[the R4 reproduction guide](reproducibility/r4/README.md). It documents the
response-balanced Test Set 2 and solver inputs, the input-RMS PINN used in
Table 1, the raw-loss sensitivity baseline, the Table 6 paired comparison,
and the exact model-selection and evaluation rules. The original `models/`
checkpoints and historical source-term solver files are not substitutes for
these final experiments.

The additional experiments are indexed in
[the R3 experiment guide](scripts/r3_revision/README.md), including fast DST
data generation, paired training-data comparisons, fixed complex domains,
and source-free nonlinear Poisson--Boltzmann prediction with FNO and LNF-NO.
The guide includes commands, reviewed small inputs, and result summaries,
and provides from-scratch PB and molecular reproduction instructions.
It does not claim that every R3 checkpoint or dataset is included.
Original benchmark files are unchanged.

The LNF-NO architecture reference is H. Wu, J. Wang and B. Lu,
"Linear-Nonlinear Fusion Neural Operator for Partial Differential Equations,"
*Computer Modeling in Engineering & Sciences* 148(2), 29 (2026),
[doi:10.32604/cmes.2026.084608](https://doi.org/10.32604/cmes.2026.084608).
The additional MAD0-trained source-free comparison is a separate experiment
documented in the R3 guide, not a reuse of that article's numerical results.

---

## Quick Start

To quickly test the pre-trained MAD and PINN models, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/bzlu-Group/MAD-Operator-Learning.git
   cd MAD-Operator-Learning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the test script:
   ```bash
   python scripts/model_testing/testlaplace2D.py
   ```

This will load the pre-trained models from the `models/` directory, evaluate them on a predefined dataset, and display the results.

---

## Advanced Usage

If you'd like to regenerate the datasets, retrain the models, or train them from scratch, follow the instructions below:

### Step 1: Generate Dataset
Use the scripts in the `data_generation` folder to generate new training and testing datasets. For example, to generate data for the Helmholtz equation:
```bash
python scripts/data_generation/gMADpoisson2D.py
```
The generated datasets will be saved in the `data/` directory.

### Step 2: Train Models
If you want to train models from scratch, delete the pre-trained models in the `models/` directory and run the training scripts. For example:
```bash
python scripts/model_training/trainMADpoisson2D.py
```
**Note:** Training on a personal laptop might require reducing the `batch_size` parameter to avoid GPU memory issues.

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

## Pre-trained Models
Pre-trained models are provided in the `models/` directory for quick evaluation:

- **`MAD1helmholtz2D_(2000,51).pth`**: Pre-trained MAD model for solving Helmholtz equations with 2000 functions and 51x51 grid resolution.
- **`PINN1helmholtz2D_(2000,51).pth`**: PINN-based model trained on the same dataset for comparison.

These models can be directly loaded and tested using the `model_testing` scripts. If you'd like to retrain the models, simply delete these files and follow the training instructions above.

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
  environment is documented separately in the R4 guide; it is not replaced
  retrospectively. Do not use unknown checkpoint files with legacy scripts.
- Standard Python libraries like `time`, `os`, `random`, and `math` are used but do not require additional installation.

---

## Notes on System Requirements

- **GPU Memory:** Training MAD and PINN models with default parameters requires significant GPU memory. If running on a personal laptop, consider reducing the `batch_size` parameter to avoid out-of-memory errors.
- **Pre-trained Models:** If you only want to evaluate the models, you can directly use the pre-trained models provided in the `models/` folder without retraining.

---

## FAQ

1. **What if I encounter GPU out-of-memory errors during training?**
   - Reduce the `batch_size` parameter in the training scripts.
   - Use a smaller dataset by reducing the `num_samples` parameter during dataset generation.

2. **Can I use this repository for other PDEs?**
   - Yes, the framework is modular and can be adapted to other PDEs by modifying the dataset generation and model training scripts.

3. **Why does the model testing script fail to run?**
   - Ensure that the pre-trained models are present in the `models/` directory. If you have deleted them, you need to retrain the models using the training scripts.

4. **What if I want to use a custom dataset?**
   - You can modify the dataset generation scripts in the `data_generation` folder to create datasets tailored to your problem.

---

## License
This project is licensed under the MIT License.
