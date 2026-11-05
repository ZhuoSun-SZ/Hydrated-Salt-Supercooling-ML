# Hydrated-Salt-Supercooling-ML
Machine learning-assisted prediction and design of supercooling in hydrated salt PCMs
# Machine Learning-Assisted Prediction and Design of Hydrated-Salt Supercooling

## Overview

This repository provides the core computational framework for the study of machine learning-assisted prediction and design of hydrated-salt phase change materials with reduced supercooling.

The framework integrates multi-model learning, NSGA-II-based multi-objective base-learner selection, out-of-fold prediction, and a stacking ensemble with an optimized multilayer perceptron (MLP) meta-learner.

## Methodology

The computational workflow consists of:

1. Data preprocessing and feature preparation.
2. Training and evaluation of candidate machine learning models.
3. Multi-objective base-learner selection using NSGA-II.
4. Generation of out-of-fold predictions for stacking.
5. Hyperparameter optimization of the MLP meta-learner using Optuna-TPE.
6. Final evaluation on an independent test set.
7. Prediction of hydrated-salt supercooling for candidate material systems.

## Repository Structure

- `src/`: Core training, optimization, stacking, and evaluation scripts.
- `configs/`: Model and optimization configurations.
- `data/`: Example data and instructions for requesting restricted data.
- `supplementary/`: Feature descriptions, parameter settings, and software versions.
- `results/`: Instructions and examples for storing model evaluation outputs.

## Installation

The recommended Python environment and package dependencies are provided in `environment.yml` and `requirements.txt`.

## Data Availability

A subset of the data may be provided in this repository, subject to applicable data-sharing permissions. The remaining data supporting this study are available from the corresponding author upon reasonable request and subject to applicable data-use restrictions.

See `data/README.md` for details.

## Reproducibility

The scripts and configuration files document the core computational workflow. The availability of the complete dataset is subject to the conditions described above. Therefore, execution using the example dataset does not necessarily reproduce the exact numerical results reported in the manuscript.

## Citation

If you use this code, please cite the associated research article and the software release when available.

## License

See the `LICENSE` file for the applicable software license.
