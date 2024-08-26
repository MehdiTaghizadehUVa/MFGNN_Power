
# Multi-Fidelity Graph Neural Networks for Efficient Power Flow Analysis

This repository contains the implementation of the Multi-Fidelity Graph Neural Network (MF-GNN) framework for efficient power flow analysis under high-dimensional demand and renewable generation uncertainty. The methodology and results are detailed in the paper titled *"Multi-fidelity Graph Neural Networks for Efficient Power Flow Analysis under High-Dimensional Demand and Renewable Generation Uncertainty"*.

## Overview

The MF-GNN model integrates low-fidelity data obtained from DC power flow simulations with high-fidelity data from AC power flow simulations. This approach reduces the computational cost of generating training data while enhancing the accuracy and robustness of power flow analysis. The repository includes Python scripts for generating training data, implementing GNN models, and training multi-fidelity models.

## Repository Structure

- **`Data_Generation.py`**: This script generates and saves training data for power system analysis. It uses the PyPower library to run power flow simulations on different IEEE test cases, perturbing load settings and recording the results. The data is saved in a format suitable for training GNN models.

- **`GNNs.py`**: This script defines the GNN models and the training routines for low-fidelity, high-fidelity, and multi-fidelity models. It includes functions for loading data, training models, and evaluating their performance on test datasets.

## Requirements

To run the scripts, you need the following libraries:

- Python 3.x
- PyTorch
- PyTorch Geometric
- NumPy
- NetworkX
- PyPower
- Matplotlib
- Optuna (for hyperparameter tuning)

You can install the required packages using pip:

```bash
pip install torch torch-geometric numpy networkx pypower matplotlib optuna
```

## Usage

### Data Generation

Use the `Data_Generation.py` script to generate training data. You can specify various parameters such as the number of samples, power system case, and power flow type (AC/DC) through command-line arguments.

Example command:

```bash
python Data_Generation.py --num_samples 1000 --save_path train_data --case_name case30 --power_flow_type ac
```

This will generate 1000 samples using the IEEE 30-bus system for AC power flow analysis and save the data in the `data/case30` directory.

### Training GNN Models

The `GNNs.py` script trains the GNN models. You can choose to train low-fidelity, high-fidelity, or multi-fidelity models by adjusting the parameters. The script also logs the training progress and saves the trained models.

Example command:

```bash
python GNNs.py --case case30 --input-dir ./data/case30 --output-dir results --input-dim 5 --hidden-dim 128 --output-dim 2 --num-epochs-low-fidelity 400 --num-epochs-high-fidelity 400
```

This command trains the models on data from the IEEE 30-bus system and saves the results in the `results/case30` directory.

### Evaluation

The trained models can be evaluated using the evaluation functions within the `GNNs.py` script. The evaluation results, including error metrics and visualizations, are saved in the specified output directory.

## Citation

If you use this code in your research, please cite the paper:

**Taghizadeh, M., Khayambashi, K., Hasnat, M.A., & Alemazkoor, N.** (2024). Multi-fidelity Graph Neural Networks for Efficient Power Flow Analysis under High-Dimensional Demand and Renewable Generation Uncertainty. *Electric Power Systems Research*.

## Contact

For any questions or issues, please contact Mehdi Taghizadeh at [mehdi@virginia.edu](mailto:jrj6wm@virginia.edu).
