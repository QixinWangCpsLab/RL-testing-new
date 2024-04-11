# RL Testing Project

Welcome to the RL Testing Project! This project focuses on testing Reinforcement Learning (RL) algorithms and models.

## Getting Started

### Prerequisites

- Python 3.10 (recommended)
- Conda environment manager

### Installation

1. Create a new Conda environment for the project:

```bash
conda create -n "SB3Testing" python=3.10
```

2. Activate the newly created environment:

```bash
conda activate SB3Testing
```

3. Install the project dependencies:

```bash
pip install -e .
```

## Project Structure

The main scripts related to RL testing can be found in the `/RLTesting` subfolder.

```
/RLTesting
├── frozenlake/           # Classic RL testing environment: FrozenLake
├── human_oracle/         # Human Oracle (HO) related materials and generation code
├── logs/                 # Training logs and comparison code between Fuzzy Oracle and Human Oracle
├── mountaincar/          # Classic RL testing environment: MountainCar
├── training_scripts/     # Scripts for training RL agents
├── bug_lib.py            # Definition of designed bugs for testing RL programs
├── config.ini            # Project configuration file
├── config_parser.py      # Parser for the configuration file
├── get_and_train_models.py   # Helper script to obtain and train RL models
├── log_parser.py         # Helper script to parse log files
└── testing_script_fuzzyoracle.ipynb   # Core testing script: injecting bugs, training agents with SB3, and evaluating with Fuzzy Oracle
```

The `testing_script_fuzzyoracle.ipynb` is the core testing script that demonstrates the process of injecting bugs into RL programs, training agents using the Stable Baselines3 (SB3) framework, and evaluating the trained RL programs using the Fuzzy Oracle (FO).

The `human_oracle` directory contains materials required for Human Oracle (HO) testing and the code for generating HO-related materials.

The `log` directory contains the log records generated during SB3 training and the code for comparing the performance of Fuzzy Oracle (FO) and Human Oracle (HO).

Please refer to the scripts in this subfolder for implementing and running the RL testing procedures.
