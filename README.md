# inject_recovery_keplerian: Synthetic RV Data Generator

This project generates synthetic radial velocity (RV) datasets for multi-planet systems. It is designed to create large, configurable batches of simulations for use in injection-recovery tests, model training, or statistical analysis.

## Project Structure

The project is organized into a simulator package and a main execution script for clarity and maintainability.

```
.
├── config.yaml                     # Main configuration file for simulations
├── main.py                         # Main script to run simulations and analysis
├── README.md                       # This file
├── requirements.txt                # Project dependencies
└── rv_simulator/                   # The core simulator package
    ├── __init__.py
    ├── analysis.py                 # Functions for loading and visualizing results
    ├── core.py                     # Core physics (Kepler's equation, RV curve generation)
    ├── simulation.py               # High-level simulation and batch processing logic
    └── utils.py                    # Helper functions (config loading, random sampling)
```

## Setup and Dependencies

### 1. Dependencies
This project requires the following Python libraries. They are listed in the `requirements.txt` file.
- `numpy`
- `pandas`
- `joblib` (for parallel processing)
- `pyyaml` (for reading the config file)
- `tqdm` (for progress bars)
- `matplotlib` (for plotting)
- `tables` (for HDF5 file support in pandas)

### 2. Installation
Clone the repository and install the required packages using pip:
```bash
git clone https://github.com/Go-Spect/inject_recovery_keplerian
cd inject_recovery_keplerian
pip install -r requirements.txt
```

## How to Run Simulations

### 1. Configure Your Simulation
All simulation parameters are controlled by the `config.yaml` file. Before running, you can edit this file to define:
- The number of simulations to generate (`n_simulations`).
- The ranges and sampling scales (`linear` or `log`) for all orbital parameters (`P`, `K`, `e`, etc.).
- The number of planets per system (`n_planets_range`).
- The observation time span and cadence.
- The output directory (`output_dir`).

### 2. Run the Batch Simulation
To start the process, simply run the `main.py` script from your terminal:
```bash
python main.py
```
The script will:
1.  Read the `config.yaml` file.
2.  Run the batch simulation, showing a progress bar.
3.  Save the results to the specified `output_dir`.
4.  Automatically generate a plot of the resulting parameter distributions for verification.
5.  Run an example analysis on the first simulation (`ID=0`) to demonstrate how to load the data.

## Output Structure and Data Verification

The simulation process generates a new directory (default: `simulation_results/`) containing the following files:

- **`batch_planet_params.csv`**: A master CSV file containing the ground-truth input parameters for **every planet** in **every simulation**. This is the key to your injection-recovery test.
    - Each row represents a single planet.
    - The `simulation_id` column links each planet to its corresponding RV data.
    - The `jitter` column shows the noise level applied to that entire simulation.

- **`batch_rv_data.h5`**: A high-performance HDF5 file containing all the time-series RV data.
    - This file acts like a Python dictionary.
    - Each simulation's RV data is stored as a pandas DataFrame under a unique key (e.g., `'sim_00000'`, `'sim_00001'`, etc.).

- **`parameter_distributions.png`**: A plot showing the distributions of all generated parameters. This is a crucial diagnostic plot to verify that the random sampling from your `config.yaml` ranges is working as expected. For parameters sampled on a `log` scale (like Period and K), the histogram with a logarithmic x-axis should appear roughly uniform.

### Tracing Inputs to Outputs

Yes, you can absolutely trace back all the parameters that built each simulation. The `simulation_id` is the link between the input parameters and the output data.

The `rv_simulator/analysis.py` module provides the `load_simulation_data` function to make this easy. Here is a clear example of how to load the data for a specific simulation (e.g., ID 42) and verify its parameters:

```python
import pandas as pd
from rv_simulator.analysis import load_simulation_data

# Define the output directory and the ID of the simulation you want to analyze
output_dir = 'simulation_results'
sim_id_to_load = 42

# Load the data
# rv_data is the DataFrame with the time-series RV curve
# planet_params is the DataFrame with the ground-truth inputs for this simulation
rv_data, planet_params = load_simulation_data(sim_id_to_load, output_dir)

if rv_data is not None:
    print("--- Ground Truth Inputs ---")
    print("These are the parameters used to generate the data.")
    print(planet_params)
    
    print("\n--- Resulting RV Data ---")
    print("This is the final, noisy RV data generated from the inputs above.")
    print(rv_data.head())
```
This workflow ensures that for any given RV curve you analyze, you have direct access to the exact ground-truth parameters that were injected, which is the foundation of an injection-recovery analysis.
