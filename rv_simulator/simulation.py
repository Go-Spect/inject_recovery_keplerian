import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
from tqdm import tqdm

from .core import generate_keplerian_rv
from .utils import load_config, generate_random_parameter_value

def _generate_single_planet_rv(time, K, P, e, omega, T0):
    """Helper function for parallel execution of generate_keplerian_rv."""
    return generate_keplerian_rv(time, K, P, e, omega, T0)


def multiple_planet_rv(config):
    """
    Generates synthetic radial velocity data for a star with multiple planets.

    This function simulates a multi-planetary system by:
    1. Randomly determining the number of planets based on the config.
    2. Randomly generating orbital parameters for each planet.
    3. Calculating the Keplerian RV curve for each planet in parallel.
    4. Summing the RV contributions to get the total stellar signal.
    5. Adding Gaussian noise (jitter) to simulate observational uncertainty.

    Args:
        config (dict): A dictionary containing the simulation parameters.

    Returns:
        tuple: A tuple containing:
            - rv_data (pd.DataFrame): DataFrame with timestamps, individual planet
              RVs, the total RV signal, and the final RV with noise.
            - planet_params (pd.DataFrame): DataFrame with the generated orbital
              parameters for each simulated planet.
            - jitter (float): The value of the jitter added to the signal.
    """
    time_span = config.get('time_span', (0, 100))
    time_cadence_sec = config.get('time_cadence_sec', 300)
    n_planets_range = config.get('n_planets_range', (1, 5))
    p_range = config.get('p_range', (1, 100))
    k_range = config.get('k_range', (0.1, 100))
    e_range = config.get('e_range', (0.01, 0.5))
    omega_range = config.get('omega_range', (0, 2 * np.pi))
    jitter_range = config.get('jitter_range', (0.1, 5.0))
    p_scale = config.get('p_scale', 'linear')
    k_scale = config.get('k_scale', 'linear')
    e_scale = config.get('e_scale', 'linear')
    omega_scale = config.get('omega_scale', 'linear')
    jitter_scale = config.get('jitter_scale', 'linear')
    n_jobs = config.get('n_jobs', -1)

    time_cadence_days = time_cadence_sec / (24 * 3600)
    time = np.arange(time_span[0], time_span[1], time_cadence_days)

    n_planets = np.random.randint(n_planets_range[0], n_planets_range[1] + 1)

    P = generate_random_parameter_value(p_range[0], p_range[1], scale=p_scale, size=n_planets)
    K = generate_random_parameter_value(k_range[0], k_range[1], scale=k_scale, size=n_planets)
    e = generate_random_parameter_value(e_range[0], e_range[1], scale=e_scale, size=n_planets)
    omega = generate_random_parameter_value(omega_range[0], omega_range[1], scale=omega_scale, size=n_planets)
    T0 = np.array([generate_random_parameter_value(0, period, scale='linear') for period in P])

    params_list = list(zip(K, P, e, omega, T0))
    individual_rvs = Parallel(n_jobs=n_jobs)(delayed(_generate_single_planet_rv)(time, *p) for p in params_list)

    total_rv_signal = np.sum(individual_rvs, axis=0)
    jitter = generate_random_parameter_value(jitter_range[0], jitter_range[1], scale=jitter_scale)
    rv_with_noise = total_rv_signal + np.random.normal(0, jitter, size=len(time))

    rv_data = pd.DataFrame({f'rv_planet_{i+1}': rv for i, rv in enumerate(individual_rvs)}, index=time)
    rv_data['rv_total_signal'] = total_rv_signal
    rv_data['rv_with_noise'] = rv_with_noise
    rv_data.index.name = 'time'

    planet_params = pd.DataFrame(params_list, columns=['K', 'P', 'e', 'omega', 'T0'], index=[f'planet_{i+1}' for i in range(n_planets)])

    return rv_data, planet_params, jitter


def run_batch_simulations(config_path):
    """
    Runs a batch of simulations in parallel based on the config file.

    This function parallelizes the generation of entire, independent simulations.


    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        tuple: A tuple containing:
            - all_rv_data (dict): A dictionary where keys are simulation IDs (e.g., 'sim_00000')
              and values are the corresponding RV data as pandas DataFrames.
            - all_planet_params (pd.DataFrame): A single DataFrame containing all planet
              parameters from all simulations.
    """
    config = load_config(config_path)
    n_simulations = config.get('n_simulations', 1)
    n_jobs = config.get('n_jobs', -1)

    all_rv_data = {}
    all_planet_params_list = []

    print(f"Generating {n_simulations} simulations in parallel using up to {n_jobs} cores...")

    # Parallelize the execution of `multiple_planet_rv`
    results = Parallel(n_jobs=n_jobs)(
        delayed(multiple_planet_rv)(config) for _ in tqdm(range(n_simulations), desc="Dispatching Simulations")
    )

    print("\nProcessing results...")
    for i, (rv_data, planet_params, jitter) in enumerate(tqdm(results, desc="Aggregating Results")):
        sim_key = f'sim_{i:05d}'
        all_rv_data[sim_key] = rv_data

        planet_params['simulation_id'] = i
        planet_params['jitter'] = jitter
        all_planet_params_list.append(planet_params)

    final_params_df = pd.concat(all_planet_params_list)
    print("Batch simulation generation complete.")
    return all_rv_data, final_params_df