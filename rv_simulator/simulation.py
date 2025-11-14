import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
import logging
from tqdm import tqdm

from .core import generate_keplerian_rv
from .utils import load_config, generate_random_parameter_value, load_time_array

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
        Returns (None, None, None) if timestamp loading fails.
    """
    period_days_range = config.get('period_days_range', (1, 100))
    semi_amplitude_ms_range = config.get('semi_amplitude_ms_range', (0.1, 100))
    eccentricity_range = config.get('eccentricity_range', (0.01, 0.5))
    arg_periastron_rad_range = config.get('arg_periastron_rad_range', (0, 2 * np.pi))
    n_planets_range = config.get('n_planets_range', (1, 5))
    period_days_scale = config.get('period_days_scale', 'linear')
    semi_amplitude_ms_scale = config.get('semi_amplitude_ms_scale', 'linear')
    eccentricity_scale = config.get('eccentricity_scale', 'linear')
    arg_periastron_rad_scale = config.get('arg_periastron_rad_scale', 'linear')
    n_jobs = config.get('n_jobs', -1)

    # --- 1. Determine the Time Vector (t_obs) ---
    external_file = config.get('external_timestamps_file')
    if external_file:
        # EXTERNAL MODE: Load from file
        column = config.get('timestamps_file_column')
        try:
            t_obs = load_time_array(external_file, column)
            logging.info(f"Successfully loaded {len(t_obs)} external timestamps from {os.path.basename(external_file)}.")
        except Exception as e:
            logging.error(f"Fatal error loading timestamps from {external_file}: {e}")
            # Return None to signal a failed simulation in the batch
            return None, None, None
    else:
        # INTERNAL MODE: Generate internally (legacy behavior)
        time_span = config.get('time_span')
        time_cadence_sec = config.get('time_cadence_sec')
        time_cadence_days = time_cadence_sec / (24 * 3600)
        # Add a small fraction of cadence to the end to ensure it's inclusive
        t_obs = np.arange(time_span[0], time_span[1] + time_cadence_days, time_cadence_days)

    n_points = len(t_obs)

    # --- 2. Determine Number of Planets and Generate Parameters ---
    n_planets = np.random.randint(n_planets_range[0], n_planets_range[1] + 1)

    if n_planets == 0:
        # --- ZERO-PLANET LOGIC ---
        # The total planetary signal is zero.
        total_rv_signal = np.zeros(n_points)
        individual_rvs = []
        # Create an empty DataFrame with the correct columns for later concatenation.
        planet_params = pd.DataFrame(columns=['K', 'P', 'e', 'omega', 'T0'])
    else:
        # --- N-PLANET LOGIC (Legacy Behavior) ---
        P = generate_random_parameter_value(period_days_range[0], period_days_range[1], scale=period_days_scale, size=n_planets)
        K = generate_random_parameter_value(semi_amplitude_ms_range[0], semi_amplitude_ms_range[1], scale=semi_amplitude_ms_scale, size=n_planets)
        e = generate_random_parameter_value(eccentricity_range[0], eccentricity_range[1], scale=eccentricity_scale, size=n_planets)
        omega = generate_random_parameter_value(arg_periastron_rad_range[0], arg_periastron_rad_range[1], scale=arg_periastron_rad_scale, size=n_planets)
        T0 = np.array([generate_random_parameter_value(0, period, scale='linear') for period in P])

        params_list = list(zip(K, P, e, omega, T0))
        individual_rvs = Parallel(n_jobs=n_jobs)(delayed(_generate_single_planet_rv)(t_obs, *p) for p in params_list)

        total_rv_signal = np.sum(individual_rvs, axis=0)
        planet_params = pd.DataFrame(params_list, columns=['K', 'P', 'e', 'omega', 'T0'], index=[f'planet_{i+1}' for i in range(n_planets)])

    # --- 3. Generate and Apply Jitter (or not) ---
    # Default to True for retro-compatibility.
    apply_jitter = config.get('apply_jitter', True)

    if apply_jitter:
        jitter_ms_range = config.get('jitter_ms_range')
        jitter_ms_scale = config.get('jitter_ms_scale')
        jitter_value = generate_random_parameter_value(jitter_ms_range[0], jitter_ms_range[1], scale=jitter_ms_scale)
        jitter_noise = np.random.normal(0, jitter_value, size=n_points)
    else:
        jitter_value = 0.0
        jitter_noise = 0.0 # Adding 0.0 does not change the signal

    # This line now works for both cases.
    rv_with_noise = total_rv_signal + jitter_noise

    rv_data = pd.DataFrame({f'rv_planet_{i+1}': rv for i, rv in enumerate(individual_rvs)}, index=t_obs)
    rv_data['rv_total_signal'] = total_rv_signal
    rv_data['rv_with_noise'] = rv_with_noise
    rv_data.index.name = 'time' # This will be renamed later in the analysis step

    return rv_data, planet_params, jitter_value


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
        # Handle simulations that may have failed (e.g., due to timestamp loading error)
        if rv_data is None or planet_params is None:
            logging.warning(f"Skipping aggregation for simulation ID {i} as it failed to generate.")
            continue

        sim_key = f'sim_{i:05d}'
        all_rv_data[sim_key] = rv_data

        planet_params['simulation_id'] = i
        planet_params['jitter'] = jitter
        all_planet_params_list.append(planet_params)

    # Concatenate all parameter dataframes. This now correctly handles empty
    # dataframes from zero-planet simulations.
    if all_planet_params_list:
        final_params_df = pd.concat(all_planet_params_list)
    else: # Handle edge case where all simulations failed
        final_params_df = pd.DataFrame()

    print("Batch simulation generation complete.")
    return all_rv_data, final_params_df