import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import yaml
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def solve_kepler_equation(mean_anomaly, 
                          eccentricity, 
                          tolerance=1e-9, 
                          max_iter=100):
    """
    Solves Kepler's equation M = E - e*sin(E) for the eccentric anomaly E.

    Args:
        mean_anomaly (float or array): Mean anomaly in radians.
        eccentricity (float): Orbital eccentricity.
        tolerance (float): The convergence tolerance.
        max_iter (int): Maximum number of iterations for the Newton-Raphson method.

    Returns:
        float or array: The eccentric anomaly in radians.
    """
    if eccentricity == 0:
        return mean_anomaly
    E = np.copy(mean_anomaly)
    for _ in range(max_iter):
        delta_E = (E - eccentricity * np.sin(E) - mean_anomaly) / (1 - eccentricity * np.cos(E))
        E -= delta_E
        if np.all(np.abs(delta_E) < tolerance):
            break

    return E

def generate_keplerian_rv(time, K, P, e, omega, T0):
    """
    Generates a Keplerian radial velocity curve.

    Args:
        time (array): Time stamps of the observations.
        K (float): RV semi-amplitude in m/s.
        P (float): Orbital period in days.
        e (float): Eccentricity (0 <= e < 1).
        omega (float): Argument of periastron in radians.
        T0 (float): Time of periastron passage in the same units as time.

    Returns:
        array: The radial velocity signal of the planet.
    """
    # 1. Calculate the mean anomaly (M)
    mean_anomaly = 2 * np.pi / P * (time - T0) 

    # 2. Solve Kepler's equation for the eccentric anomaly (E)
    eccentric_anomaly = solve_kepler_equation(mean_anomaly, e)

    # 3. Calculate the true anomaly (nu)
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(eccentric_anomaly / 2),
                                  np.sqrt(1 - e) * np.cos(eccentric_anomaly / 2))

    # 4. Calculate the RV curve
    rv_signal = K * (np.cos(true_anomaly + omega) + e * np.cos(omega))

    return rv_signal

def generate_random_parameter_value(min_val, max_val, scale='linear', size=None):
    """
    Generates random numbers from a uniform distribution on a linear or log scale.

    A 'log' scale corresponds to a Jeffreys prior for a scale parameter,
    which is uniform in log-space, p(x) ~ 1/x.
    A 'linear' scale corresponds to a uniform prior, p(x) ~ const.

    Args:
        min_val (float): The minimum value of the range.
        max_val (float): The maximum value of the range.
        scale (str): The scale for the uniform distribution.
                     Can be 'linear' or 'log'. Defaults to 'linear'.
        size (int or tuple of ints, optional): Output shape. If None, a single
            value is returned.

    Returns:
        float or np.ndarray: Random number(s) drawn from the specified prior.
    """
    if scale == 'linear':
        return np.random.uniform(min_val, max_val, size)
    elif scale == 'log':
        if min_val <= 0:
            raise ValueError("min_val must be > 0 for a 'log' scale.")
        log_min = np.log(min_val)
        log_max = np.log(max_val)
        log_random = np.random.uniform(log_min, log_max, size)
        return np.exp(log_random)
    else:
        raise ValueError("scale must be 'linear' or 'log'.")


def load_config(config_path):
    """Loads simulation parameters from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise


def _generate_single_planet_rv(time, K, P, e, omega, T0):
    """Helper function for parallel execution of generate_keplerian_rv."""
    return generate_keplerian_rv(time, K, P, e, omega, T0)


def multiple_planet_rv(config_path):
    """
    Generates synthetic radial velocity data for a star with multiple planets.

    This function simulates a multi-planetary system by:
    1. Loading parameters from the specified YAML file.
    2. Randomly determining the number of planets.
    3. Randomly generating orbital parameters for each planet.
    4. Calculating the Keplerian RV curve for each planet in parallel.
    5. Summing the RV contributions to get the total stellar signal.
    6. Adding Gaussian noise (jitter) to simulate observational uncertainty.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        tuple: A tuple containing:
            - rv_data (pd.DataFrame): DataFrame with timestamps, individual planet
              RVs, the total RV signal, and the final RV with noise.
            - planet_params (pd.DataFrame): DataFrame with the generated orbital
              parameters for each simulated planet.
            - jitter (float): The value of the jitter added to the signal.
    """
    # 1. Load configuration from YAML file
    config = load_config(config_path)

    # Unpack parameters from config, providing defaults
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

    # 2. Generate Timestamps
    time_cadence_days = time_cadence_sec / (24 * 3600)
    time = np.arange(time_span[0], time_span[1], time_cadence_days)

    # 3. Determine Number of Planets
    n_planets = np.random.randint(n_planets_range[0], n_planets_range[1] + 1)

    # 4. Generate Orbital Parameters for each planet
    P = generate_random_parameter_value(p_range[0], p_range[1], scale=p_scale, size=n_planets)
    K = generate_random_parameter_value(k_range[0], k_range[1], scale=k_scale, size=n_planets)
    e = generate_random_parameter_value(e_range[0], e_range[1], scale=e_scale, size=n_planets)
    omega = generate_random_parameter_value(omega_range[0], omega_range[1], scale=omega_scale, size=n_planets)
    T0 = np.array([generate_random_parameter_value(0, period, scale='linear') for period in P])

    # 5. Generate RV for each planet in parallel
    params_list = list(zip(K, P, e, omega, T0))
    individual_rvs = Parallel(n_jobs=n_jobs)(delayed(_generate_single_planet_rv)(time, *p) for p in params_list)

    # 6. Sum RVs and add jitter
    total_rv_signal = np.sum(individual_rvs, axis=0)
    jitter = generate_random_parameter_value(jitter_range[0], jitter_range[1], scale=jitter_scale)
    rv_with_noise = total_rv_signal + np.random.normal(0, jitter, size=len(time))

    # 7. Build the final DataFrames
    rv_data = pd.DataFrame({f'rv_planet_{i+1}': rv for i, rv in enumerate(individual_rvs)}, index=time)
    rv_data['rv_total_signal'] = total_rv_signal
    rv_data['rv_with_noise'] = rv_with_noise
    rv_data.index.name = 'time'

    planet_params = pd.DataFrame(params_list, columns=['K', 'P', 'e', 'omega', 'T0'], index=[f'planet_{i+1}' for i in range(n_planets)])

    return rv_data, planet_params, jitter


def run_batch_simulations(config_path):
    """
    Runs a batch of simulations based on the config file and saves the results.

    This function reads the number of simulations to run from the config,
    then iteratively generates each simulation. The results are saved to
    an output directory specified in the config.

    - RV data is stored in a single HDF5 file.
    - All planet parameters are aggregated into a single CSV file.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    config = load_config(config_path)
    n_simulations = config.get('n_simulations', 1)
    output_dir = config.get('output_dir', 'simulation_results')

    os.makedirs(output_dir, exist_ok=True)

    rv_data_path = os.path.join(output_dir, 'batch_rv_data.h5')
    params_path = os.path.join(output_dir, 'batch_planet_params.csv')

    all_planet_params = []

    print(f"Generating {n_simulations} simulations...")
    print(f"RV data will be saved to: {rv_data_path}")
    print(f"Planet parameters will be saved to: {params_path}")

    # Using HDFStore to efficiently save DataFrames
    with pd.HDFStore(rv_data_path, mode='w') as store:
        for i in tqdm(range(n_simulations), desc="Generating Simulations"):
            # Generate one full simulation
            rv_data, planet_params, jitter = multiple_planet_rv(config_path)

            # Store the RV time-series data in the HDF5 file
            store.put(f'sim_{i:05d}', rv_data)

            # Prepare and collect planet parameters for the final CSV
            planet_params['simulation_id'] = i
            planet_params['jitter'] = jitter
            all_planet_params.append(planet_params)

    # Combine and save all planet parameters into a single CSV
    final_params_df = pd.concat(all_planet_params)
    final_params_df.to_csv(params_path)

    print("\nBatch simulation complete.")


def visualize_parameter_distributions(params_csv_path, output_dir):
    """
    Reads aggregated planet parameters and visualizes their distributions.

    This function generates and saves a grid of histograms to provide a
    visual check on the distribution of the randomly sampled parameters
    from the simulation batch.

    Args:
        params_csv_path (str): Path to the batch_planet_params.csv file.
        output_dir (str): Directory to save the plot image.
    """
    try:
        df = pd.read_csv(params_csv_path)
    except FileNotFoundError:
        print(f"Error: Parameters file not found at {params_csv_path}")
        print("Please run the simulations first to generate the file.")
        return

    # Create a 3x3 grid for the plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Distribution of Generated Simulation Parameters', fontsize=20)
    axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration

    # 1. Number of planets per system
    n_planets_dist = df.groupby('simulation_id').size()
    n_planets_dist.value_counts().sort_index().plot(kind='bar', ax=axes[0], rot=0, color='C0', edgecolor='black')
    axes[0].set_title('Number of Planets per System')
    axes[0].set_xlabel('Number of Planets')
    axes[0].set_ylabel('Count of Systems')

    # 2. Jitter (one value per simulation)
    df.groupby('simulation_id')['jitter'].first().hist(ax=axes[1], bins=40, color='C1', edgecolor='black')
    axes[1].set_title('Jitter Distribution')
    axes[1].set_xlabel('Jitter (m/s)')
    axes[1].set_ylabel('Count of Systems')

    # 3. & 4. Period (P) on linear and log scales
    df['P'].hist(ax=axes[2], bins=50, color='C2', edgecolor='black')
    axes[2].set_title('Period (P) - Linear Scale')
    axes[2].set_xlabel('Period (days)')

    df['P'].hist(ax=axes[3], bins=50, color='C2', edgecolor='black')
    axes[3].set_xscale('log')
    axes[3].set_title('Period (P) - Log Scale (Jeffreys Prior)')
    axes[3].set_xlabel('Period (days)')

    # 5. & 6. Semi-amplitude (K) on linear and log scales
    df['K'].hist(ax=axes[4], bins=50, color='C3', edgecolor='black')
    axes[4].set_title('Semi-amplitude (K) - Linear Scale')
    axes[4].set_xlabel('K (m/s)')

    df['K'].hist(ax=axes[5], bins=50, color='C3', edgecolor='black')
    axes[5].set_xscale('log')
    axes[5].set_title('Semi-amplitude (K) - Log Scale (Jeffreys Prior)')
    axes[5].set_xlabel('K (m/s)')

    # 7. & 8. Eccentricity (e) and Omega (ω)
    df['e'].hist(ax=axes[6], bins=40, color='C4', edgecolor='black')
    axes[6].set_title('Eccentricity (e) Distribution')
    axes[6].set_xlabel('Eccentricity')

    df['omega'].hist(ax=axes[7], bins=40, color='C5', edgecolor='black')
    axes[7].set_title('Argument of Periastron (ω) Distribution')
    axes[7].set_xlabel('Omega (radians)')

    # Hide the last unused subplot
    axes[8].set_visible(False)

    # General aesthetics
    for i, ax in enumerate(axes[:8]):
        if i > 1:  # The first two plots have a different y-axis label
            ax.set_ylabel('Count of Planets')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = os.path.join(output_dir, 'parameter_distributions.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nParameter distribution plot saved to: {plot_path}")
    plt.close(fig)  # Close the figure to free up memory


def load_simulation_data(simulation_id, output_dir):
    """
    Loads the data for a single simulation from the batch results.

    Args:
        simulation_id (int): The ID of the simulation to load (e.g., 0, 1, 2...).
        output_dir (str): The directory where the simulation results are stored.

    Returns:
        tuple: A tuple containing:
            - rv_data (pd.DataFrame): The time-series RV data for the simulation.
            - planet_params (pd.DataFrame): The orbital parameters for the simulation.
        Returns (None, None) if the data cannot be found.
    """
    params_path = os.path.join(output_dir, 'batch_planet_params.csv')
    rv_data_path = os.path.join(output_dir, 'batch_rv_data.h5')
    sim_key = f'sim_{simulation_id:05d}'

    try:
        # Load the specific RV data from the HDF5 file
        with pd.HDFStore(rv_data_path, mode='r') as store:
            if sim_key not in store.keys():
                print(f"Error: Simulation ID {simulation_id} ({sim_key}) not found in {rv_data_path}")
                return None, None
            rv_data = store[sim_key]

        # Load all parameters and filter for the specific simulation
        all_params = pd.read_csv(params_path, index_col=0)
        planet_params = all_params[all_params['simulation_id'] == simulation_id]

        return rv_data, planet_params

    except FileNotFoundError:
        print(f"Error: Could not find simulation files in '{output_dir}'. Please run the batch simulation first.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None, None


if __name__ == '__main__':
    # Define the path to the configuration file
    config_file = 'config.yaml'

    # Create an example config.yaml if it doesn't exist
    if not os.path.exists(config_file):
        print(f"Configuration file '{config_file}' not found. Creating an example file.")
        config_content = """
# Configuration for the synthetic RV data generator

# --- Batch Simulation Settings ---
n_simulations: 10000
output_dir: 'simulation_results' # Directory to save the output files

# --- General Simulation Settings ---
time_span: [0, 100]          # The start and end time of observations in days.
time_cadence_sec: 300        # The time between observations in seconds.
n_jobs: -1                   # Number of CPU cores for parallel processing (-1 means all).

# Planet generation settings
n_planets_range: [1, 5]      # The min and max number of planets to simulate.

# Orbital period
p_range: [1, 100]            # Min/max in days.
p_scale: 'log'               # 'log' to better cover the dynamic range (Jeffreys prior).

# RV semi-amplitude
k_range: [0.1, 100]          # Min/max in m/s.
k_scale: 'log'               # 'log' to better cover the dynamic range (Jeffreys prior).

# Eccentricity
e_range: [0.01, 0.5]
e_scale: 'linear'

# Argument of periastron (use float for 2*pi, approx 6.2832)
omega_range: [0, 6.2831853]  # Min/max in radians.
omega_scale: 'linear'

# --- Noise Settings (Jitter) ---
jitter_range: [0.1, 5.0]     # Min/max for Gaussian noise in m/s.
jitter_scale: 'linear'
"""
        with open(config_file, 'w') as f:
            f.write(config_content)

    # Run the batch simulation process
    run_batch_simulations(config_file)

    # Visualize the distributions of the generated parameters
    print("\nGenerating visualization of parameter distributions...")
    config = load_config(config_file)
    output_dir = config.get('output_dir', 'simulation_results')
    params_path = os.path.join(output_dir, 'batch_planet_params.csv')
    visualize_parameter_distributions(params_path, output_dir)

    # --- Example of loading and analyzing a single simulation ---
    print("\n--- Loading and analyzing a single simulation (ID=0) ---")
    sim_id_to_load = 0
    rv_data, planet_params = load_simulation_data(sim_id_to_load, output_dir)

    if rv_data is not None:
        print(f"\nParameters for Simulation ID: {sim_id_to_load}")
        print(planet_params)

        print(f"\nRV Data for Simulation ID: {sim_id_to_load} (first 5 rows)")
        print(rv_data.head())

        # You can now plot this specific simulation's data
        plt.figure(figsize=(15, 6))
        plt.title(f"RV Data for Simulation ID: {sim_id_to_load}")
        plt.plot(rv_data.index, rv_data['rv_with_noise'], 'k.', alpha=0.5, label='RV with Noise')
        plt.plot(rv_data.index, rv_data['rv_total_signal'], 'r-', lw=2, label='Total Keplerian Signal')
        plt.xlabel("Time [days]")
        plt.ylabel("Radial Velocity [m/s]")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        single_sim_plot_path = os.path.join(output_dir, f'simulation_{sim_id_to_load:05d}_plot.png')
        plt.savefig(single_sim_plot_path, dpi=150)
        print(f"\nPlot for simulation {sim_id_to_load} saved to: {single_sim_plot_path}")
        plt.close()
