import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# --- Nomenclature Standardization for Output Files ---

# Defines the final column names for the ground truth parameters CSV file.
NAME_MAP_CSV_COLUMNS = {
    'P': 'period_days',
    'K': 'semi_amplitude_ms',
    'e': 'eccentricity',
    'omega': 'arg_periastron_rad',
    'T0': 'time_periastron_bjd',
    'jitter': 'jitter_ms'
}

# Defines the final column names for the time-series data in the HDF5 file.
NAME_MAP_H5_COLUMNS = {
    'rv_with_noise': 'RV [m/s]',
    'rv_total_signal': 'rv_model_ms'
}

def save_results(all_rv_data, all_params_df, run_output_dir):
    """
    Saves the simulation results to disk with standardized column names.

    This function renames the columns of the parameters DataFrame and each
    RV data DataFrame according to the ESSP4/ML framework standard before
    saving them to CSV and HDF5 files.

    Args:
        all_rv_data (dict): Dictionary of RV data DataFrames.
        all_params_df (pd.DataFrame): DataFrame of all planet parameters.
        run_output_dir (str): The directory to save the output files.
    """
    rv_data_path = os.path.join(run_output_dir, 'batch_rv_data.h5')
    params_path = os.path.join(run_output_dir, 'batch_planet_params.csv')

    # --- Save RV Time Series Data (HDF5) ---
    logging.info(f"Saving standardized RV data to: {rv_data_path}")
    with pd.HDFStore(rv_data_path, mode='w') as store:
        for key, df in all_rv_data.items():
            # Rename columns to the standard format before saving
            df_renamed = df.rename(columns=NAME_MAP_H5_COLUMNS)
            store.put(key, df_renamed)

    # --- Save Planet Parameters (CSV) ---
    logging.info(f"Saving standardized planet parameters to: {params_path}")
    # Create a copy to avoid SettingWithCopyWarning if the df is used later
    params_df_renamed = all_params_df.copy()
    # Rename columns to the standard format before saving
    params_df_renamed.rename(columns=NAME_MAP_CSV_COLUMNS, inplace=True)
    # Ensure simulation_id is an integer
    if 'simulation_id' in params_df_renamed.columns:
        params_df_renamed['simulation_id'] = params_df_renamed['simulation_id'].astype(int)

    params_df_renamed.to_csv(params_path, index=False)
    logging.info("All data saved successfully with standard nomenclature.")

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
    except FileNotFoundError as e:
        logging.error(f"Parameters file not found at {params_csv_path}. Cannot generate distribution plot.", exc_info=True)
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Distribution of Generated Simulation Parameters', fontsize=20)
    axes = axes.ravel()

    n_planets_dist = df.groupby('simulation_id').size()
    n_planets_dist.value_counts().sort_index().plot(kind='bar', ax=axes[0], rot=0, color='C0', edgecolor='black')
    axes[0].set_title('Number of Planets per System')
    axes[0].set_xlabel('Number of Planets')
    axes[0].set_ylabel('Count of Systems')

    df.groupby('simulation_id')['jitter_ms'].first().hist(ax=axes[1], bins=40, color='C1', edgecolor='black')
    axes[1].set_title('Jitter Distribution')
    axes[1].set_xlabel('Jitter (m/s)')
    axes[1].set_ylabel('Count of Systems')

    df['period_days'].hist(ax=axes[2], bins=50, color='C2', edgecolor='black')
    axes[2].set_title('Period (P) - Linear Scale')
    axes[2].set_xlabel('Period (days)')

    df['period_days'].hist(ax=axes[3], bins=50, color='C2', edgecolor='black')
    axes[3].set_xscale('log')
    axes[3].set_title('Period (P) - Log Scale (Jeffreys Prior)')
    axes[3].set_xlabel('Period (days)')

    df['semi_amplitude_ms'].hist(ax=axes[4], bins=50, color='C3', edgecolor='black')
    axes[4].set_title('Semi-amplitude (K) - Linear Scale')
    axes[4].set_xlabel('K (m/s)')

    df['semi_amplitude_ms'].hist(ax=axes[5], bins=50, color='C3', edgecolor='black')
    axes[5].set_xscale('log')
    axes[5].set_title('Semi-amplitude (K) - Log Scale (Jeffreys Prior)')
    axes[5].set_xlabel('K (m/s)')

    df['eccentricity'].hist(ax=axes[6], bins=40, color='C4', edgecolor='black')
    axes[6].set_title('Eccentricity (e) Distribution')
    axes[6].set_xlabel('Eccentricity')

    df['arg_periastron_rad'].hist(ax=axes[7], bins=40, color='C5', edgecolor='black')
    axes[7].set_title('Argument of Periastron (ω) Distribution')
    axes[7].set_xlabel('Omega (radians)')

    axes[8].set_visible(False)

    for i, ax in enumerate(axes[:8]):
        if i > 1:
            ax.set_ylabel('Count of Planets')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = os.path.join(output_dir, 'parameter_distributions.png')
    plt.savefig(plot_path, dpi=150)
    logging.info(f"Parameter distribution plot saved to: {plot_path}")
    plt.close(fig)

def visualize_corner_plot(params_csv_path, output_dir, corner_params):
    """
    Generates and saves a corner plot (pairplot) of specified parameters.

    Args:
        params_csv_path (str): Path to the batch_planet_params.csv file.
        output_dir (str): Directory to save the plot image.
        corner_params (list): A list of column names to include in the plot.
    """
    try:
        df = pd.read_csv(params_csv_path)
    except FileNotFoundError:
        logging.error(f"Parameters file for corner plot not found at {params_csv_path}", exc_info=True)
        return

    if not all(param in df.columns for param in corner_params):
        logging.error(f"One or more specified corner_params not found in the CSV file.")
        return

    logging.info("Generating corner plot...")
    sns.set_theme(style="ticks")

    pair_plot = sns.pairplot(df[corner_params], diag_kind='hist', corner=True)
    pair_plot.fig.suptitle('Corner Plot of Simulated Planet Parameters', y=1.02, fontsize=20)

    plot_path = os.path.join(output_dir, 'corner_plot.png')
    plt.savefig(plot_path, dpi=150)
    logging.info(f"Corner plot saved to: {plot_path}")
    plt.close(pair_plot.fig)
    
def visualize_rv_mosaic(n_examples, run_output_dir):
    """
    Generates a mosaic of example RV curves from the simulation batch.

    Args:
        n_examples (int): The number of example plots to generate.
        output_dir (str): The directory where simulation results are stored.
    """
    logging.info("Generating RV curve mosaic...")
    # Determine grid size (e.g., 9 examples -> 3x3 grid)
    grid_size = int(np.sqrt(n_examples))
    if grid_size**2 != n_examples:
        logging.warning(f"n_examples ({n_examples}) is not a perfect square. Using the largest possible square grid.")
        n_examples = grid_size**2

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(18, 18), sharex=True, sharey=True)
    fig.suptitle('Example RV Curves from Simulations', fontsize=24)
    axes = axes.ravel()

    for i in range(n_examples):
        sim_id = i
        rv_data, planet_params = load_simulation_data(sim_id, run_output_dir)

        if rv_data is None:
            axes[i].text(0.5, 0.5, f'Sim ID {sim_id}\nNot Found', ha='center', va='center')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            continue

        n_planets = len(planet_params)
        jitter = planet_params['jitter_ms'].iloc[0]

        axes[i].plot(rv_data.index, rv_data['RV [m/s]'], 'k.', markersize=2, alpha=0.6)
        axes[i].plot(rv_data.index, rv_data['rv_model_ms'], 'r-', lw=1.5)
        axes[i].set_title(f'Sim ID: {sim_id} ({n_planets} Pl, Jitter: {jitter:.2f} m/s)', fontsize=10)

    # Common labels
    fig.text(0.5, 0.07, 'Time [days]', ha='center', va='center', fontsize=18)
    fig.text(0.08, 0.5, 'Radial Velocity [m/s]', ha='center', va='center', rotation='vertical', fontsize=18)

    plt.tight_layout(rect=[0.08, 0.08, 1, 0.95])
    plot_path = os.path.join(run_output_dir, 'rv_mosaic_plot.png')
    plt.savefig(plot_path, dpi=150)
    logging.info(f"RV mosaic plot saved to: {plot_path}")
    plt.close(fig)

def load_simulation_data(simulation_id, run_output_dir):
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
    params_path = os.path.join(run_output_dir, 'batch_planet_params.csv')
    rv_data_path = os.path.join(run_output_dir, 'batch_rv_data.h5')
    sim_key = f'sim_{simulation_id:05d}'

    try:
        with pd.HDFStore(rv_data_path, mode='r') as store:
            if sim_key not in store.keys():
                logging.error(f"Simulation ID {simulation_id} ({sim_key}) not found in {rv_data_path}")
                return None, None
            rv_data = store[sim_key]

        all_params = pd.read_csv(params_path) # Removed index_col=0
        planet_params = all_params[all_params['simulation_id'] == simulation_id]

        return rv_data, planet_params

    except FileNotFoundError:
        logging.error(f"Could not find simulation files in '{run_output_dir}'. Please run the batch simulation first.", exc_info=True)
        return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading data for sim_id {simulation_id}.", exc_info=True)
        return None, None