import pandas as pd
import os
import matplotlib.pyplot as plt

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

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Distribution of Generated Simulation Parameters', fontsize=20)
    axes = axes.ravel()

    n_planets_dist = df.groupby('simulation_id').size()
    n_planets_dist.value_counts().sort_index().plot(kind='bar', ax=axes[0], rot=0, color='C0', edgecolor='black')
    axes[0].set_title('Number of Planets per System')
    axes[0].set_xlabel('Number of Planets')
    axes[0].set_ylabel('Count of Systems')

    df.groupby('simulation_id')['jitter'].first().hist(ax=axes[1], bins=40, color='C1', edgecolor='black')
    axes[1].set_title('Jitter Distribution')
    axes[1].set_xlabel('Jitter (m/s)')
    axes[1].set_ylabel('Count of Systems')

    df['P'].hist(ax=axes[2], bins=50, color='C2', edgecolor='black')
    axes[2].set_title('Period (P) - Linear Scale')
    axes[2].set_xlabel('Period (days)')

    df['P'].hist(ax=axes[3], bins=50, color='C2', edgecolor='black')
    axes[3].set_xscale('log')
    axes[3].set_title('Period (P) - Log Scale (Jeffreys Prior)')
    axes[3].set_xlabel('Period (days)')

    df['K'].hist(ax=axes[4], bins=50, color='C3', edgecolor='black')
    axes[4].set_title('Semi-amplitude (K) - Linear Scale')
    axes[4].set_xlabel('K (m/s)')

    df['K'].hist(ax=axes[5], bins=50, color='C3', edgecolor='black')
    axes[5].set_xscale('log')
    axes[5].set_title('Semi-amplitude (K) - Log Scale (Jeffreys Prior)')
    axes[5].set_xlabel('K (m/s)')

    df['e'].hist(ax=axes[6], bins=40, color='C4', edgecolor='black')
    axes[6].set_title('Eccentricity (e) Distribution')
    axes[6].set_xlabel('Eccentricity')

    df['omega'].hist(ax=axes[7], bins=40, color='C5', edgecolor='black')
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
    print(f"\nParameter distribution plot saved to: {plot_path}")
    plt.close(fig)

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
        with pd.HDFStore(rv_data_path, mode='r') as store:
            if sim_key not in store.keys():
                print(f"Error: Simulation ID {simulation_id} ({sim_key}) not found in {rv_data_path}")
                return None, None
            rv_data = store[sim_key]

        all_params = pd.read_csv(params_path, index_col=0)
        planet_params = all_params[all_params['simulation_id'] == simulation_id]

        return rv_data, planet_params

    except FileNotFoundError:
        print(f"Error: Could not find simulation files in '{output_dir}'. Please run the batch simulation first.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None, None