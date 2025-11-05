import os
from rv_simulator.simulation import run_batch_simulations
from rv_simulator.analysis import visualize_parameter_distributions, load_simulation_data
from rv_simulator.utils import load_config
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Define the path to the configuration file
    config_file = 'config.yaml'

    # --- Run the Batch Simulation ---
    # This will generate the HDF5 and CSV output files.
    run_batch_simulations(config_file)

    # --- Post-processing and Verification ---
    config = load_config(config_file)
    output_dir = config.get('output_dir', 'simulation_results')
    params_path = os.path.join(output_dir, 'batch_planet_params.csv')

    # 1. Visualize the distributions of all generated parameters
    print("\nGenerating visualization of parameter distributions...")
    visualize_parameter_distributions(params_path, output_dir)

    # 2. Example of loading and analyzing a single simulation
    print("\n--- Example: Loading and analyzing a single simulation (ID=0) ---")
    sim_id_to_load = 0
    rv_data, planet_params = load_simulation_data(sim_id_to_load, output_dir)

    if rv_data is not None:
        print(f"\nInput Parameters for Simulation ID: {sim_id_to_load}")
        print(planet_params)

        print(f"\nResulting RV Data for Simulation ID: {sim_id_to_load} (first 5 rows)")
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