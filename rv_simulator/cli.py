import os
import shutil
import logging
import argparse
from datetime import datetime
from .simulation import run_batch_simulations
from .analysis import (
    visualize_parameter_distributions,
    load_simulation_data,
    visualize_corner_plot,
    visualize_rv_mosaic
)
from .utils import load_config, validate_config, setup_logging
import matplotlib.pyplot as plt
import pandas as pd

def main():
    """Main entry point for the command-line interface."""
    try:
        # --- Setup Command-Line Argument Parsing ---
        parser = argparse.ArgumentParser(
            description="Run a batch of synthetic RV simulations based on a configuration file."
        )
        parser.add_argument(
            '--config',
            type=str,
            default='config.yaml',
            help='Path to the YAML configuration file (default: config.yaml)'
        )
        args = parser.parse_args()
        config_file = args.config

        config = load_config(config_file)

        # --- Create a unique output directory for this simulation run ---
        base_output_dir = config.get('output_dir', 'simulation_results')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)

        # --- Set up logging ---
        setup_logging(run_output_dir)
        logging.info(f"Created unique output directory: {run_output_dir}")

        # --- Validate the configuration before proceeding ---
        validate_config(config)
        logging.info("Configuration validated successfully.")

        # --- Archive the configuration file for reproducibility ---
        archived_config_path = os.path.join(run_output_dir, 'config.yaml')
        shutil.copy(config_file, archived_config_path)
        logging.info(f"Archived configuration to: {archived_config_path}")

        # --- Run the Batch Simulation ---
        all_rv_data, all_params_df = run_batch_simulations(archived_config_path)

        # --- Save all generated data to the unique directory ---
        rv_data_path = os.path.join(run_output_dir, 'batch_rv_data.h5')
        params_path = os.path.join(run_output_dir, 'batch_planet_params.csv')

        logging.info(f"Saving RV data to: {rv_data_path}")
        with pd.HDFStore(rv_data_path, mode='w') as store:
            for key, df in all_rv_data.items():
                store.put(key, df)

        logging.info(f"Saving planet parameters to: {params_path}")
        all_params_df.to_csv(params_path, index=False)
        logging.info("All data saved successfully.")

        # --- Post-processing and Verification ---
        logging.info("\n--- Generating Analysis Plots ---")
        visualize_parameter_distributions(params_path, run_output_dir)

        corner_params = config.get('corner_plot_params', ['P', 'K', 'e', 'omega'])
        visualize_corner_plot(params_path, run_output_dir, corner_params)

        n_mosaic = config.get('n_rv_mosaic_examples', 9)
        visualize_rv_mosaic(n_mosaic, run_output_dir) # Pass the correct run directory

        logging.info("\n--- Example: Loading and analyzing a single simulation (ID=0) ---")
        sim_id_to_load = 0
        rv_data, planet_params = load_simulation_data(sim_id_to_load, run_output_dir) # Pass the correct run directory

        if rv_data is not None:
            logging.info(f"\nInput Parameters for Simulation ID: {sim_id_to_load}\n{planet_params}")
            logging.info(f"\nResulting RV Data for Simulation ID: {sim_id_to_load} (first 5 rows)\n{rv_data.head()}")

            plt.figure(figsize=(15, 6))
            plt.title(f"RV Data for Simulation ID: {sim_id_to_load}")
            plt.plot(rv_data.index, rv_data['rv_with_noise'], 'k.', alpha=0.5, label='RV with Noise')
            plt.plot(rv_data.index, rv_data['rv_total_signal'], 'r-', lw=2, label='Total Keplerian Signal')
            plt.xlabel("Time [days]")
            plt.ylabel("Radial Velocity [m/s]")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            single_sim_plot_path = os.path.join(run_output_dir, f'example_simulation_{sim_id_to_load:05d}_plot.png')
            plt.savefig(single_sim_plot_path, dpi=150)
            logging.info(f"\nPlot for simulation {sim_id_to_load} saved to: {single_sim_plot_path}")
            plt.close()

    except Exception as e:
        logging.critical("A critical error occurred, and the program had to stop.")
        logging.exception(e) # This will log the full traceback of the error

if __name__ == '__main__':
    main()