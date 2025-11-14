import numpy as np
import pandas as pd
import yaml
import logging
import os
import warnings

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
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Configuration file not found at {config_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e

def load_time_array(filepath, column_name=None):
    """
    Loads a time array from a file, supporting .npy, .txt, or .csv.

    Args:
        filepath (str): The path to the file.
        column_name (str, optional): The name of the column to use if the file
                                     is a .csv. Defaults to None.

    Returns:
        np.ndarray: The loaded time array.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is unsupported or if a column_name is
                    required for a .csv but not provided.
        IOError: If there is an error reading a .csv file.
        KeyError: If the specified column_name is not in the .csv file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Timestamp file not found: {filepath}")

    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.txt'):
        return np.loadtxt(filepath)
    elif filepath.endswith('.csv'):
        if column_name is None:
            raise ValueError("For .csv files, 'timestamps_file_column' is required.")
        try:
            df = pd.read_csv(filepath)
            return df[column_name].values
        except KeyError:
            raise KeyError(f"Column '{column_name}' not found in {filepath}. Available columns: {df.columns.tolist()}")
        except Exception as e:
            raise IOError(f"Failed to read CSV {filepath}: {e}")
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .npy, .txt, or .csv.")

# --- Nomenclature Translation Layer (Adapter Pattern) ---

NAME_MAP_CONFIG = {
    'p_range': 'period_days_range',
    'p_scale': 'period_days_scale',
    'k_range': 'semi_amplitude_ms_range',
    'k_scale': 'semi_amplitude_ms_scale',
    'e_range': 'eccentricity_range',
    'e_scale': 'eccentricity_scale',
    'omega_range': 'arg_periastron_rad_range',
    'omega_scale': 'arg_periastron_rad_scale',
    'jitter_range': 'jitter_ms_range',
    'jitter_scale': 'jitter_ms_scale'
}

def translate_config(config):
    """
    Translates old config keys to new ones in-memory for retro-compatibility.
    Issues a DeprecationWarning if an old key is used.
    """
    for old_key, new_key in NAME_MAP_CONFIG.items():
        if old_key in config:
            # Issue a warning to the user
            warnings.warn(
                f"Config key '{old_key}' is deprecated and will be removed in a future version. "
                f"Please use '{new_key}' instead.",
                DeprecationWarning
            )
            # Translate the key
            config[new_key] = config.pop(old_key)
    return config

def validate_config(config):
    """
    Validates the configuration dictionary against a schema for required keys,
    data types, and value constraints.

    Args:
        config (dict): The configuration dictionary loaded from the YAML file.

    Raises:
        ValueError: If a key is missing.
        TypeError: If a value has the wrong data type.
        ValueError: If a value fails a constraint check (e.g., is out of range).
    """
    # --- 1. Translate Legacy Keys (Adapter Pattern) ---
    config = translate_config(config)

    # --- Schema Definition ---
    # Note: time_span and time_cadence_sec are now optional at this stage
    # as they are only required for the 'internal' time generation mode.
    # Their presence will be validated conditionally later.
    # Define the validation schema for the configuration
    config_schema = {
        'n_simulations': {'type': int, 'validator': lambda v: v > 0},
        'output_dir': {'type': str},
        'n_jobs': {'type': int},
        'time_span': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: v[0] < v[1], 'optional': True},
        'time_cadence_sec': {'type': (int, float), 'validator': lambda v: v > 0, 'optional': True},
        'external_timestamps_file': {'type': str, 'optional': True},
        'timestamps_file_column': {'type': str, 'optional': True},
        'n_planets_range': {'type': list, 'list_len': 2, 'list_elem_type': int, 'validator': lambda v: 0 <= v[0] <= v[1]}, # No name change
        'period_days_range': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: 0 < v[0] < v[1]},
        'period_days_scale': {'type': str, 'validator': lambda v: v in ['linear', 'log']},
        'semi_amplitude_ms_range': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: 0 < v[0] < v[1]},
        'semi_amplitude_ms_scale': {'type': str, 'validator': lambda v: v in ['linear', 'log']},
        'eccentricity_range': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: 0 <= v[0] < v[1] < 1},
        'eccentricity_scale': {'type': str, 'validator': lambda v: v in ['linear', 'log']},
        'arg_periastron_rad_range': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: 0 <= v[0] < v[1]},
        'arg_periastron_rad_scale': {'type': str, 'validator': lambda v: v in ['linear', 'log']},
        'apply_jitter': {'type': bool, 'optional': True},
        'jitter_ms_range': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: 0 <= v[0] < v[1], 'optional': True},
        'jitter_ms_scale': {'type': str, 'validator': lambda v: v in ['linear', 'log'], 'optional': True},
        'n_rv_mosaic_examples': {'type': int, 'validator': lambda v: v > 0},
        'corner_plot_params': {'type': list, 'list_elem_type': str}
    }

    # --- Initial Validation Pass ---
    # Check for missing required keys
    required_keys = {k for k, v in config_schema.items() if not v.get('optional')}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in config file: {sorted(list(missing_keys))}")

    # Validate each key against the schema
    for key, rules in config_schema.items():
        if key not in config:
            continue # Skip optional keys that are not present

        value = config[key]
        
        # 1. Validate Type
        if not isinstance(value, rules['type']):
            raise TypeError(f"Config Error: '{key}' must be of type {rules['type'].__name__}, but got {type(value).__name__}.")

        # 2. Validate List-specific rules (length and element types)
        if rules['type'] is list:
            if 'list_len' in rules and len(value) != rules['list_len']:
                raise ValueError(f"Config Error: '{key}' must be a list of length {rules['list_len']}, but got length {len(value)}.")
            if 'list_elem_type' in rules:
                for i, elem in enumerate(value):
                    if not isinstance(elem, rules['list_elem_type']):
                        raise TypeError(f"Config Error: Element {i} of '{key}' must be of type {rules['list_elem_type']}, but got {type(elem).__name__}.")

        # 3. Validate Value Constraints
        if 'validator' in rules and not rules['validator'](value):
            raise ValueError(f"Config Error: Value for '{key}' ({value}) is not valid. Please check the constraints.")

    # --- Conditional Validation for Time Generation ---
    external_file = config.get('external_timestamps_file')
    time_span = config.get('time_span')
    time_cadence = config.get('time_cadence_sec')

    if external_file:
        # EXTERNAL MODE
        if not os.path.exists(external_file):
            raise FileNotFoundError(f"Timestamp file not found: {external_file}")
        
        if time_span or time_cadence:
            logging.warning(f"Warning: 'external_timestamps_file' provided. Ignoring 'time_span' and 'time_cadence_sec'.")

        # Extra validation for CSV
        if external_file.endswith('.csv') and not config.get('timestamps_file_column'):
            raise ValueError("Using a .csv for timestamps, but 'timestamps_file_column' was not defined in the config.")
    else:
        # INTERNAL MODE
        if not time_span or not time_cadence:
            raise ValueError("Internal time generation mode: 'time_span' and 'time_cadence_sec' are required when 'external_timestamps_file' is not provided.")

    # --- Conditional Validation for Jitter ---
    # Default to True for retro-compatibility if the key is missing.
    apply_jitter = config.get('apply_jitter', True)

    if apply_jitter:
        # JITTER ON: jitter_range and jitter_scale become mandatory.
        if config.get('jitter_ms_range') is None or config.get('jitter_ms_scale') is None:
            raise ValueError("'apply_jitter' is true (or default), but 'jitter_ms_range' or 'jitter_ms_scale' is missing from the config.")
    else:
        # JITTER OFF: Warn if the user provided unnecessary keys.
        if config.get('jitter_ms_range') or config.get('jitter_ms_scale'):
            logging.warning("Warning: 'apply_jitter' is false. Ignoring 'jitter_ms_range' and 'jitter_ms_scale'.")



    # Specific check for log scale ranges
    for key in ['period_days_range', 'semi_amplitude_ms_range']:
        scale_key = key.replace('_range', '_scale')
        if config[scale_key] == 'log' and config[key][0] <= 0:
            raise ValueError(f"Config Error: When '{scale_key}' is 'log', the minimum of '{key}' must be > 0.")

    logging.info("Configuration file validated successfully.")

def setup_logging(log_dir):
    """
    Configures the root logger to output to both a file and the console.

    Args:
        log_dir (str): The directory where the log file will be saved.
    """
    log_filename = os.path.join(log_dir, 'simulation.log')

    # Configure logging to write to a file and to the console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler() # This handler sends logs to the console (stdout/stderr)
        ]
    )

    # Get the root logger
    logger = logging.getLogger()
    logger.info("Logging configured. All messages will be sent to the console and %s", log_filename)