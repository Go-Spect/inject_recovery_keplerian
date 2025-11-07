import numpy as np
import yaml
import logging
import os

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
    # Define the validation schema for the configuration
    config_schema = {
        'n_simulations': {'type': int, 'validator': lambda v: v > 0},
        'output_dir': {'type': str},
        'time_span': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: v[0] < v[1]},
        'time_cadence_sec': {'type': (int, float), 'validator': lambda v: v > 0},
        'n_jobs': {'type': int},
        'n_planets_range': {'type': list, 'list_len': 2, 'list_elem_type': int, 'validator': lambda v: 0 < v[0] <= v[1]},
        'p_range': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: 0 < v[0] < v[1]},
        'p_scale': {'type': str, 'validator': lambda v: v in ['linear', 'log']},
        'k_range': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: 0 < v[0] < v[1]},
        'k_scale': {'type': str, 'validator': lambda v: v in ['linear', 'log']},
        'e_range': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: 0 <= v[0] < v[1] < 1},
        'e_scale': {'type': str, 'validator': lambda v: v in ['linear', 'log']},
        'omega_range': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: 0 <= v[0] < v[1]},
        'omega_scale': {'type': str, 'validator': lambda v: v in ['linear', 'log']},
        'jitter_range': {'type': list, 'list_len': 2, 'list_elem_type': (int, float), 'validator': lambda v: 0 <= v[0] < v[1]},
        'jitter_scale': {'type': str, 'validator': lambda v: v in ['linear', 'log']},
        'n_rv_mosaic_examples': {'type': int, 'validator': lambda v: v > 0},
        'corner_plot_params': {'type': list, 'list_elem_type': str}
    }

    # Check for missing keys first
    missing_keys = set(config_schema.keys()) - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in config file: {sorted(list(missing_keys))}")

    # Validate each key against the schema
    for key, rules in config_schema.items():
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

    # Specific check for log scale ranges
    for key in ['p_range', 'k_range']:
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