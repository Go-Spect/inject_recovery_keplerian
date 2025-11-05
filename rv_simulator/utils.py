import numpy as np
import yaml

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