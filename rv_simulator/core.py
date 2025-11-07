import numpy as np

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
        
    # Ensure E is a float array for the operations
    E = np.copy(mean_anomaly).astype(float)

    for _ in range(max_iter):
        # Numerical stability: Add a small epsilon to the denominator to prevent division by zero
        # in the edge case where eccentricity is 1 and cos(E) is 1.
        denominator = 1 - eccentricity * np.cos(E)
        # Avoid division by zero if the denominator is extremely close to 0
        delta_E = (E - eccentricity * np.sin(E) - mean_anomaly) / np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
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