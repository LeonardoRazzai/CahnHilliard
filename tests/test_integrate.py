import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
from CahnHilliard import integrate

def test_integrate_shape():
    """
    GIVEN a differential equation with initial condition u0
    WHEN integrated applying the integrate function with Nt time steps
    THEN the result has shape (Nt, u0.shape).
    """
    # GIVEN
    def harmonic_oscillator(u, t, omega):
        return np.array([u[1], -omega**2 * u[0]])

    # WHEN
    Nt = 100
    u0 = np.array([1.0, 0.0])  # Initial position and velocity
    omega = 1.0  # Frequency of the harmonic oscillator
    t = np.linspace(0.0, 1.0, Nt)
    result = integrate(harmonic_oscillator, u0, t, omega)

    # THEN
    expected_shape = (Nt, *u0.shape)  # Each time step has two values: position and velocity
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

def test_integrate_linear_decay():
    """
    GIVEN a simple linear decay differential equation
    WHEN integrated using the integrate function with an initial value of [1.0] and time from 0.0 to 1.0 with step 1e-2
    THEN the result matches the analytical solution with tolerance < 1e-2.
    """
    # GIVEN
    def linear_decay(u, t):
        return -u  # Simple linear decay

    # WHEN
    u0 = np.array([1.0])  # Make sure u0 is a NumPy array
    time_steps = np.linspace(0.0, 1.0, 100)
    result = integrate(linear_decay, u0, time_steps)

    # THEN
    expected_result = np.exp(-time_steps)
    np.testing.assert_allclose(result, expected_result.reshape((100, 1)), rtol=1e-2)


def test_integrate_sine_wave():
    """
    GIVEN a differential equation with a sine wave
    WHEN integrated using the integrate function with an initial value of [0.0] and time from 0.0 to 1.0 with step 1e-2
    THEN the result matches the analytical result with tolerance < 2e-2.
    """
    # GIVEN
    def sine_wave(u, t):
        return np.sin(t)  # Sine wave

    # WHEN
    u0 = np.array([-1])
    time_steps = np.linspace(0.0, 1, 100)
    result = integrate(sine_wave, u0, time_steps)

    # THEN
    expected_result = -np.cos(time_steps)
    np.testing.assert_allclose(result, expected_result.reshape((100, 1)), rtol=2e-2)
    

def test_integrate_harmonic_oscillator():
    """
    GIVEN a harmonic oscillator differential equation
    WHEN integrated using the integrate function with an initial value of [1.0, 0.0] and time from 0.0 to 1.0, with step 1e-2
    THEN the result matches the analytical solution [cos(t), -sin(t)] with tolerance < 2.5e-2.
    """
    # GIVEN
    def harmonic_oscillator(u, t, omega):
        # u[0] represents the position, u[1] represents the velocity
        return np.array([u[1], -omega**2 * u[0]])

    # WHEN
    u0 = np.array([1.0, 0.0])  # Initial position and velocity
    omega = 1.0  # Frequency of the harmonic oscillator
    t = np.linspace(0.0, 1.0, 100)
    result = integrate(harmonic_oscillator, u0, t, omega)

    # THEN
    expected_result = np.array([np.cos(t), -np.sin(t)])
    np.testing.assert_allclose(result.transpose(), expected_result, rtol=2.5e-2)