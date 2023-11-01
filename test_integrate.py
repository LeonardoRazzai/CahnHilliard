import numpy as np
from CahnHilliard import integrate, Cahn_Hilliard

def test_integrate_shape_match():
    # Create a test input array with a specific shape
    shape = (5, 5)  # Replace with the desired shape
    u0 = np.zeros(shape)  # Create a test initial condition array with zeros and the specified shape
    t = np.linspace(0, 0.1, 10)  # Create a test time array

    # Set other test parameters
    args = (0.1, 0.01, 0.01)  # Replace with the desired values for dx, D and a

    # Call the integrate function
    result = integrate(Cahn_Hilliard, u0, t, *args)

    # Check if the shape of the result matches the shape of the initial condition
    assert result.shape == (len(t),) + shape, "Test integrate_shape_match failed: Output shape does not match input shape."