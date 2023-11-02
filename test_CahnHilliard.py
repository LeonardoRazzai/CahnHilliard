import numpy as np
from CahnHilliard import Cahn_Hilliard

def test_Cahn_Hilliard_shapes():
    # Create a test input array with zeros and the specified shape
    shape = (5, 5)
    u = np.zeros(shape)

    # Set the parameters for the Cahn_Hilliard function
    t = 0.1  # time value
    dx = 0.01  # spatial step value
    D = 0.01  # diffusion coefficient
    a = 0.001  # interfacial parameter

    # Call the Cahn_Hilliard function
    result = Cahn_Hilliard(u, t, dx, D, a)

    # Check if the shape of the result matches the shape of the input
    assert result.shape == shape, "Test shape_match failed: Output shape does not match input shape."
    # Check if all zeros input gives all zeros output
    np.testing.assert_array_almost_equal(u, result, decimal=8), "Test result values failed: Input with all zeros doesn't give output with all zeros."