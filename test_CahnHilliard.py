import numpy as np
from CahnHilliard import Cahn_Hilliard

def test_Cahn_Hilliard_shapes():
    # Create a test input array with a specific shape
    shape = (5, 5)  # Replace with the desired shape
    u = np.zeros(shape)  # Create a test input array with zeros and the specified shape

    # Set the parameters for the Cahn_Hilliard function
    t = 0.1  # Replace with the desired time value
    dx = 0.01  # Replace with the desired spatial step value
    D = 0.1  # Replace with the desired diffusion coefficient
    a = 0.01  # Replace with the desired parameter 'a'

    # Call the Cahn_Hilliard function
    result = Cahn_Hilliard(u, t, dx, D, a)

    # Check if the shape of the result matches the shape of the input
    assert result.shape == shape, "Test shape_match failed: Output shape does not match input shape."