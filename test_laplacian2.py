import numpy as np
from CahnHilliard import laplacian2

def test_laplacian2_shapes():
    # Test case 1: Square input matrix
    input_image = np.array([[10, 20, 30],
                            [40, 50, 60],
                            [70, 80, 90]])
    result = laplacian2(input_image)
    assert result.shape == input_image.shape, "Test case 1 failed"

    # Test case 2: Rectangular input matrix
    input_image = np.array([[10, 20],
                            [30, 40],
                            [50, 60]])
    result = laplacian2(input_image)
    assert result.shape == input_image.shape, "Test case 2 failed"