import numpy as np
from CahnHilliard import laplacian2

def test_laplacian2_shapes():
    '''Test input and output shape matching'''
    # Test case 1: Square input matrix
    input_field = np.array([[10, 20, 30],
                            [40, 50, 60],
                            [70, 80, 90]])
    result = laplacian2(input_field)
    assert result.shape == input_field.shape, "Test case 1 failed"

    # Test case 2: Rectangular input matrix
    input_field = np.array([[10, 20],
                            [30, 40],
                            [50, 60]])
    result = laplacian2(input_field)
    assert result.shape == input_field.shape, "Test case 2 failed"
    
def test_constant_field():
    '''Testing that constant input field gives all zeros result'''
    shape = (6, 6)
    # set same shape for input field and all zeros field
    input_field = np.ones(shape)
    zeros = np.zeros(shape)
    result = laplacian2(input_field)
    
    np.testing.assert_equal(zeros, result)