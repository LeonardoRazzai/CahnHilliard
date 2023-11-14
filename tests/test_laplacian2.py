import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
from CahnHilliard import laplacian2

def test_laplacian2_input_output_shapes():
    """
    GIVEN an input field with all zeros
    WHEN the laplacian2 function is applied
    THEN the shape of the result is equal to the shape of the input.
    """
    input_field = np.zeros((5, 5))  # Replace with your desired input shape
    result = laplacian2(input_field)

    np.testing.assert_equal(input_field.shape, result.shape, "Test Failed: Input and output shapes are not equal")


def test_laplacian2_with_central_peak():
    """
    GIVEN an central peaked input field
    WHEN the laplacian2 function is applied
    THEN the result is the expected Laplacian-filtered output.
    """
    input_field = np.array([[0, 1, 0],
                            [1, 3, 1],
                            [0, 1, 0]])

    expected_output = np.array([[ 2,  0,  2],
                                [ 0, -8,  0],
                                [ 2,  0,  2]])

    result = laplacian2(input_field)

    np.testing.assert_array_equal(result, expected_output, "Test Failed: Central peak")
    

def test_laplacian2_with_zeros():
    """
    GIVEN an input field containing all zeros
    WHEN the laplacian2 function is applied
    THEN the result is an array of zeros.
    """
    input_field = np.zeros((10, 10))
    result = laplacian2(input_field)
    expected_output = np.zeros((10, 10))
    
    np.testing.assert_array_equal(result, expected_output, "Test Failed: Zeros input")

def test_laplacian2_with_ones():
    """
    GIVEN an input field containing all ones
    WHEN the laplacian2 function is applied
    THEN the result is an array of zeros.
    """
    input_field = np.ones((10, 10))
    result = laplacian2(input_field)
    expected_output = np.zeros((10, 10))
    
    np.testing.assert_array_equal(result, expected_output, "Test Failed: Ones input")