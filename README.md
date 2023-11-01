# CahnHilliard
The module `CahnHilliard` is meant to be used to:
- Compute a numerical solution for the Cahn-Hilliard equation, i.e. the time evolution of the concentration field
- Choose different innitial conditions and physical parametrs (like diffusivity `D`)
- Compute time evolution of concentration, its distribution and its Fourier transform
- Create gifs to show the time evolution of these quantities


## Functions

### 1. `laplacian2`

Apply a 2D Laplacian filter to a given 2D array.

#### Parameters

- `state` (`np.ndarray`): A 2D NumPy array representing the input data or image on which the Laplacian filter will be applied.

#### Returns

- `np.ndarray`: A 2D NumPy array representing the filtered result after applying the Laplacian filter. The output has the same shape as the input.

#### Notes

The Laplacian filter is applied using a 3x3 kernel defined as follows:
[[ 0,  1,  0],
 [ 1, -4,  1],
 [ 0,  1,  0]]

The 'same' mode is used for convolution, ensuring that the output has the
same dimensions as the input. The 'symm' boundary condition is used for
handling boundary pixels.

#### Example:
  >>> input_image = np.array([[10, 20, 30],
  ...                         [40, 50, 60],
  ...                         [70, 80, 90]])
  >>> filtered_image = laplacian2(input_image)
  >>> print(filtered_image)
  array([[ -4,  -8,  -4],
          [-12,  40, -12],
          [ -4, -16,  -4]])

# To do
- Add a method to compute the initial growth rate of the fastest growing fourier component, to compare with small perturbation analysis result