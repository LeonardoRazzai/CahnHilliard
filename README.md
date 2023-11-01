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

## 2. `Cahn_Hilliard`

Calculate the Cahn-Hilliard equation for phase separation in a 2D system.

### Parameters

- `u` (`np.ndarray`): A 2D NumPy array representing the concentration field at the current time.
- `t` (`float`): The current time.
- `dx` (`float`): Spatial step size (grid spacing) in the x and y directions.
- `D` (`float`): Diffusion coefficient controlling the rate of phase separation.
- `a` (`float`): Parameter affecting the interfacial energy between phases.

### Returns

- `np.ndarray`: A 2D NumPy array representing the updated concentration field 'u' at the next time step 't+dt,' where 'dt' is determined by the numerical scheme used.

### Notes

The Cahn-Hilliard equation is defined as follows:

```markdown
du/dt = (D / dx^2) * laplacian2(u^3 - u - (a / dx^2) * laplacian2(u))

# To do
- Add a method to compute the initial growth rate of the fastest growing fourier component, to compare with small perturbation analysis result