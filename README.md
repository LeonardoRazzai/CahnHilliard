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
```markdown
[[ 0,  1,  0],
 [ 1, -4,  1],
 [ 0,  1,  0]]
```
The 'same' mode is used for convolution, ensuring that the output has the
same dimensions as the input. The 'symm' boundary condition is used for
handling boundary pixels.

### 2. `Cahn_Hilliard`

Calculate the Cahn-Hilliard equation for phase separation in a 2D system.

#### Parameters

- `u` (`np.ndarray`): A 2D NumPy array representing the concentration field at the current time.
- `t` (`float`): The current time.
- `dx` (`float`): Spatial step size (grid spacing) in the x and y directions.
- `D` (`float`): Diffusion coefficient controlling the rate of phase separation.
- `a` (`float`): Parameter affecting the interfacial energy between phases.

#### Returns

- `np.ndarray`: A 2D NumPy array representing the updated concentration field 'u' at the next time step 't+dt,' where 'dt' is determined by the numerical scheme used.

#### Notes

The Cahn-Hilliard equation is defined as follows:

```markdown
du/dt = (D / dx^2) * laplacian2(u^3 - u - (a / dx^2) * laplacian2(u))
```
Where laplacian2(u) represents the Laplacian of 'u' calculated using the 'laplacian2' function.

The function calculates the right-hand side of the equation for the given concentration field 'u', time 't', spatial step size 'dx', diffusion coefficient 'D', and interfacial energy parameter 'a'.

#### Example

```python
>>> initial_concentration = np.random.rand(100, 100)
>>> t = 0.0
>>> dx = 0.1
>>> D = 0.1
>>> a = 1.0
>>> updated_concentration = Cahn_Hilliard(initial_concentration, t, dx, D, a)
```

### 3. `integrate`

Numerically integrate a given differential equation over time using the Euler method.

#### Parameters

- `func` (`callable`): The function representing the differential equation to be integrated. It should accept the following arguments:
  - `u` (`np.ndarray`): The current state of the system.
  - `t` (`float`): The current time.
  - `*args` (`tuple`): Additional arguments to be passed to `func`. The function should return the rate of change of 'u' at the given time 't'.
- `u0` (`np.ndarray`): A 2D NumPy array representing the initial state of the system.
- `t` (`np.ndarray`): A 1D NumPy array containing the discrete time steps at which to compute the solution.
- `*args` (`tuple`, optional): Additional arguments to be passed to `func`.

#### Returns

- `np.ndarray`: A 3D NumPy array representing the solution of the differential equation over time. The first dimension corresponds to time steps, and the shape of each step matches the shape of 'u0.'


# To do
- Add a method to compute the initial growth rate of the fastest growing fourier component, to compare with small perturbation analysis result