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

- `np.ndarray`: A 2D NumPy array representing the updated concentration field `u` at the next time step `t+dt,` where `dt` is determined by the numerical scheme used.

#### Notes

The Cahn-Hilliard equation is defined as follows:

```markdown
du/dt = (D / dx^2) * laplacian2(u^3 - u - (a / dx^2) * laplacian2(u))
```
Where laplacian2(u) represents the Laplacian of `u` calculated using the `laplacian2` function.

The function calculates the right-hand side of the equation for the given concentration field `u`, time `t`, spatial step size `dx`, diffusion coefficient `D`, and interfacial energy parameter `a`.

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
  - `*args` (`tuple`): Additional arguments to be passed to `func`. The function should return the rate of change of `u` at the given time `t`.
- `u0` (`np.ndarray`): A 2D NumPy array representing the initial state of the system.
- `t` (`np.ndarray`): A 1D NumPy array containing the discrete time steps at which to compute the solution.
- `*args` (`tuple`, optional): Additional arguments to be passed to `func`.

#### Returns

- `np.ndarray`: A 3D NumPy array representing the solution of the differential equation over time. The first dimension corresponds to time steps, and the shape of each step matches the shape of `u0`.


### 4. `Sol_CahnHilliard`

A class for simulating and analyzing solutions of the Cahn-Hilliard equation.

#### Parameters

- `L` (`float`): Length of the spatial domain.
- `N` (`int`): Number of spatial grid points.
- `D` (`float`): Diffusion coefficient controlling the rate of phase separation.
- `a` (`float`): Parameter affecting the interfacial energy between phases.

#### Attributes

- `D` (`float`): Diffusion coefficient.
- `a` (`float`): Interfacial energy parameter.
- `x` (`np.ndarray`): 1D NumPy array representing the spatial grid.
- `sol` (`np.ndarray`): 3D NumPy array representing the concentration field over time.
- `t` (`np.ndarray`): 1D NumPy array representing time steps.
- `step` (`int`): Step size for data analysis and visualization. Default value is 10.
- `ft_sol` (`np.ndarray`): 2D NumPy array representing the Fourier transformed concentration profiles.
- `ft_t` (`np.ndarray`): 1D NumPy array representing time steps for Fourier analysis.
- `histo` (`list`): List of histograms representing concentration distributions over time.

#### Methods

##### `compute_sol(self, c0, t)`

Simulate the evolution of the concentration field.

- `c0` (`np.ndarray`): Initial concentration field.
- `t` (`np.ndarray`): 1D NumPy array representing time steps.

##### `set_step(self, step)`

Set the step size for data analysis and visualization.

- `step` (`int`): New step size.

##### `ComputeFT(self)`

Compute the average Fourier transform of concentration profiles along x and y at times specified by `self.step`.

##### `Compute_histo(self)`

Compute histograms of concentration at times specified by `self.step`.

##### `MakeGif_sol(self, file_name='cahn_hilliard.gif')`

Create a GIF animation of concentration field evolution.

- `file_name` (`str`, optional): Name of the output GIF file.

##### `MakeGif_FT(self, file_name='ft_vs_time.gif')`

Create a GIF animation of Fourier components over time.

- `file_name` (`str`, optional): Name of the output GIF file.

##### `MakeGif_tot(self, file_name='cahn_hilliard_tot.gif')`

Create a GIF animation of concentration and Fourier analysis over time.

- `file_name` (`str`, optional): Name of the output GIF file.

This class allows you to simulate and analyze solutions of the Cahn-Hilliard equation, providing methods for data analysis and visualization.