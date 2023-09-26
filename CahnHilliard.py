import numpy as np
from scipy.signal import convolve2d

LAP2 = np.array(
    [[0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]]
)

def laplacian2(state: np.ndarray) -> np.ndarray:
    """
    Apply a 2D Laplacian filter to a given 2D array.

    Parameters:
    ----------
    state : np.ndarray
        A 2D NumPy array representing the input data or image on which the
        Laplacian filter will be applied.

    Returns:
    -------
    np.ndarray
        A 2D NumPy array representing the filtered result after applying the
        Laplacian filter. The output has the same shape as the input.

    Notes:
    ------
    The Laplacian filter is applied using a 3x3 kernel defined as follows:
    [[ 0,  1,  0],
     [ 1, -4,  1],
     [ 0,  1,  0]]

    The 'same' mode is used for convolution, ensuring that the output has the
    same dimensions as the input. The 'symm' boundary condition is used for
    handling boundary pixels.

    Example:
    --------
    >>> input_image = np.array([[10, 20, 30],
    ...                         [40, 50, 60],
    ...                         [70, 80, 90]])
    >>> filtered_image = laplacian2(input_image)
    >>> print(filtered_image)
    array([[ -4,  -8,  -4],
           [-12,  40, -12],
           [ -4, -16,  -4]])
    """
    return convolve2d(state, LAP2, mode='same', boundary='symm')
  
  
def Cahn_Hilliard(u, t: float, dx: float, D: float, a: float):
    """
    Calculate the Cahn-Hilliard equation for phase separation.

    The Cahn-Hilliard equation is a partial differential equation used to model
    phase separation in materials science and physics. It describes the evolution
    of a concentration field 'u' over time 't' in a 2D system.

    Parameters:
    ----------
    u : np.ndarray
        A 2D NumPy array representing the concentration field at the current time.
    t : float
        The current time.
    dx : float
        Spatial step size (grid spacing) in the x and y directions.
    D : float
        Diffusion coefficient controlling the rate of phase separation.
    a : float
        Parameter affecting the interfacial energy between phases.

    Returns:
    -------
    np.ndarray
        A 2D NumPy array representing the updated concentration field 'u' at
        the next time step 't+dt', where 'dt' is determined by the numerical
        scheme used.

    Notes:
    ------
    The Cahn-Hilliard equation is defined as follows:

    du/dt = (D / dx^2) * laplacian2(u^3 - u - (a / dx^2) * laplacian2(u))

    Where Laplacian^2(u) represents the Laplacian of 'u' calculated using the
    'laplacian2' function.

    The function calculates the right-hand side of the equation for the given
    concentration field 'u', time 't', spatial step size 'dx', diffusion
    coefficient 'D', and interfacial energy parameter 'a'.

    Example:
    --------
    >>> initial_concentration = np.random.rand(100, 100)
    >>> t = 0.0
    >>> dx = 0.1
    >>> D = 0.1
    >>> a = 1.0
    >>> updated_concentration = Cahn_Hilliard(initial_concentration, t, dx, D, a)
    """
    return (D / dx**2) * laplacian2(np.power(u, 3) - u - (a / dx**2) * laplacian2(u))