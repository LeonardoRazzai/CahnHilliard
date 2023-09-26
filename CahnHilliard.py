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
  
  
def integrate(func, u0, t, *args):
    """
    Numerically integrate a given differential equation over time.

    This function performs numerical integration of a differential equation
    defined by the function 'func' using the Euler method. It computes the
    solution at discrete time steps specified by the array 't'.

    Parameters:
    ----------
    func : callable
        The function representing the differential equation to be integrated.
        It should accept the following arguments:
        - u : np.ndarray
            The current state of the system.
        - t : float
            The current time.
        - *args : tuple
            Additional arguments to be passed to 'func'.
        The function should return the rate of change of 'u' at the given time 't'.
    u0 : np.ndarray
        A 2D NumPy array representing the initial state of the system.
    t : np.ndarray
        A 1D NumPy array containing the discrete time steps at which to compute
        the solution.
    *args : tuple, optional
        Additional arguments to be passed to 'func'.

    Returns:
    -------
    np.ndarray
        A 3D NumPy array representing the solution of the differential equation
        over time. The first dimension corresponds to time steps, and the shape
        of each step matches the shape of 'u0'.

    Notes:
    ------
    This function uses the Euler method for numerical integration, which may
    introduce numerical errors, especially for stiff differential equations.
    It is recommended to use more advanced integration methods for accuracy
    when necessary.

    Example:
    --------
    >>> import numpy as np
    >>> def Cahn_Hilliard(u, t, dx, D, a):
    ...     # Define the Cahn-Hilliard equation here.
    ...     return (D / dx**2) * laplacian2(u**3 - u - (a / dx**2) * laplacian2(u))
    >>> initial_concentration = np.random.rand(100, 100)
    >>> time_steps = np.linspace(0.0, 1.0, 10)
    >>> dx = 0.1
    >>> D = 0.1
    >>> a = 1.0
    >>> solution = integrate(Cahn_Hilliard, initial_concentration, time_steps, dx, D, a)
    """
    Nt = len(t)
    dt = np.max(t) / Nt

    u = np.zeros((Nt, u0.shape[0], u0.shape[1]))
    u[0] = u0
    for i in range(1, Nt):
        u[i] = u[i-1] + func(u[i-1], t, *args) * dt
    
    return u
  
  
class Sol_CahnHilliard:
    
    """
    Class for simulating and analyzing the Cahn-Hilliard equation solutions.

    This class provides methods for simulating and analyzing the solutions of the
    Cahn-Hilliard equation, a partial differential equation used to model phase
    separation in materials science and physics.

    Parameters:
    ----------
    L : float
        Length of the spatial domain.
    N : int
        Number of spatial grid points.
    D : float
        Diffusion coefficient controlling the rate of phase separation.
    a : float
        Parameter affecting the interfacial energy between phases.

    Attributes:
    -----------
    D : float
        Diffusion coefficient.
    a : float
        Interfacial energy parameter.
    x : np.ndarray
        1D NumPy array representing the spatial grid.
    sol : np.ndarray
        3D NumPy array representing the concentration field over time.
    t : np.ndarray
        1D NumPy array representing time steps.

    Methods:
    --------
    compute_sol(c0, t)
        Simulate the evolution of the concentration field.
    """
    
    def __init__(self, L, N, D, a) -> None:

        """
        Initialize a Sol_CahnHilliard instance with specified parameters.

        Parameters:
        ----------
        L : float
            Length of the spatial domain.
        N : int
            Number of spatial grid points.
        D : float
            Diffusion coefficient controlling the rate of phase separation.
        a : float
            Parameter affecting the interfacial energy between phases
        """
        
        self.D = D
        self.a = a
        self.x = np.linspace(-L/2, L/2, N)
        self.step = 10
        
        self.sol = None
        self.t = None
        
        self.ft_sol = None
        self.ft_t = None

    def compute_sol(self, c0, t):
        """
        Simulate the evolution of the concentration field.

        Parameters:
        ----------
        c0 : np.ndarray
            Initial concentration field.
        t : np.ndarray
            1D NumPy array representing time steps.
        """
        self.t = t
        dx = self.x[1] - self.x[0]
        self.sol = integrate(Cahn_Hilliard, c0, t, dx, self.D, self.a)
    
    def set_step(self, step: int):
        """
        Set the step size for data analysis and visualization.

        Parameters:
        ----------
        step : int
            New step size.
        """
        self.step = step

    def ComputeFT(self):
        """
        Compute the average Fourier transform of concentration profiles along x and y
        at times specified by self.step.
        """
        self.ft_t = self.t[0:-1:self.step]
        sect_x = np.mean(self.sol, axis=1)
        sect_y = np.mean(self.sol, axis=2)
        sect = (sect_x[0:-1:self.step] + sect_y[0:-1:self.step])/2

        ft_sol = np.zeros((len(sect), len(self.sol[0])))

        for i in range(0, len(sect)):
            hat = np.fft.fft(sect[i] - np.mean(sect[i]))
            psd = np.sqrt(np.real(np.conj(hat) * hat))
            ft_sol[i] = psd

        self.ft_sol = ft_sol
    
    def Compute_histo(self):
        """
        Compute histograms of concentration at times specified by self.step.
        """
        N = len(self.sol[0])
        Nt = len(self.sol)
        histo = []
        for i in range(0, Nt, self.step):
            conc_array = np.reshape(self.sol[i], (N**2))
            histo.append(np.histogram(conc_array, N))
        
        self.histo = histo