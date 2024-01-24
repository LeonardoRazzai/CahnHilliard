import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d
from tqdm import tqdm

SMALL_SIZE = 11
MEDIUM_SIZE = 14
BIGGER_SIZE = 20

base_font = {'family': 'serif',
        'color':  'black',
        'size': SMALL_SIZE,
        }

title_font = {'family': 'serif',
        'color':  'black',
        'size': MEDIUM_SIZE,
        }

title_figure = {'family': 'serif',
        'color':  'darkred',
        'size': BIGGER_SIZE,
        'weight' : 'bold'
        }

def init_plotting():
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

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
        A 2D NumPy array representing the time derivative concentration field 'u' at
        the current time step 't'.

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
    >>> dudt = Cahn_Hilliard(initial_concentration, t, dx, D, a)
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
    >>> def harmonic_oscillator(u, t, omega):
    ...     # Define the harmonic oscillator equation.
    ...     # u[0] represents the position, u[1] represents the velocity.
    ...     return np.array([u[1], -omega**2 * u[0]])
    
    >>> initial_conditions = np.array([1.0, 0.0])  # Initial position and velocity
    >>> time_steps = np.linspace(0.0, 1.0, 100)
    >>> omega = 1.0
    >>> solution = integrate(harmonic_oscillator, initial_conditions, time_steps, omega)
    """
    
    Nt = len(t)
    dt = np.max(t) / Nt

    u = np.zeros((Nt, *u0.shape))
    
    u[0] = u0
    barstep = 1
    with tqdm(total=Nt) as pbar:
        for i in range(1, Nt):
            u[i] = u[i-1] + func(u[i-1], t[i-1], *args) * dt
            pbar.update(barstep)
        pbar.close()
    
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
    step : int
        Step size for data analysis and visualization. Default value is 10.
    ft_sol : np.ndarray
        2D NumPy array representing the Fourier transformed concentration profiles.
    ft_t : np.ndarray
        1D NumPy array representing time steps for Fourier analysis.
    histo : list
        List of histograms representing concentration distributions over time.

    Methods:
    --------
    compute_sol(c0, t)
        Simulate the time evolution of the concentration field with initial condition c0.
    ComputeFT()
        Compute the average Fourier transform of concentration profiles along x and y.
    Compute_histo()
        Compute concentration histograms over time.
    set_step(step)
        Set the step size for data analysis and visualization.
    MakeGif_sol(file_name = 'cahn_hilliard.gif'):
        Create a GIF animation of concentration field evolution.
    MakeGif_FT(file_name = 'ft_vs_time.gif'):
        Create a GIF animation of Fourier components over time.
    MakeGif_tot(file_name = 'cahn_hilliard_tot.gif'):
        Create a GIF animation of concentration field, Fourier components and histogram over time.
    """
    
    def __init__(self, L: float, N: int, D: float, a: float) -> None:

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
        self.histo = None

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
        if self.sol is None:
            raise TypeError("Error: you didn't compute the solution. Run the method compute_sol.")
        else:
            self.ft_t = self.t[0:-1:self.step]
            sect_x = np.mean(self.sol, axis=1)
            sect_y = np.mean(self.sol, axis=2)
            sect = (sect_x[0:-1:self.step] + sect_y[0:-1:self.step])/2

            ft_sol = np.zeros((len(sect), len(self.sol[0])))

            for i in range(0, len(sect)):
                hat = np.fft.fft(sect[i] - np.mean(sect[i]))
                psd = np.sqrt(np.real(np.conj(hat) * hat))
                ft_sol[i] = psd
            self.ft_sol = ft_sol / (ft_sol[0] + 0.000001)
    
    def Compute_histo(self):
        """
        Compute histograms of concentration at times specified by self.step.
        """
        if self.sol is None:
            raise TypeError("Error: you didn't compute the solution. Run the method compute_sol.")
        else:
            N = len(self.sol[0])
            Nt = len(self.sol)
            histo = []

            for i in range(0, Nt, self.step):
                conc_array = np.reshape(self.sol[i], (N**2))
                histo.append(np.histogram(conc_array, N))
            self.histo = histo
        
    def MakeGif_sol(self, file_name = 'cahn_hilliard.gif'):
        """
        Create a GIF animation of concentration field evolution.

        Parameters:
        ----------
        file_name : str, optional
            Name of the output GIF file.
        """
        if self.sol is None:
            raise TypeError("Error: you didn't compute the solution. Run the method compute_sol.")
        else:
            N = len(self.x)
            Nt = len(self.sol)
            sol_to_plot = self.sol[0:Nt:self.step]
            t_to_plot = self.t[0:Nt:self.step]
            
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))

            ax[0].plot([0, 400], [200, 200], color = 'darkorchid')
            img = ax[0].imshow(sol_to_plot[0], cmap='inferno')
            ax[0].set_title(f'Concentration in x-y plane', fontdict=title_font)

            ln2, = ax[1].plot(self.x, sol_to_plot[0][N//2], color='darkorchid')
            ax[1].plot([np.min(self.x), np.max(self.x)], [1/np.sqrt(3), 1/np.sqrt(3)], '--', color='black', label='Spinodal')
            ax[1].plot([np.min(self.x), np.max(self.x)], [-1/np.sqrt(3), -1/np.sqrt(3)], '--', color='black')
            ax[1].set_title('Concentration at y=0', fontdict=title_font)
            ax[1].set_ylabel('Conc', fontdict=base_font)
            ax[1].set_xlabel('x (cm)', fontdict=base_font)
            ax[1].set_ylim(-1, 1)
            ax[1].set_xlim(self.x[0], self.x[-1])
            ax[1].legend(loc='upper left')

            fig.suptitle('Time evolution Fourier components, t = 0.0 s', fontdict=title_figure)

            def update(i):
                img.set_data(sol_to_plot[i])
                ln2.set_data(self.x, sol_to_plot[i][N//2])
                fig.suptitle(f'Time evolution Fourier components, t = {t_to_plot[i]:.1f} s', fontdict=title_figure)


            tmax = self.t[-1]
            ani = FuncAnimation(fig, update, frames = len(sol_to_plot)-1)
            ani.save(file_name, writer='pillow', fps= 10)
            
    def MakeGif_FT(self, file_name = 'ft_vs_time.gif'):
        """
        Create a GIF animation of Fourier components over time.

        Parameters:
        ----------
        file_name : str, optional
            Name of the output GIF file.
        """
        if self.ft_sol is None:
            raise TypeError("Error: you didn't compute the fourier transform. Run the method Compute_FT.")
        else:
            N = len(self.x)
            dx = self.x[1] - self.x[0]
            
            k = np.fft.fftfreq(N, dx) * 2*np.pi
            
            n_steps = len(self.ft_sol)
            index_max = np.argmax(self.ft_sol[int(n_steps/10), :N//2])
            ft_max = self.ft_sol[:, index_max]

            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].set_ylim(0, np.max(self.ft_sol[:])+0.5)
            ax[1].set_ylim(0, np.max(ft_max)+0.5)

            ln1, = ax[0].plot(k[:N//2], self.ft_sol[0][:N//2])
            ln2, = ax[0].plot([k[index_max]], [ft_max[0]], 'o', ms=5, color='black')

            ax[0].axvline(1/(np.sqrt(self.a)), ls='--',color='red', label='Critical line\n'+r'1/$\sqrt{a}$')
            ax[0].set_xlabel('k', fontdict=base_font)
            ax[0].set_ylabel('A(k, t)/A(k, 0)', fontdict=base_font)
            ax[0].set_title('Fourier spectrum', fontdict=title_font)
            ax[0].legend()
            
            ax[1].plot(self.ft_t, ft_max)
            ln3, = ax[1].plot([self.t[0]], [ft_max[0]], 'o', ms=5, color='black')
            ax[1].set_xlabel('t (s)', fontdict=base_font)
            ax[1].set_ylabel(f'Component k = {k[index_max]:.1f}', fontdict=base_font)
            ax[1].set_title('Fastest growing Fourier component', fontdict=title_font)

            fig.suptitle('Time evolution Fourier components, t = 0.0 s', fontdict=title_figure)

            def update(i):
                ln1.set_data(k[:N//2], self.ft_sol[i][:N//2])
                ln2.set_data([k[index_max]], [ft_max[i]])
                ln3.set_data([self.ft_t[i]], [ft_max[i]])
                fig.suptitle(f'Time evolution Fourier components, t = {self.ft_t[i]:.1f} s', fontdict=title_figure)

            tmax = self.t[-1]
            ani = FuncAnimation(fig, update, frames = len(self.ft_sol)-1)
            ani.save(file_name, writer='pillow', fps= 10)
        
    def MakeGif_tot(self, file_name = 'cahn_hilliard_tot.gif'):
        """
        Create a GIF animation of concentration and Fourier analysis over time.

        Parameters:
        ----------
        file_name : str, optional
            Name of the output GIF file.
        """
        if self.ft_sol is None:
            raise TypeError("Error: you didn't compute the fourier transform. Run the method Compute_FT.")
        elif self.histo is None:
            raise TypeError("Error: you didn't compute the histogram. Run the method compute_histo.")
        else:
            N = len(self.x)
            Nt = len(self.sol)
            sol_to_plot = self.sol[0:Nt:self.step]
            sol_to_plot.shape

            fig = plt.figure(figsize=(15, 7))
            grid = plt.GridSpec(4, 4, hspace=0.8, wspace=0.3)

            # image
            img_ax = fig.add_subplot(grid[:, :2])
            img_ax.set_xticks([])
            img_ax.set_yticks([])
            img = img_ax.imshow(sol_to_plot[0], cmap='inferno')
            img_ax.set_title(f'Concentration in x-y plane', fontdict=title_font)

            # conc_x
            concx_ax = fig.add_subplot(grid[:2, 2:])
            histo = self.histo[0]
            bins = histo[1]
            counts = histo[0]
            ln1, = concx_ax.plot(bins[:-1], counts / np.max(counts), color='darkorchid')
            concx_ax.axvline(1/np.sqrt(3), ls='--', color='black', label='Spinodal')
            concx_ax.axvline(-1/np.sqrt(3), ls='--', color='black')
            concx_ax.set_title('Concentration distribution', fontdict=title_font)
            concx_ax.set_ylabel('counts', fontdict=base_font)
            concx_ax.set_xlabel('Conc', fontdict=base_font)
            concx_ax.set_xlim(-1.2, 1.2)
            concx_ax.set_ylim(0.05, 1.1)
            concx_ax.legend(loc='upper left')

            # ft_conc_x
            ft_concx_ax = fig.add_subplot(grid[2:, 2:])
            ft_concx_ax.set_ylim(0, np.max(self.ft_sol[:])+1)

            dx = self.x[1] - self.x[0]
            k = np.fft.fftfreq(N, dx) * 2*np.pi

            ft_ln1, = ft_concx_ax.plot(k[:N//2], self.ft_sol[0][:N//2])

            ft_concx_ax.axvline(1/(np.sqrt(self.a)), ls='--',color='red', label='Critical line\n'+r'1/$\sqrt{a}$')
            ft_concx_ax.set_xlabel(r'k (cm$^{-1}$)', fontdict=base_font)
            ft_concx_ax.set_ylabel('A(k, t)/A(k, 0)', fontdict=base_font)
            ft_concx_ax.set_title('FT of concentration profile', fontdict=title_font)
            ft_concx_ax.legend()

            fig.suptitle('Time evolution of concentration, t = 0.0 s', fontdict=title_figure)

            def update(i):
                img.set_data(sol_to_plot[i])
                histo = self.histo[i]
                bins = histo[1]
                counts = histo[0]
                ln1.set_data(bins[:-1], counts / np.max(counts))
                ft_ln1.set_data(k[:N//2], self.ft_sol[i][:N//2])
                fig.suptitle(f'Time evolution of concentration, t = {self.ft_t[i]:.1f} s', fontdict=title_figure)

            tmax = self.t[-1]
            ani = FuncAnimation(fig, update, frames = len(sol_to_plot)-1)
            ani.save(file_name, writer='pillow', fps= 10)