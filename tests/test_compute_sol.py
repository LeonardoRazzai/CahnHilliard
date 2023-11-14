import sys
sys.path.append('./')
sys.path.append('../')
from CahnHilliard import *

# set seed for random values
np.random.seed(42)

def test_result_shape():
    """
    GIVEN a Sol_CahnHilliard instance with specific parameters
    WHEN computing the solution with a random initial concentration field and time array
    THEN the shape of the computed solution matches the expected shape (len(t), N, N).
    """

    # Define test parameters
    L = 10.0
    N = 100
    D = 0.01
    a = 0.001
    t = np.linspace(0.0, 1.0, 10)
    
    # Create a Sol_CahnHilliard instance
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    
    # Create an initial concentration field (random values)
    c0 = np.random.rand(N, N)
    
    # Compute the solution
    sol_solver.compute_sol(c0, t)
    
    # Check if the solution shape matches the expected shape
    assert sol_solver.sol.shape == (len(t), N, N)


def test_initial_concentration():
    """
    GIVEN a Sol_CahnHilliard instance with specific parameters
    WHEN computing the solution with a random initial concentration field and time array
    THEN the initial concentration field matches the concentration at the first time step.
    """

    # Define test parameters
    L = 10.0
    N = 100
    D = 0.01
    a = 0.001
    t = np.linspace(0.0, 1.0, 10)
    
    # Create a Sol_CahnHilliard instance
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    
    # Create an initial concentration field (random values)
    c0 = np.random.rand(N, N)
    
    # Compute the solution
    sol_solver.compute_sol(c0, t)
    
    # Check if the initial concentration field matches the concentration at the first time step
    np.testing.assert_equal(sol_solver.sol[0], c0)


def test_constant_concentration():
    """
    GIVEN a Sol_CahnHilliard instance with specific parameters
    WHEN computing the solution with an initial constant concentration field and time array
    THEN the concentration at the final time step matches the initial constant concentration field.
    """

    # Define test parameters
    L = 10.0
    N = 100
    D = 0.01
    a = 0.001
    t = np.linspace(0.0, 1.0, 10)
    
    # Create a Sol_CahnHilliard instance
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    
    # Create an initial constant concentration field
    c0 = np.ones((N, N))
    
    # Compute the solution
    sol_solver.compute_sol(c0, t)
    
    # Check if the concentration at the final time step matches the initial constant concentration field
    np.testing.assert_equal(sol_solver.sol[-1], c0)

    
def test_zero_diffusivity():
    """
    GIVEN a Sol_CahnHilliard instance with zero diffusion coefficient
    WHEN computing the solution with a random initial concentration field and time array
    THEN the concentration at the final time step matches the initial concentration field.
    """

    # Define test parameters
    L = 10.0
    N = 100
    D = 0.0  # Set zero diffusivity
    a = 0.001
    t = np.linspace(0.0, 1.0, 10)
    
    # Create a Sol_CahnHilliard instance
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    
    # Create an initial concentration field (random values)
    c0 = np.random.randn(N, N)
    
    # Compute the solution
    sol_solver.compute_sol(c0, t)
    
    # Check if the concentration at the final time step matches the initial concentration field
    np.testing.assert_equal(sol_solver.sol[-1], c0)


def test_compute_sol_bounded():
    """
    GIVEN a Sol_CahnHilliard instance with specified parameters,
    WHEN computing the solution with a random initial concentration field,
    THEN the solution concentration field is bounded between -1 and 1 at any time.
    """
    # Parameters
    L = 40 # side length of domain
    N = 400 # number of points per spatial dimension
    dx = L / N # spatial step along x and y
    
    mean = 0.0 # average concentration
    c0 = np.zeros((N, N)) + mean

    D = dx**2 * 3 # diffusivity cm**2 / s
    gamma = 500
    beta = N / dx / gamma # in this way lmax is 2*pi * L/gamma
    a = dx**4 * beta**2 # fatstest growing wavevector is 1/sqrt(a)

    amp = 0.01
    c0 = c0 + amp * np.random.randn(N, N) # initial concentration

    tmax = 4.5
    Nt = 500
    t = np.linspace(0, tmax, Nt)

    # GIVEN
    sol_cahn_hilliard = Sol_CahnHilliard(L, N, D, a)

    # WHEN
    sol_cahn_hilliard.compute_sol(c0, t)

    # THEN
    np.testing.assert_(np.all(sol_cahn_hilliard.sol >= -1) and np.all(sol_cahn_hilliard.sol <= 1),
                       "Resulting values are not bounded between -1 and 1.")
