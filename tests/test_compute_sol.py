import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
from CahnHilliard import Sol_CahnHilliard

def test_result_shape():
    """
    GIVEN a Sol_CahnHilliard instance with NxN spatial grid
    WHEN computing the solution with a random initial concentration field and time array
    THEN the shape of the computed solution matches the expected shape (len(t), N, N).
    """
    # set seed for random number generators
    np.random.seed(42)

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
    WHEN computing the solution with a random initial concentration field
    THEN the initial concentration field matches the concentration at the first time step.
    """
    # set seed for random number generators
    np.random.seed(42)

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
    WHEN computing the solution with an initial constant concentration field
    THEN the concentration at the final time step matches the initial constant concentration field.
    """
    # set seed for random number generators
    np.random.seed(42)
    
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
    GIVEN a Sol_CahnHilliard instance with diffusion coefficient D = 0.0
    WHEN computing the solution with a random initial concentration field
    THEN the concentration at the final time step matches the initial concentration field.
    """
    # set seed for random number generators
    np.random.seed(42)
    
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
    # set seed for random number generators
    np.random.seed(42)
    
    # Parameters
    L = 40 
    N = 400
    D = 1e-2
    a = 1e-3

    mean = 0.0
    c0 = np.zeros((N, N)) +  mean
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
    
def test_compute_sol_average_conc():
    """
    GIVEN A spatial NxN grid,
    WHEN computing the solution with a random initial concentration field with normal distribution with std `amp`,
    THEN the average concentration is constant in time within 4*amp / N.
    """
    # set seed for random number generators
    np.random.seed(42)
    
    # Parameters
    L = 40 
    N = 400
    D = 1e-2
    a = 1e-3

    mean = 0.0
    c0 = np.zeros((N, N)) +  mean
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
    np.testing.assert_almost_equal(np.average(sol_cahn_hilliard.sol.reshape(Nt, N**2), axis=1), np.ones(Nt) * mean, 4*amp/N,
                       "Average concentratioin is not constant.")

def test_idempot():
    
    """
    GIVEN Sol_CahnHilliard instance
    WHEN calling compute_sol twice with same initial condition
    THEN the result is the same.
    """
    # set seed for random number generators
    np.random.seed(42)
    
    # Define test parameters
    L = 10.0
    N = 100
    D = 0.0  # diffusion coefficient
    a = 0.001  # interfacial parameter
    t = np.linspace(0.0, 1.0, 10)

    # GIVEN
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    c0 = np.random.rand(N, N)

    # WHEN
    sol_solver.compute_sol(c0, t)
    first_sol = sol_solver.sol
    sol_solver.compute_sol(c0, t)
    second_sol = sol_solver.sol
    
    np.testing.assert_equal(first_sol, second_sol)