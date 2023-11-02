import sys
sys.path.append('./')
sys.path.append('../')
from CahnHilliard import *

def test_compute_sol():
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
    
    # Check if the initial concentration field matches the first time step
    np.testing.assert_equal(sol_solver.sol[0], c0)


def test_constant_concentration():
    # Define test parameters
    L = 10.0
    N = 10
    D = 0.01
    a = 0.001
    t = np.linspace(0.0, 1.0, 10)
    
    # Create a Sol_CahnHilliard instance
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    
    # Create an initial constant concentration field
    c0 = np.ones((N, N))
    
    # Compute the solution
    sol_solver.compute_sol(c0, t)
    
    # Check if the initial concentration field matches the final concentration
    np.testing.assert_equal(sol_solver.sol[-1], c0)
    
def test_zero_diffusivity():
    # Define test parameters
    L = 10.0
    N = 10
    D = 0.00 # Set zero diffusivity
    a = 0.001
    t = np.linspace(0.0, 1.0, 10)
    
    # Create a Sol_CahnHilliard instance
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    
    # Create an initial concentration field (random values)
    c0 = np.random.randn(N, N)
    
    # Compute the solution
    sol_solver.compute_sol(c0, t)
    
    # Check if the initial concentration field matches the final concentration
    np.testing.assert_equal(sol_solver.sol[-1], c0)