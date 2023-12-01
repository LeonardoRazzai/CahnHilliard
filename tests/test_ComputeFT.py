import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
from CahnHilliard import Sol_CahnHilliard


def test_shape():
    """
    GIVEN a Sol_CahnHilliard instance and a solution (sol)
    WHEN we set the step value and compute Fourier Transforms
    THEN ft_sol and ft_t are computed and not None
    AND ft_sol and ft_t have the same length as expected (len(t) // step).
    """
    # Define test parameters
    L = 10.0
    N = 100
    D = 0.01
    a = 0.001
    t = np.linspace(0.0, 1.0, 10)
    step = 2
    
    # Create a Sol_CahnHilliard instance
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    
    # Create an initial concentration field
    c0 = np.random.rand(N, N)
    
    # Compute the solution
    sol_solver.compute_sol(c0, t)
    
    # Set the step value and compute Fourier Transforms
    sol_solver.set_step(step)
    sol_solver.ComputeFT()
    
    # Check if ft_sol and ft_t are computed and not None
    assert sol_solver.ft_sol is not None
    assert sol_solver.ft_t is not None
    
    # Check if ft_sol and ft_t have the same length as expected (len(t) // step)
    assert len(sol_solver.ft_sol) == len(t) // step
    assert len(sol_solver.ft_t) == len(t) // step


def test_zero_diffusivity():
    """
    GIVEN a solution (sol) with diffusivity D=0.0
    WHEN computing FTs with ComputeFT
    THEN initial FT is equal to final FT.
    """
    # Define test parameters
    L = 10.0
    N = 100
    D = 0.00
    a = 0.001
    t = np.linspace(0.0, 1.0, 10)
    
    # Create a Sol_CahnHilliard instance
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    
    # Create an initial concentration field (random values)
    c0 = np.random.rand(N, N)
    
    # Compute the solution
    sol_solver.compute_sol(c0, t)
    
    # Compute Fourier Transforms
    sol_solver.ComputeFT()
    
    # Check if the initial spectrum matches final spectrum when D=0.0
    np.testing.assert_array_almost_equal(sol_solver.ft_sol[0], sol_solver.ft_sol[-1], decimal=8)


def test_constant_concentration():
    """
    GIVEN a solution (sol) with uniform initial concentration (c0)
    WHEN computing the FTs with ComputeFT
    THEN ft_sol is all zeros.
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
    
    # Compute Fourier Transforms
    sol_solver.ComputeFT()
    
    # Check if the FTs are all zeros
    assert np.all(sol_solver.ft_sol == 0)