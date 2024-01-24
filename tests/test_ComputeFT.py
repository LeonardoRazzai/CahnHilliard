import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
from CahnHilliard import Sol_CahnHilliard
import pytest


def test_shape():
    """
    GIVEN a Sol_CahnHilliard instance and a solution (sol)
    WHEN we set the step value and compute Fourier Transforms
    THEN ft_sol and ft_t are computed and not None
    AND ft_sol and ft_t have the same length as expected (len(t) // step).
    """
    # set seed for random number generators
    np.random.seed(42)
    
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
    # set seed for random number generators
    np.random.seed(42)
    
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

def test_None_ComputeFT():
    """
    GIVEN Sol_CahnHilliard instance
    WHEN calling ComputeFT wihtout having computed a solution
    THEN a TypeError is raised.
    """
    # Define test parameters
    L = 10.0
    N = 100
    D = 0.00
    a = 0.001
    t = np.linspace(0.0, 1.0, 10)
    
    # Create a Sol_CahnHilliard instance
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    
    with pytest.raises(TypeError):
        sol_solver.ComputeFT()
        
def test_None_MakeGif():
    """
    GIVEN Sol_CahnHilliard instance
    WHEN calling MakeGif_FT wihtout having computed the FT
    THEN a TypeError is raised.
    """
    # Define test parameters
    L = 10.0
    N = 100
    D = 0.00
    a = 0.001
    t = np.linspace(0.0, 1.0, 10)
    
    # Create a Sol_CahnHilliard instance
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    
    with pytest.raises(TypeError):
        sol_solver.MakeGif_FT()
    
def test_idempot():
    
    """
    GIVEN Sol_CahnHilliard instance
    WHEN calling computeFT twice
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
    sol_solver.compute_sol(c0, t)

    # WHEN
    sol_solver.ComputeFT()
    first_ft = sol_solver.ft_sol
    sol_solver.ComputeFT()
    second_ft = sol_solver.ft_sol
    
    np.testing.assert_equal(first_ft, second_ft)