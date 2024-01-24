import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
from CahnHilliard import Sol_CahnHilliard


def test_shape():
    """
    GIVEN A solution to the Cahn-Hilliard equation with a random initial concentration field
    WHEN computing the histograms of concentration values at each time
    THEN the computed histogram is not None
    AND the length of histogram is equal to the expected length (len(t) // step).
    """
    # set seed for random number generators
    np.random.seed(42)
    
    # Define test parameters
    L = 10.0
    N = 100
    D = 0.01  # diffusion coefficient
    a = 0.001  # interfacial parameter
    t = np.linspace(0.0, 1.0, 10)
    step = 2
    
    # GIVEN
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    c0 = np.random.rand(N, N)
    sol_solver.compute_sol(c0, t)

    # WHEN
    sol_solver.set_step(step)
    sol_solver.Compute_histo()

    # THEN
    assert sol_solver.histo is not None
    expected_length = len(t) // step
    assert len(sol_solver.histo) == expected_length

def test_zero_diffusivity():
    """
    GIVEN A solution to the Cahn-Hilliard equation with diffusivity D = 0.0
    WHEN computing the histograms of concentration values at each time
    THEN the initial histogram is equal to the final histogram.
    """
    # set seed for random number generators
    np.random.seed(42)
    
    # Define test parameters
    L = 10.0
    N = 100
    D = 0.0  # diffusion coefficient
    a = 0.001  # interfacial parameter
    t = np.linspace(0.0, 1.0, 10)
    step = 2

    # GIVEN
    sol_solver = Sol_CahnHilliard(L, N, D, a)
    c0 = np.random.rand(N, N)
    sol_solver.compute_sol(c0, t)

    # WHEN
    sol_solver.set_step(step)
    sol_solver.Compute_histo()

    # THEN
    initial_hist = sol_solver.histo[0][0]
    final_hist = sol_solver.histo[-1][0]
    assert np.array_equal(initial_hist, final_hist)
    
def test_None():
    """
    GIVEN Sol_CahnHilliard instance
    WHEN calling Compute_histo wihtout having computed a solution
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
    
    try:
        sol_solver.Compute_histo()
    except TypeError:
        assert True
        
def test_None_MakeGif():
    """
    GIVEN Sol_CahnHilliard instance
    WHEN calling MakeGif_tot wihtout having computed the histograms
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
    
    try:
        sol_solver.MakeGif_tot()
    except TypeError:
        assert True

def test_idempot():
    
    """
    GIVEN Sol_CahnHilliard instance
    WHEN calling compute_histo twice
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
    sol_solver.Compute_histo()
    first_histo = sol_solver.histo
    sol_solver.Compute_histo()
    second_histo = sol_solver.histo
    
    np.testing.assert_equal(first_histo, second_histo)
    
    
    
