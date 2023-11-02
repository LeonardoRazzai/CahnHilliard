import unittest
import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
from CahnHilliard import Sol_CahnHilliard

class TestComputeFT(unittest.TestCase):

    def test_shape(self):
        # Define test parameters
        L = 10.0
        N = 100
        D = 0.01
        a = 0.001
        t = np.linspace(0.0, 1.0, 10)
        step = 2
        
        # Create a Sol_CahnHilliard instance
        sol_solver = Sol_CahnHilliard(L, N, D, a)
        
        # Create an initial concentration field (random values)
        c0 = np.random.rand(N, N)
        
        # Compute the solution
        sol_solver.compute_sol(c0, t)
        
        # Set the step value and compute histograms
        sol_solver.set_step(step)
        sol_solver.ComputeFT()
        
        # Check if ft_sol and ft_t are computed
        self.assertIsNotNone(sol_solver.ft_sol)
        self.assertIsNotNone(sol_solver.ft_t)
        # Check if ft_sol and ft_t have same length and equal to len(t) // step
        self.assertEqual(len(sol_solver.ft_sol), len(t) // step)
        self.assertEqual(len(sol_solver.ft_t), len(t) // step)
    
    def test_zero_diffusivity(self):
        # Define test parameters
        L = 10.0
        N = 100
        D = 0.00
        a = 0.001
        t = np.linspace(0.0, 1.0, 10)
        step = 2
        
        # Create a Sol_CahnHilliard instance
        sol_solver = Sol_CahnHilliard(L, N, D, a)
        
        # Create an initial concentration field (random values)
        c0 = np.random.rand(N, N)
        
        # Compute the solution
        sol_solver.compute_sol(c0, t)
        
        # Set the step value and compute histograms
        sol_solver.set_step(step)
        sol_solver.ComputeFT()
        
        # Check if the initial spectrum matches final spectrum when D=0.0
        np.testing.assert_array_almost_equal(sol_solver.ft_sol[0], sol_solver.ft_sol[-1], decimal=8)

if __name__ == '__main__':
    unittest.main()