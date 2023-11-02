import unittest
import numpy as np
import sys
sys.path.append('./')
sys.path.append('../')
from CahnHilliard import Sol_CahnHilliard

class Test_Compute_histo(unittest.TestCase):

    def test_shape(self):
        # Define test parameters
        L = 10.0
        N = 100
        D = 0.01  # diffusion coefficient
        a = 0.001  # interfacial parameter
        t = np.linspace(0.0, 1.0, 10)
        step = 2
        
        # Create a Sol_CahnHilliard instance with specific D and a values
        sol_solver = Sol_CahnHilliard(L, N, D, a)
        
        # Create an initial concentration field (random values)
        c0 = np.random.rand(N, N)
        
        # Compute the solution
        sol_solver.compute_sol(c0, t)
        
        # Set the step value and compute histograms
        sol_solver.set_step(step)
        sol_solver.Compute_histo()
        
        # Check if histograms are computed and have the expected length
        self.assertIsNotNone(sol_solver.histo)
        self.assertEqual(len(sol_solver.histo), len(t) // step)
        
    def test_zero_diffusivity(self):
        # Define test parameters
        L = 10.0
        N = 100
        D = 0.0  # diffusion coefficient
        a = 0.001  # interfacial parameter
        t = np.linspace(0.0, 1.0, 10)
        step = 2
        
        # Create a Sol_CahnHilliard instance with specific D and a values
        sol_solver = Sol_CahnHilliard(L, N, D, a)
        
        # Create an initial concentration field (random values)
        c0 = np.random.rand(N, N)
        
        # Compute the solution
        sol_solver.compute_sol(c0, t)
        
        # Set the step value and compute histograms
        sol_solver.set_step(step)
        sol_solver.Compute_histo()
        
        # Check if zero diffusivity gives same initial and final histogram
        initial_hist = sol_solver.histo[0][0]
        final_hist = sol_solver.histo[-1][0]
        np.testing.assert_almost_equal(initial_hist, final_hist, decimal=8) 

if __name__ == '__main__':
    unittest.main()
