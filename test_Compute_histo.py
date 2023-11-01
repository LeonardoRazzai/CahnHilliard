import unittest
import numpy as np
from CahnHilliard import Sol_CahnHilliard

class TestSolCahnHilliard(unittest.TestCase):

    def test_compute_histo(self):
        # Define test parameters
        L = 10.0
        N = 100
        D = 0.01  # Specific value
        a = 0.001  # Specific value
        t = np.linspace(0.0, 1.0, 10)
        step = 2  # Choose a reasonable step value
        
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
        
        # Add more specific checks based on your expectations for the histograms

if __name__ == '__main__':
    unittest.main()
