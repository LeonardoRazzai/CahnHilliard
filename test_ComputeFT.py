import unittest
import numpy as np
from CahnHilliard import Sol_CahnHilliard

class TestSolCahnHilliard(unittest.TestCase):

    def test_compute_sol(self):
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
        self.assertEqual(sol_solver.sol.shape, (len(t), N, N))
        
        # Check if the initial concentration field matches the first time step
        np.testing.assert_array_almost_equal(sol_solver.sol[0], c0, decimal=5)

if __name__ == '__main__':
    unittest.main()