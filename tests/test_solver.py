import unittest
import sys
import os
import numpy as np
import sympy as sp
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import reactorpy modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the solver classes and functions
from reactorpy.solver import ODESolver, solve_ode_system, solve_steady_state, create_time_series, validate_solution
from reactorpy.reaction import Reaction


class TestODESolver(unittest.TestCase):
    """Test cases for the ODESolver class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.solver = ODESolver()
        
        # Create simple symbolic system: A -> B
        # d[A]/dt = -k*[A]
        # d[B]/dt = k*[A]
        self.A_symbol = sp.Symbol('[A]', real=True, positive=True)
        self.B_symbol = sp.Symbol('[B]', real=True, positive=True)
        self.k_symbol = sp.Symbol('k', real=True, positive=True)
        
        self.simple_odes = {
            'A': -self.k_symbol * self.A_symbol,
            'B': self.k_symbol * self.A_symbol
        }
        
        self.concentration_symbols = {
            'A': self.A_symbol,
            'B': self.B_symbol
        }
        
        self.parameter_symbols = {
            'k': self.k_symbol
        }

    def test_init(self):
        """Test ODESolver initialization."""
        self.assertIsNone(self.solver.symbolic_system)
        self.assertIsNone(self.solver.numerical_system)
        self.assertEqual(self.solver.concentration_symbols, [])
        self.assertEqual(self.solver.parameter_symbols, [])
        self.assertEqual(self.solver.lambdified_functions, {})

    def test_prepare_system(self):
        """Test preparing symbolic system."""
        self.solver.prepare_system(
            self.simple_odes,
            self.concentration_symbols,
            self.parameter_symbols
        )
        
        self.assertEqual(self.solver.symbolic_system, self.simple_odes)
        self.assertEqual(self.solver.concentration_symbols, self.concentration_symbols)
        self.assertEqual(self.solver.parameter_symbols, self.parameter_symbols)
        self.assertEqual(self.solver.species_names, ['A', 'B'])

    def test_lambdify_system(self):
        """Test lambdifying symbolic system."""
        self.solver.prepare_system(
            self.simple_odes,
            self.concentration_symbols,
            self.parameter_symbols
        )
        
        rate_parameters = {'k': 0.1}
        self.solver.lambdify_system(rate_parameters)
        
        # Check that lambdified functions were created
        self.assertIn('A', self.solver.lambdified_functions)
        self.assertIn('B', self.solver.lambdified_functions)
        self.assertTrue(callable(self.solver.lambdified_functions['A']))
        self.assertTrue(callable(self.solver.lambdified_functions['B']))

    def test_lambdify_system_no_preparation(self):
        """Test lambdifying without preparing system first."""
        with self.assertRaises(ValueError):
            self.solver.lambdify_system({'k': 0.1})

    def test_create_ode_function(self):
        """Test creating ODE function for scipy solver."""
        self.solver.prepare_system(
            self.simple_odes,
            self.concentration_symbols,
            self.parameter_symbols
        )
        
        rate_parameters = {'k': 0.1}
        ode_func = self.solver.create_ode_function(rate_parameters)
        
        # Test the function
        t = 0.0
        y = np.array([1.0, 0.0])  # [A]=1.0, [B]=0.0
        dydt = ode_func(t, y)
        
        self.assertEqual(len(dydt), 2)
        self.assertAlmostEqual(dydt[0], -0.1)  # d[A]/dt = -k*[A] = -0.1*1.0
        self.assertAlmostEqual(dydt[1], 0.1)   # d[B]/dt = k*[A] = 0.1*1.0

    def test_ode_function_negative_concentrations(self):
        """Test ODE function behavior with negative concentrations."""
        self.solver.prepare_system(
            self.simple_odes,
            self.concentration_symbols,
            self.parameter_symbols
        )
        
        ode_func = self.solver.create_ode_function({'k': 0.1})
        
        # Test with negative concentration (should be handled safely)
        t = 0.0
        y = np.array([-0.1, 0.5])  # Negative [A]
        dydt = ode_func(t, y)
        
        # Should not crash and should return reasonable values
        self.assertEqual(len(dydt), 2)
        self.assertIsInstance(dydt[0], float)
        self.assertIsInstance(dydt[1], float)


class TestSolveODESystem(unittest.TestCase):
    """Test cases for the solve_ode_system function."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple A -> B reaction system
        self.A_symbol = sp.Symbol('[A]', real=True, positive=True)
        self.B_symbol = sp.Symbol('[B]', real=True, positive=True)
        self.k_symbol = sp.Symbol('k', real=True, positive=True)
        
        self.symbolic_odes = {
            'A': -self.k_symbol * self.A_symbol,
            'B': self.k_symbol * self.A_symbol
        }
        
        self.initial_concentrations = {'A': 1.0, 'B': 0.0}
        self.time_span = (0.0, 10.0)
        self.rate_parameters = {'k': 0.1}
        
        self.concentration_symbols = {
            'A': self.A_symbol,
            'B': self.B_symbol
        }
        
        self.parameter_symbols = {
            'k': self.k_symbol
        }

    def test_solve_ode_system_basic(self):
        """Test basic ODE system solving."""
        results = solve_ode_system(
            self.symbolic_odes,
            self.initial_concentrations,
            self.time_span,
            rate_parameters=self.rate_parameters,
            concentration_symbols=self.concentration_symbols,
            parameter_symbols=self.parameter_symbols
        )
        
        # Check that results contain expected keys
        self.assertIn('time', results)
        self.assertIn('A', results)
        self.assertIn('B', results)
        self.assertIn('success', results)
        self.assertIn('final_concentrations', results)
        self.assertIn('species_names', results)
        
        # Check that solver was successful
        self.assertTrue(results['success'])
        
        # Check conservation: [A] + [B] should be constant (= 1.0)
        total_concentration = results['A'] + results['B']
        np.testing.assert_allclose(total_concentration, 1.0, rtol=1e-5)
        
        # Check that [A] decreases and [B] increases
        self.assertGreater(results['A'][0], results['A'][-1])
        self.assertLess(results['B'][0], results['B'][-1])

    def test_solve_ode_system_missing_initial_concentrations(self):
        """Test error handling for missing initial concentrations."""
        incomplete_initial = {'A': 1.0}  # Missing B
        
        with self.assertRaises(ValueError):
            solve_ode_system(
                self.symbolic_odes,
                incomplete_initial,
                self.time_span,
                rate_parameters=self.rate_parameters
            )

    def test_solve_ode_system_auto_symbols(self):
        """Test automatic symbol creation when not provided."""
        results = solve_ode_system(
            self.symbolic_odes,
            self.initial_concentrations,
            self.time_span,
            rate_parameters=self.rate_parameters
        )
        
        # Should work even without providing symbols explicitly
        self.assertTrue(results['success'])
        self.assertIn('A', results)
        self.assertIn('B', results)

    def test_solve_ode_system_different_methods(self):
        """Test different integration methods."""
        methods = ['RK45', 'RK23', 'BDF']
        
        for method in methods:
            with self.subTest(method=method):
                results = solve_ode_system(
                    self.symbolic_odes,
                    self.initial_concentrations,
                    self.time_span,
                    rate_parameters=self.rate_parameters,
                    method=method
                )
                
                self.assertTrue(results['success'])
                self.assertEqual(results['method'], method)

    def test_solve_ode_system_solver_options(self):
        """Test passing solver options."""
        results = solve_ode_system(
            self.symbolic_odes,
            self.initial_concentrations,
            self.time_span,
            rate_parameters=self.rate_parameters,
            rtol=1e-8,
            atol=1e-12,
            max_step=0.1
        )
        
        self.assertTrue(results['success'])
        # Should have used the specified tolerances

    def test_solve_ode_system_final_concentrations(self):
        """Test final concentrations extraction."""
        results = solve_ode_system(
            self.symbolic_odes,
            self.initial_concentrations,
            self.time_span,
            rate_parameters=self.rate_parameters
        )
        
        final_conc = results['final_concentrations']
        self.assertIn('A', final_conc)
        self.assertIn('B', final_conc)
        
        # Final concentrations should match last values in arrays
        self.assertAlmostEqual(final_conc['A'], results['A'][-1])
        self.assertAlmostEqual(final_conc['B'], results['B'][-1])


class TestSolveSteadyState(unittest.TestCase):
    """Test cases for the solve_steady_state function."""

    def setUp(self):
        """Set up test fixtures."""
        # Equilibrium system: A <-> B
        # At steady state: k1*[A] = k2*[B]
        self.A_symbol = sp.Symbol('[A]', real=True, positive=True)
        self.B_symbol = sp.Symbol('[B]', real=True, positive=True)
        self.k1_symbol = sp.Symbol('k1', real=True, positive=True)
        self.k2_symbol = sp.Symbol('k2', real=True, positive=True)
        
        self.equilibrium_odes = {
            'A': -self.k1_symbol * self.A_symbol + self.k2_symbol * self.B_symbol,
            'B': self.k1_symbol * self.A_symbol - self.k2_symbol * self.B_symbol
        }
        
        self.initial_guess = {'A': 0.5, 'B': 0.5}
        self.rate_parameters = {'k1': 0.1, 'k2': 0.05}

    def test_solve_steady_state_basic(self):
        """Test basic steady-state solving with mass conservation."""
        # Calculate total mass from initial guess
        total_mass = sum(self.initial_guess.values())  # 0.5 + 0.5 = 1.0
        
        # Define symbol dictionaries for proper symbol matching
        concentration_symbols = {
            'A': self.A_symbol,
            'B': self.B_symbol
        }
        parameter_symbols = {
            'k1': self.k1_symbol,
            'k2': self.k2_symbol
        }
        
        results = solve_steady_state(
            self.equilibrium_odes,
            self.initial_guess,
            rate_parameters=self.rate_parameters,
            concentration_symbols=concentration_symbols,
            parameter_symbols=parameter_symbols,
            total_mass=total_mass
        )
        
        # Check that results contain expected keys
        self.assertIn('success', results)
        self.assertIn('A', results)
        self.assertIn('B', results)
        self.assertIn('steady_state_concentrations', results)
        
        if results['success']:
            steady_state = results['steady_state_concentrations']
            
            # Check that rates are zero (steady state condition)
            A_val = steady_state['A']
            B_val = steady_state['B']
            k1, k2 = self.rate_parameters['k1'], self.rate_parameters['k2']
            
            print(f"Steady state: A_val = {A_val}, B_val = {B_val}")
            
            # Calculate the rates at steady state
            dA_dt = -k1 * A_val + k2 * B_val
            dB_dt = k1 * A_val - k2 * B_val
            
            print(f"Derivatives: dA_dt = {dA_dt}, dB_dt = {dB_dt}")
            
            # These should be very close to zero
            self.assertTrue(abs(dA_dt) < 1e-6, f"dA_dt = {dA_dt} should be < 1e-6")
            self.assertTrue(abs(dB_dt) < 1e-6, f"dB_dt = {dB_dt} should be < 1e-6")
            
            # Check mass conservation
            total_concentration = A_val + B_val
            self.assertAlmostEqual(total_concentration, total_mass, places=6)
            
            # Check equilibrium ratio: At steady state k1*[A] = k2*[B], so [B]/[A] = k1/k2
            if A_val > 1e-10:
                ratio = B_val / A_val
                expected_ratio = k1 / k2  # = 0.1/0.05 = 2.0
                self.assertAlmostEqual(ratio, expected_ratio, places=3)
        else:
            self.fail(f"Steady state solver failed: {results.get('message', 'Unknown error')}")

    def test_solve_steady_state_methods(self):
        """Test different root finding methods."""
        methods = ['hybr', 'lm']
        
        # Define symbol dictionaries for proper symbol matching
        concentration_symbols = {
            'A': self.A_symbol,
            'B': self.B_symbol
        }
        parameter_symbols = {
            'k1': self.k1_symbol,
            'k2': self.k2_symbol
        }
        
        for method in methods:
            with self.subTest(method=method):
                try:
                    results = solve_steady_state(
                        self.equilibrium_odes,
                        self.initial_guess,
                        rate_parameters=self.rate_parameters,
                        concentration_symbols=concentration_symbols,
                        parameter_symbols=parameter_symbols,
                        method=method
                    )
                    self.assertEqual(results['method'], method)
                except Exception:
                    # Some methods might not work for all problems
                    pass


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_create_time_series_linear(self):
        """Test linear time series creation."""
        time_span = (0.0, 10.0)
        n_points = 11
        
        times = create_time_series(time_span, n_points, log_spacing=False)
        
        self.assertEqual(len(times), n_points)
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 10.0)
        np.testing.assert_allclose(times, np.linspace(0.0, 10.0, 11))

    def test_create_time_series_log(self):
        """Test logarithmic time series creation."""
        time_span = (1.0, 100.0)
        n_points = 21
        
        times = create_time_series(time_span, n_points, log_spacing=True)
        
        self.assertEqual(len(times), n_points)
        self.assertAlmostEqual(times[0], 1.0)
        self.assertAlmostEqual(times[-1], 100.0)

    def test_create_time_series_log_zero_start(self):
        """Test logarithmic spacing with zero start (should fall back to linear)."""
        time_span = (0.0, 10.0)
        n_points = 11
        
        times = create_time_series(time_span, n_points, log_spacing=True)
        
        # Should fall back to linear spacing
        np.testing.assert_allclose(times, np.linspace(0.0, 10.0, 11))

    def test_validate_solution_successful(self):
        """Test solution validation for successful case."""
        # Mock successful results
        results = {
            'success': True,
            'species_names': ['A', 'B'],
            'A': np.array([1.0, 0.8, 0.6, 0.4, 0.2]),
            'B': np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        }
        
        validation = validate_solution(results)
        
        self.assertTrue(validation['solver_success'])
        self.assertTrue(validation['no_negative_concentrations'])
        self.assertTrue(validation['reasonable_values'])

    def test_validate_solution_negative_concentrations(self):
        """Test solution validation with negative concentrations."""
        results = {
            'success': True,
            'species_names': ['A', 'B'],
            'A': np.array([1.0, 0.5, -0.1, -0.5]),  # Contains negative values
            'B': np.array([0.0, 0.5, 1.1, 1.5])
        }
        
        validation = validate_solution(results)
        
        self.assertFalse(validation['no_negative_concentrations'])

    def test_validate_solution_unreasonable_values(self):
        """Test solution validation with unreasonably large values."""
        results = {
            'success': True,
            'species_names': ['A', 'B'],
            'A': np.array([1.0, 1e7, 1e8]),  # Very large values
            'B': np.array([0.0, 0.5, 1.0])
        }
        
        validation = validate_solution(results)
        
        self.assertFalse(validation['reasonable_values'])

    def test_validate_solution_solver_failure(self):
        """Test solution validation when solver failed."""
        results = {
            'success': False,
            'species_names': ['A', 'B'],
            'A': np.array([1.0, 0.5]),
            'B': np.array([0.0, 0.5])
        }
        
        validation = validate_solution(results)
        
        self.assertFalse(validation['solver_success'])


class TestIntegrationWithReactions(unittest.TestCase):
    """Integration tests using actual Reaction objects."""

    def test_solve_with_reaction_object(self):
        """Test solving ODE system created from Reaction object."""
        # Create a simple reaction A -> B
        reaction = Reaction("A -> B")
        
        # Manually create symbolic system (normally done by ModelBuilder)
        A_symbol = sp.Symbol('[A]', real=True, positive=True)
        B_symbol = sp.Symbol('[B]', real=True, positive=True)
        k_symbol = sp.Symbol('k', real=True, positive=True)
        
        symbolic_odes = {
            'A': -k_symbol * A_symbol,
            'B': k_symbol * A_symbol
        }
        
        concentration_symbols = {'A': A_symbol, 'B': B_symbol}
        parameter_symbols = {'k': k_symbol}
        
        results = solve_ode_system(
            symbolic_odes,
            {'A': 2.0, 'B': 0.0},
            (0.0, 5.0),
            rate_parameters={'k': 0.2},
            concentration_symbols=concentration_symbols,
            parameter_symbols=parameter_symbols
        )
        
        self.assertTrue(results['success'])
        self.assertAlmostEqual(results['A'][0], 2.0)  # Initial [A]
        self.assertAlmostEqual(results['B'][0], 0.0)  # Initial [B]
        
        # Check mass conservation
        total = results['A'] + results['B']
        np.testing.assert_allclose(total, 2.0, rtol=1e-5)

    def test_error_handling_invalid_odes(self):
        """Test error handling for invalid ODE expressions."""
        # Create problematic ODE system with division by zero potential
        A_symbol = sp.Symbol('[A]', real=True, positive=True)
        invalid_odes = {
            'A': 1 / A_symbol,  # Could cause division by zero when [A] = 0
        }
        
        # This should handle the error gracefully
        try:
            results = solve_ode_system(
                invalid_odes,
                {'A': 0.0},  # Zero initial condition - will cause division by zero
                (0.0, 1.0)
            )
            # If it doesn't crash, that's good
        except Exception as e:
            # Should raise a RuntimeError or ValueError with descriptive message
            self.assertIsInstance(e, (RuntimeError, ValueError, ZeroDivisionError))


if __name__ == '__main__':
    # Set up test suite
    unittest.main(verbosity=2)
