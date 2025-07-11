import unittest
import sys
import os
import sympy as sp

# Add the parent directory to the path to import reactorpy modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the model builder classes and functions
from reactorpy.model_builder import ModelBuilder, generate_symbolic_ODE
from reactorpy.reaction import Reaction
from reactorpy.species import Species


class TestModelBuilder(unittest.TestCase):
    """Test cases for the ModelBuilder class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.builder = ModelBuilder()
        
        # Create test reactions
        self.simple_reaction = Reaction("A + B -> C")
        self.reaction_with_rate = Reaction(
            "A + B -> C",
            rate_expression="k * [A] * [B]",
            rate_parameters={'k': 0.1}
        )
        self.reversible_reaction = Reaction("A <-> B")
        
        # Create test species
        self.species_A = Species("A", concentration=1.0)
        self.species_B = Species("B", concentration=0.5)
        self.species_C = Species("C", concentration=0.0)

    def test_init(self):
        """Test ModelBuilder initialization."""
        self.assertEqual(self.builder.concentration_symbols, {})
        self.assertEqual(self.builder.parameter_symbols, {})
        self.assertIsInstance(self.builder.time_symbol, sp.Symbol)

    def test_create_concentration_symbol(self):
        """Test creation of concentration symbols."""
        symbol_A = self.builder.create_concentration_symbol("A")
        
        self.assertIsInstance(symbol_A, sp.Symbol)
        self.assertEqual(str(symbol_A), "[A]")
        self.assertTrue(symbol_A.is_real)
        self.assertTrue(symbol_A.is_positive)
        
        # Test that same symbol is returned for same species
        symbol_A2 = self.builder.create_concentration_symbol("A")
        self.assertEqual(symbol_A, symbol_A2)
        
        # Test different species get different symbols
        symbol_B = self.builder.create_concentration_symbol("B")
        self.assertNotEqual(symbol_A, symbol_B)

    def test_create_parameter_symbol(self):
        """Test creation of parameter symbols."""
        symbol_k = self.builder.create_parameter_symbol("k")
        
        self.assertIsInstance(symbol_k, sp.Symbol)
        self.assertEqual(str(symbol_k), "k")
        self.assertTrue(symbol_k.is_real)
        self.assertTrue(symbol_k.is_positive)
        
        # Test that same symbol is returned for same parameter
        symbol_k2 = self.builder.create_parameter_symbol("k")
        self.assertEqual(symbol_k, symbol_k2)
        
        # Test different parameters get different symbols
        symbol_k1 = self.builder.create_parameter_symbol("k1")
        self.assertNotEqual(symbol_k, symbol_k1)

    def test_parse_rate_expression_empty(self):
        """Test parsing empty rate expression."""
        result = self.builder.parse_rate_expression("")
        self.assertEqual(result, sp.sympify(0))
        
        result = self.builder.parse_rate_expression(None)
        self.assertEqual(result, sp.sympify(0))

    def test_parse_rate_expression_simple(self):
        """Test parsing simple rate expressions."""
        # Test k * [A]
        expr = self.builder.parse_rate_expression("k * [A]", {"k": 0.1})
        
        # Should contain [A] symbol and k symbol
        symbols = expr.free_symbols
        symbol_names = [str(s) for s in symbols]
        self.assertIn("[A]", symbol_names)
        self.assertIn("k", symbol_names)

    def test_parse_rate_expression_complex(self):
        """Test parsing complex rate expressions."""
        # Test k1 * [A] * [B] - k2 * [C]
        expr = self.builder.parse_rate_expression(
            "k1 * [A] * [B] - k2 * [C]",
            {"k1": 0.1, "k2": 0.05}
        )
        
        symbols = expr.free_symbols
        symbol_names = [str(s) for s in symbols]
        self.assertIn("[A]", symbol_names)
        self.assertIn("[B]", symbol_names)
        self.assertIn("[C]", symbol_names)
        self.assertIn("k1", symbol_names)
        self.assertIn("k2", symbol_names)

    def test_parse_rate_expression_powers(self):
        """Test parsing rate expressions with powers."""
        # Test k * [A]^2 * [B]
        expr = self.builder.parse_rate_expression("k * [A]^2 * [B]", {"k": 0.1})
        
        # Should handle ^ to ** conversion
        symbols = expr.free_symbols
        symbol_names = [str(s) for s in symbols]
        self.assertIn("[A]", symbol_names)
        self.assertIn("[B]", symbol_names)
        self.assertIn("k", symbol_names)

    def test_parse_rate_expression_invalid(self):
        """Test parsing invalid rate expressions."""
        with self.assertRaises(ValueError):
            self.builder.parse_rate_expression("invalid expression !@#", {"k": 0.1})

    def test_generate_single_reaction_odes_no_rate(self):
        """Test generating ODEs for reaction without rate expression."""
        odes = self.builder.generate_single_reaction_odes(self.simple_reaction)
        
        # Should return contributions for A, B, C
        self.assertIn('A', odes)
        self.assertIn('B', odes)
        self.assertIn('C', odes)
        
        # A and B should have negative contributions (reactants)
        # C should have positive contribution (product)
        self.assertEqual(odes['A'], sp.sympify(0))  # No rate expression means zero rate
        self.assertEqual(odes['B'], sp.sympify(0))
        self.assertEqual(odes['C'], sp.sympify(0))

    def test_generate_single_reaction_odes_with_rate(self):
        """Test generating ODEs for reaction with rate expression."""
        odes = self.builder.generate_single_reaction_odes(self.reaction_with_rate)
        
        self.assertIn('A', odes)
        self.assertIn('B', odes)
        self.assertIn('C', odes)
        
        # Check that rate expression is incorporated
        # For A + B -> C with rate k*[A]*[B]:
        # d[A]/dt = -k*[A]*[B]
        # d[B]/dt = -k*[A]*[B]
        # d[C]/dt = +k*[A]*[B]
        
        # Get the rate expression
        rate_expr = odes['C']  # Positive rate for product
        
        # Should contain [A], [B], and k symbols
        symbols = rate_expr.free_symbols
        symbol_names = [str(s) for s in symbols]
        self.assertIn("[A]", symbol_names)
        self.assertIn("[B]", symbol_names)
        self.assertIn("k", symbol_names)

    def test_generate_ode_system_single_reaction(self):
        """Test generating ODE system from single reaction."""
        reactions = [self.reaction_with_rate]
        
        ode_system = self.builder.generate_ode_system(reactions)
        
        # Should have entries for A, B, C
        self.assertIn('A', ode_system)
        self.assertIn('B', ode_system)
        self.assertIn('C', ode_system)
        
        # Check stoichiometry is correct
        # For A + B -> C: d[A]/dt should be negative, d[C]/dt positive
        rate_A = ode_system['A']
        rate_C = ode_system['C']
        
        # A should have negative rate, C should have positive rate
        # They should be equal in magnitude (stoichiometry = 1)
        self.assertEqual(rate_A, -rate_C)

    def test_generate_ode_system_multiple_reactions(self):
        """Test generating ODE system from multiple reactions."""
        # A + B -> C and C -> A + B
        reaction1 = Reaction("A + B -> C", rate_expression="k1 * [A] * [B]", rate_parameters={'k1': 0.1})
        reaction2 = Reaction("C -> A + B", rate_expression="k2 * [C]", rate_parameters={'k2': 0.05})
        
        reactions = [reaction1, reaction2]
        ode_system = self.builder.generate_ode_system(reactions)
        
        # Should have contributions from both reactions
        # d[A]/dt = -k1*[A]*[B] + k2*[C]
        # d[B]/dt = -k1*[A]*[B] + k2*[C]
        # d[C]/dt = +k1*[A]*[B] - k2*[C]
        
        rate_A = ode_system['A']
        rate_C = ode_system['C']
        
        # Both should contain terms from both reactions
        symbols_A = rate_A.free_symbols
        symbols_C = rate_C.free_symbols
        
        symbol_names_A = [str(s) for s in symbols_A]
        symbol_names_C = [str(s) for s in symbols_C]
        
        # Should contain all species and both rate constants
        for symbols_list in [symbol_names_A, symbol_names_C]:
            self.assertIn("[A]", symbols_list)
            self.assertIn("[B]", symbols_list)
            self.assertIn("[C]", symbols_list)
            self.assertIn("k1", symbols_list)
            self.assertIn("k2", symbols_list)

    def test_generate_ode_system_specified_species(self):
        """Test generating ODE system with specified species list."""
        reactions = [self.reaction_with_rate]
        species_list = ['A', 'B']  # Only include A and B, not C
        
        ode_system = self.builder.generate_ode_system(reactions, species_list)
        
        # Should only have A and B
        self.assertIn('A', ode_system)
        self.assertIn('B', ode_system)
        self.assertNotIn('C', ode_system)

    def test_generate_ode_system_with_species_objects(self):
        """Test generating ODE system with Species objects."""
        reactions = [self.reaction_with_rate]
        species_objects = [self.species_A, self.species_B, self.species_C]
        
        ode_system = self.builder.generate_ode_system_with_species_objects(
            reactions, species_objects
        )
        
        # Should have entries for all species
        self.assertIn('A', ode_system)
        self.assertIn('B', ode_system)
        self.assertIn('C', ode_system)

    def test_get_system_matrix_form(self):
        """Test converting ODE system to matrix form."""
        reactions = [self.reaction_with_rate]
        ode_system = self.builder.generate_ode_system(reactions)
        
        conc_vector, ode_vector, param_symbols = self.builder.get_system_matrix_form(ode_system)
        
        # Check concentration vector
        self.assertEqual(len(conc_vector), 3)  # A, B, C
        conc_names = [str(s) for s in conc_vector]
        self.assertIn("[A]", conc_names)
        self.assertIn("[B]", conc_names)
        self.assertIn("[C]", conc_names)
        
        # Check ODE vector
        self.assertEqual(len(ode_vector), 3)
        
        # Check parameter symbols
        param_names = [str(s) for s in param_symbols]
        self.assertIn("k", param_names)

    def test_substitute_numerical_values(self):
        """Test substituting numerical values."""
        reactions = [self.reaction_with_rate]
        ode_system = self.builder.generate_ode_system(reactions)
        
        numerical_params = {'k': 0.2}
        species_objects = [self.species_A, self.species_B, self.species_C]
        
        numerical_system = self.builder.substitute_numerical_values(
            ode_system, species_objects, numerical_params
        )
        
        # Should have substituted k = 0.2
        # Check that k symbol is no longer in the expressions
        for species, ode in numerical_system.items():
            symbols = ode.free_symbols
            symbol_names = [str(s) for s in symbols]
            # k should be substituted out, but concentration symbols should remain
            self.assertNotIn("k", symbol_names)

    def test_reversible_reaction_handling(self):
        """Test handling of reversible reactions."""
        # Create reversible reaction A <-> B
        reversible = Reaction("A <-> B", rate_expression="k1 * [A] - k2 * [B]", 
                             rate_parameters={'k1': 0.1, 'k2': 0.05})
        
        ode_system = self.builder.generate_ode_system([reversible])
        
        # For A <-> B:
        # d[A]/dt = -(k1*[A] - k2*[B]) = -k1*[A] + k2*[B]
        # d[B]/dt = +(k1*[A] - k2*[B]) = +k1*[A] - k2*[B]
        
        rate_A = ode_system['A']
        rate_B = ode_system['B']
        
        # A and B rates should be negatives of each other
        self.assertEqual(rate_A, -rate_B)

    def test_complex_stoichiometry(self):
        """Test reactions with complex stoichiometry."""
        # 2A + 3B -> C + 4D
        complex_reaction = Reaction("2A + 3B -> C + 4D", 
                                   rate_expression="k * [A]^2 * [B]^3",
                                   rate_parameters={'k': 0.01})
        
        ode_system = self.builder.generate_ode_system([complex_reaction])
        
        # Check stoichiometric coefficients
        # d[A]/dt = -2 * rate
        # d[B]/dt = -3 * rate
        # d[C]/dt = +1 * rate
        # d[D]/dt = +4 * rate
        
        rate_A = ode_system['A']
        rate_B = ode_system['B']
        rate_C = ode_system['C']
        rate_D = ode_system['D']
        
        # Check ratios based on stoichiometry
        # rate_A / rate_C should be -2/1 = -2
        # rate_B / rate_C should be -3/1 = -3
        # rate_D / rate_C should be 4/1 = 4
        
        # Verify by checking that rate_A = -2 * rate_C
        self.assertEqual(rate_A, -2 * rate_C)
        self.assertEqual(rate_B, -3 * rate_C)
        self.assertEqual(rate_D, 4 * rate_C)

    def test_species_with_numbers_and_underscores(self):
        """Test species names with numbers and underscores."""
        reaction = Reaction("H2O + CO2 -> H2CO3", 
                           rate_expression="k * [H2O] * [CO2]",
                           rate_parameters={'k': 0.1})
        
        ode_system = self.builder.generate_ode_system([reaction])
        
        # Should handle species names with numbers
        self.assertIn('H2O', ode_system)
        self.assertIn('CO2', ode_system)
        self.assertIn('H2CO3', ode_system)
        
        # Check that symbols were created correctly
        symbols = ode_system['H2O'].free_symbols
        symbol_names = [str(s) for s in symbols]
        self.assertIn("[H2O]", symbol_names)
        self.assertIn("[CO2]", symbol_names)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Reaction with only reactants (A -> )
        reaction_decomp = Reaction("A -> ", rate_expression="k * [A]", rate_parameters={'k': 0.1})
        ode_system = self.builder.generate_ode_system([reaction_decomp])
        
        # Should have negative rate for A
        self.assertIn('A', ode_system)
        rate_A = ode_system['A']
        
        # Should contain [A] and k symbols
        symbols = rate_A.free_symbols
        symbol_names = [str(s) for s in symbols]
        self.assertIn("[A]", symbol_names)
        self.assertIn("k", symbol_names)

    def test_parameter_name_boundaries(self):
        """Test parameter name boundary detection in expressions."""
        # Test that 'k' in 'k1' doesn't get replaced when looking for 'k'
        expr = self.builder.parse_rate_expression("k1 * [A] + k * [B]", {"k": 0.1, "k1": 0.2})
        
        symbols = expr.free_symbols
        symbol_names = [str(s) for s in symbols]
        
        # Should have both k and k1 as separate symbols
        self.assertIn("k", symbol_names)
        self.assertIn("k1", symbol_names)


class TestLegacyFunction(unittest.TestCase):
    """Test cases for the legacy generate_symbolic_ODE function."""

    def test_generate_symbolic_ODE_basic(self):
        """Test legacy function with basic reaction."""
        reaction = Reaction("A -> B", rate_expression="k * [A]", rate_parameters={'k': 0.1})
        species = ['A', 'B']
        
        ode_dict = generate_symbolic_ODE(reaction, species)
        
        # Should return dictionary with A and B entries
        self.assertIn('A', ode_dict)
        self.assertIn('B', ode_dict)
        
        # Check that it uses ModelBuilder internally
        self.assertIsInstance(ode_dict['A'], sp.Expr)
        self.assertIsInstance(ode_dict['B'], sp.Expr)

    def test_generate_symbolic_ODE_with_species_objects(self):
        """Test legacy function with Species objects."""
        reaction = Reaction("A -> B", rate_expression="k * [A]", rate_parameters={'k': 0.1})
        species_A = Species("A", concentration=1.0)
        species_B = Species("B", concentration=0.0)
        species = [species_A, species_B]
        
        ode_dict = generate_symbolic_ODE(reaction, species)
        
        # Should extract names from Species objects
        self.assertIn('A', ode_dict)
        self.assertIn('B', ode_dict)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_invalid_species_name_in_rate(self):
        """Test handling of species in rate expression not in reaction."""
        # Rate expression references [X] but reaction doesn't involve X
        reaction = Reaction("A -> B", rate_expression="k * [X]", rate_parameters={'k': 0.1})
        builder = ModelBuilder()
        
        # Should still work - creates symbol for [X] even if not in reaction
        odes = builder.generate_single_reaction_odes(reaction)
        self.assertIn('A', odes)
        self.assertIn('B', odes)

    def test_empty_reaction_list(self):
        """Test handling of empty reaction list."""
        builder = ModelBuilder()
        
        ode_system = builder.generate_ode_system([], species_list=['A', 'B'])
        
        # Should return zero ODEs for all species
        self.assertEqual(ode_system['A'], sp.sympify(0))
        self.assertEqual(ode_system['B'], sp.sympify(0))

    def test_no_rate_parameters(self):
        """Test handling when no rate parameters are provided."""
        builder = ModelBuilder()
        
        expr = builder.parse_rate_expression("k * [A]")  # No rate_parameters dict
        
        # Should still create k symbol
        symbols = expr.free_symbols
        symbol_names = [str(s) for s in symbols]
        self.assertIn("k", symbol_names)
        self.assertIn("[A]", symbol_names)


if __name__ == '__main__':
    # Set up test suite
    unittest.main(verbosity=2)
