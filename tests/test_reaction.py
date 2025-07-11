import unittest
import sys
import os

# Add the parent directory to the path to import reactorpy modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the Reaction class directly to avoid dependency issues
from reactorpy.reaction import Reaction


class TestReaction(unittest.TestCase):
    """Test cases for the Reaction class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Simple irreversible reaction
        self.simple_reaction = Reaction("A + B -> C")
        
        # Reversible reaction with stoichiometry
        self.reversible_reaction = Reaction("2A + B <-> C + 3D")
        
        # Reaction with rate parameters
        self.parametrized_reaction = Reaction(
            "A + B -> C",
            rate_expression="k * [A] * [B]",
            rate_parameters={'k': 0.1, 'Ea': 50000}
        )

    def test_init_from_reaction_string(self):
        """Test initialization from reaction string."""
        reaction = Reaction("A + B -> C")
        
        self.assertEqual(reaction.reactants, {'A': 1.0, 'B': 1.0})
        self.assertEqual(reaction.products, {'C': 1.0})
        self.assertFalse(reaction.is_reversible)
        self.assertEqual(reaction.reaction_string, "A + B -> C")

    def test_init_from_explicit_species(self):
        """Test initialization from explicit reactants and products."""
        reactants = {'A': 2.0, 'B': 1.0}
        products = {'C': 1.0, 'D': 3.0}
        
        reaction = Reaction(
            reactants=reactants,
            products=products,
            is_reversible=True
        )
        
        self.assertEqual(reaction.reactants, reactants)
        self.assertEqual(reaction.products, products)
        self.assertTrue(reaction.is_reversible)
        self.assertEqual(reaction.reaction_string, "2.0A + B <-> C + 3.0D")

    def test_parse_simple_reaction(self):
        """Test parsing of simple reaction strings."""
        reaction = Reaction("A -> B")
        
        self.assertEqual(reaction.reactants, {'A': 1.0})
        self.assertEqual(reaction.products, {'B': 1.0})
        self.assertFalse(reaction.is_reversible)

    def test_parse_reaction_with_stoichiometry(self):
        """Test parsing reactions with stoichiometric coefficients."""
        reaction = Reaction("2A + 3B -> C + 4D")
        
        self.assertEqual(reaction.reactants, {'A': 2.0, 'B': 3.0})
        self.assertEqual(reaction.products, {'C': 1.0, 'D': 4.0})

    def test_parse_reversible_reaction(self):
        """Test parsing reversible reactions."""
        # Test with <->
        reaction1 = Reaction("A + B <-> C")
        self.assertTrue(reaction1.is_reversible)
        self.assertEqual(reaction1.reactants, {'A': 1.0, 'B': 1.0})
        self.assertEqual(reaction1.products, {'C': 1.0})
        
        # Test with equilibrium arrow ⇌
        reaction2 = Reaction("A + B ⇌ C")
        self.assertTrue(reaction2.is_reversible)

    def test_parse_reaction_with_decimal_coefficients(self):
        """Test parsing reactions with decimal stoichiometric coefficients."""
        reaction = Reaction("0.5A + 1.5B -> 2.0C")
        
        self.assertEqual(reaction.reactants, {'A': 0.5, 'B': 1.5})
        self.assertEqual(reaction.products, {'C': 2.0})

    def test_parse_invalid_reaction_strings(self):
        """Test that invalid reaction strings raise appropriate errors."""
        # No arrow
        with self.assertRaises(ValueError):
            Reaction("A + B")
            
        # Multiple arrows
        with self.assertRaises(ValueError):
            Reaction("A + B -> -> C")
            
        # Invalid coefficient format
        with self.assertRaises(ValueError):
            Reaction("2.5.5A -> B")

    def test_get_all_species(self):
        """Test getting all species involved in the reaction."""
        reaction = Reaction("A + B -> C + D")
        species = reaction.get_all_species()
        
        self.assertEqual(species, {'A', 'B', 'C', 'D'})

    def test_get_stoichiometric_coefficient(self):
        """Test getting stoichiometric coefficients for species."""
        reaction = Reaction("2A + B -> 3C")
        
        # Reactants should have negative coefficients
        self.assertEqual(reaction.get_stoichiometric_coefficient('A'), -2.0)
        self.assertEqual(reaction.get_stoichiometric_coefficient('B'), -1.0)
        
        # Products should have positive coefficients
        self.assertEqual(reaction.get_stoichiometric_coefficient('C'), 3.0)
        
        # Non-participating species should return 0
        self.assertEqual(reaction.get_stoichiometric_coefficient('D'), 0.0)

    def test_rate_expression_handling(self):
        """Test setting and getting rate expressions."""
        reaction = Reaction("A + B -> C")
        
        # Initially no rate expression
        self.assertIsNone(reaction.rate_expression)
        
        # Set rate expression
        reaction.set_rate_expression("k * [A] * [B]")
        self.assertEqual(reaction.rate_expression, "k * [A] * [B]")

    def test_rate_parameters(self):
        """Test adding and getting rate parameters."""
        reaction = Reaction("A -> B")
        
        # Add rate parameters
        reaction.add_rate_parameter('k', 0.5)
        reaction.add_rate_parameter('Ea', 25000)
        
        self.assertEqual(reaction.get_rate_parameter('k'), 0.5)
        self.assertEqual(reaction.get_rate_parameter('Ea'), 25000)
        self.assertIsNone(reaction.get_rate_parameter('nonexistent'))
        
        # Test backward compatibility with rate_constant
        self.assertEqual(reaction.rate_constant, 0.5)

    def test_initialization_with_rate_parameters(self):
        """Test initialization with rate parameters."""
        rate_params = {'k': 0.2, 'Ea': 30000, 'A': 1e6}
        reaction = Reaction(
            "A + B -> C",
            rate_expression="k * [A] * [B]",
            rate_parameters=rate_params
        )
        
        self.assertEqual(reaction.rate_parameters, rate_params)
        self.assertEqual(reaction.rate_expression, "k * [A] * [B]")
        self.assertEqual(reaction.rate_constant, 0.2)  # Backward compatibility

    def test_equilibrium_constant(self):
        """Test equilibrium constant for reversible reactions."""
        reaction = Reaction(
            "A + B <-> C",
            is_reversible=True,
            equilibrium_constant=5.0
        )
        
        self.assertTrue(reaction.is_reversible)
        self.assertEqual(reaction.equilibrium_constant, 5.0)

    def test_is_balanced(self):
        """Test reaction mass balance check."""
        # This is a placeholder test since the current implementation
        # is simplified and doesn't check actual mass balance
        reaction = Reaction("A + B -> C")
        self.assertTrue(reaction.is_balanced())
        
        # Test empty reaction
        empty_reaction = Reaction(reactants={}, products={})
        self.assertFalse(empty_reaction.is_balanced())

    def test_generate_reaction_string(self):
        """Test generation of reaction string from reactants and products."""
        reactants = {'A': 2.0, 'B': 1.0}
        products = {'C': 1.0, 'D': 3.0}
        
        # Test irreversible
        reaction1 = Reaction(reactants=reactants, products=products)
        expected1 = "2.0A + B -> C + 3.0D"
        self.assertEqual(reaction1.reaction_string, expected1)
        
        # Test reversible
        reaction2 = Reaction(reactants=reactants, products=products, is_reversible=True)
        expected2 = "2.0A + B <-> C + 3.0D"
        self.assertEqual(reaction2.reaction_string, expected2)

    def test_special_characters_in_species_names(self):
        """Test reactions with underscores and numbers in species names."""
        reaction = Reaction("H2O + CO2 -> H2CO3")
        
        self.assertEqual(reaction.reactants, {'H2O': 1.0, 'CO2': 1.0})
        self.assertEqual(reaction.products, {'H2CO3': 1.0})

    def test_whitespace_handling(self):
        """Test that whitespace is properly handled in reaction strings."""
        reaction1 = Reaction("A+B->C")  # No spaces
        reaction2 = Reaction("  A  +  B  ->  C  ")  # Extra spaces
        
        self.assertEqual(reaction1.reactants, {'A': 1.0, 'B': 1.0})
        self.assertEqual(reaction1.products, {'C': 1.0})
        self.assertEqual(reaction2.reactants, {'A': 1.0, 'B': 1.0})
        self.assertEqual(reaction2.products, {'C': 1.0})

    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        reaction = Reaction(
            "A + B -> C",
            rate_expression="k * [A] * [B]",
            rate_parameters={'k': 0.1}
        )
        
        # Test __repr__
        repr_str = repr(reaction)
        self.assertEqual(repr_str, "Reaction('A + B -> C')")
        
        # Test __str__
        str_repr = str(reaction)
        self.assertIn("Reaction: A + B -> C", str_repr)
        self.assertIn("Rate law: k * [A] * [B]", str_repr)
        self.assertIn("Parameters: {'k': 0.1}", str_repr)

    def test_edge_cases(self):
        """Test various edge cases."""
        # Single species reaction
        reaction1 = Reaction("A -> B")
        self.assertEqual(reaction1.reactants, {'A': 1.0})
        self.assertEqual(reaction1.products, {'B': 1.0})
        
        # Large stoichiometric coefficients
        reaction2 = Reaction("100A -> 50B")
        self.assertEqual(reaction2.reactants, {'A': 100.0})
        self.assertEqual(reaction2.products, {'B': 50.0})
        
        # Empty products (edge case - technically allowed by parser)
        reaction3 = Reaction("A + B -> ")
        self.assertEqual(reaction3.reactants, {'A': 1.0, 'B': 1.0})
        self.assertEqual(reaction3.products, {})

    def test_copy_and_modification(self):
        """Test that modifications don't affect original reaction."""
        original = Reaction("A + B -> C")
        
        # Get the species set
        species1 = original.get_all_species()
        species2 = original.get_all_species()
        
        # Modify one set
        species1.add('D')
        
        # Original should be unchanged
        self.assertEqual(species2, {'A', 'B', 'C'})


if __name__ == '__main__':
    # Set up test suite
    unittest.main(verbosity=2)