import unittest
import sys
import os
import numpy as np

# Add the parent directory to the path to import reactorpy modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the reactor classes and dependencies directly
from reactorpy.reactor import Reactor, BatchReactor, CSTR, PFR, SemiBatchReactor
from reactorpy.species import Species
from reactorpy.reaction import Reaction


class TestReactor(unittest.TestCase):
    """Test cases for the base Reactor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.reactor = Reactor(
            name="Test Reactor",
            volume=2.0,
            temperature=323.15,
            pressure=200000
        )
        
        # Create test species
        self.species_A = Species(name="A", concentration=1.0, molarW=0.050)
        self.species_B = Species(name="B", concentration=0.5, molarW=0.075)
        self.species_C = Species(name="C", concentration=0.0, molarW=0.100)
        
        # Create test reaction
        self.reaction = Reaction(
            "A + B -> C",
            rate_expression="k * [A] * [B]",
            rate_parameters={'k': 0.1}
        )

    def test_init(self):
        """Test reactor initialization."""
        self.assertEqual(self.reactor.name, "Test Reactor")
        self.assertEqual(self.reactor.volume, 2.0)
        self.assertEqual(self.reactor.temperature, 323.15)
        self.assertEqual(self.reactor.pressure, 200000)
        self.assertEqual(len(self.reactor.species), 0)
        self.assertEqual(len(self.reactor.reactions), 0)
        self.assertIsNone(self.reactor.ode_system)
        self.assertIsNone(self.reactor.simulation_results)

    def test_default_init(self):
        """Test reactor initialization with default values."""
        default_reactor = Reactor()
        self.assertEqual(default_reactor.name, "Generic Reactor")
        self.assertEqual(default_reactor.volume, 1.0)
        self.assertEqual(default_reactor.temperature, 298.15)
        self.assertEqual(default_reactor.pressure, 101325)

    def test_add_single_species(self):
        """Test adding a single species."""
        self.reactor.add_species(self.species_A)
        
        self.assertEqual(len(self.reactor.species), 1)
        self.assertIn("A", self.reactor.species)
        self.assertEqual(self.reactor.species["A"], self.species_A)

    def test_add_multiple_species(self):
        """Test adding multiple species as a list."""
        species_list = [self.species_A, self.species_B, self.species_C]
        self.reactor.add_species(species_list)
        
        self.assertEqual(len(self.reactor.species), 3)
        self.assertIn("A", self.reactor.species)
        self.assertIn("B", self.reactor.species)
        self.assertIn("C", self.reactor.species)

    def test_add_single_reaction(self):
        """Test adding a single reaction."""
        self.reactor.add_reaction(self.reaction)
        
        self.assertEqual(len(self.reactor.reactions), 1)
        self.assertEqual(self.reactor.reactions[0], self.reaction)

    def test_add_multiple_reactions(self):
        """Test adding multiple reactions as a list."""
        reaction2 = Reaction("C -> A + B", rate_parameters={'k': 0.05})
        reactions_list = [self.reaction, reaction2]
        
        self.reactor.add_reaction(reactions_list)
        
        self.assertEqual(len(self.reactor.reactions), 2)
        self.assertEqual(self.reactor.reactions[0], self.reaction)
        self.assertEqual(self.reactor.reactions[1], reaction2)

    def test_get_species(self):
        """Test getting species by name."""
        self.reactor.add_species(self.species_A)
        
        retrieved_species = self.reactor.get_species("A")
        self.assertEqual(retrieved_species, self.species_A)
        
        # Test non-existent species
        self.assertIsNone(self.reactor.get_species("NonExistent"))

    def test_get_initial_concentrations(self):
        """Test getting initial concentrations."""
        self.reactor.add_species([self.species_A, self.species_B, self.species_C])
        
        concentrations = self.reactor.get_initial_concentrations()
        expected = {"A": 1.0, "B": 0.5, "C": 0.0}
        self.assertEqual(concentrations, expected)

    def test_get_initial_concentrations_with_none(self):
        """Test getting initial concentrations when some are None."""
        species_none = Species(name="D", concentration=None)
        self.reactor.add_species([self.species_A, species_none])
        
        concentrations = self.reactor.get_initial_concentrations()
        expected = {"A": 1.0, "D": 0.0}
        self.assertEqual(concentrations, expected)

    def test_set_initial_concentration(self):
        """Test setting initial concentration for a species."""
        self.reactor.add_species(self.species_A)
        
        self.reactor.set_initial_concentration("A", 2.5)
        self.assertEqual(self.reactor.species["A"].concentration, 2.5)

    def test_set_initial_concentration_invalid_species(self):
        """Test setting concentration for non-existent species."""
        with self.assertRaises(ValueError):
            self.reactor.set_initial_concentration("NonExistent", 1.0)

    def test_build_ode_system_no_reactions(self):
        """Test building ODE system with no reactions."""
        with self.assertRaises(ValueError):
            self.reactor.build_ode_system()

    def test_build_ode_system_with_reactions(self):
        """Test building ODE system with reactions and species."""
        self.reactor.add_species([self.species_A, self.species_B, self.species_C])
        
        # Create a simple reaction without rate expression for testing
        simple_reaction = Reaction("A + B -> C")
        self.reactor.add_reaction(simple_reaction)
        
        # This should work without error
        ode_system = self.reactor.build_ode_system()
        self.assertIsNotNone(ode_system)
        self.assertIsNotNone(self.reactor.ode_system)

    def test_get_ode_system(self):
        """Test getting ODE system, building if necessary."""
        self.reactor.add_species([self.species_A, self.species_B, self.species_C])
        
        # Create a simple reaction without rate expression
        simple_reaction = Reaction("A + B -> C")
        self.reactor.add_reaction(simple_reaction)
        
        # Should build automatically
        ode_system = self.reactor.get_ode_system()
        self.assertIsNotNone(ode_system)

    def test_describe(self):
        """Test reactor description."""
        self.reactor.add_species([self.species_A, self.species_B])
        
        # Use simple reaction for description test
        simple_reaction = Reaction("A + B -> C")
        self.reactor.add_reaction(simple_reaction)
        
        description = self.reactor.describe()
        
        self.assertIn("Test Reactor", description)
        self.assertIn("2.0 m³", description)
        self.assertIn("323.15 K", description)
        self.assertIn("200000 Pa", description)
        self.assertIn("Species (2)", description)
        self.assertIn("Reactions (1)", description)
        self.assertIn("A + B -> C", description)

    def test_rate_expression_handling(self):
        """Test that reactions with rate expressions are stored correctly."""
        # Test that rate expressions are preserved even if they can't be parsed by ModelBuilder
        self.reactor.add_species([self.species_A, self.species_B, self.species_C])
        self.reactor.add_reaction(self.reaction)
        
        # Verify the reaction is stored with its rate expression
        self.assertEqual(len(self.reactor.reactions), 1)
        stored_reaction = self.reactor.reactions[0]
        self.assertEqual(stored_reaction.rate_expression, "k * [A] * [B]")
        self.assertEqual(stored_reaction.rate_parameters, {'k': 0.1})


class TestBatchReactor(unittest.TestCase):
    """Test cases for the BatchReactor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_reactor = BatchReactor(
            name="Test Batch",
            volume=1.5,
            temperature=300.0
        )

    def test_init(self):
        """Test BatchReactor initialization."""
        self.assertEqual(self.batch_reactor.name, "Test Batch")
        self.assertEqual(self.batch_reactor.volume, 1.5)
        self.assertEqual(self.batch_reactor.temperature, 300.0)
        self.assertEqual(self.batch_reactor.reactor_type, "Batch")

    def test_default_init(self):
        """Test BatchReactor with default values."""
        default_batch = BatchReactor()
        self.assertEqual(default_batch.name, "Batch Reactor")
        self.assertEqual(default_batch.reactor_type, "Batch")

    def test_describe(self):
        """Test BatchReactor description."""
        description = self.batch_reactor.describe()
        
        self.assertIn("Batch", description)
        self.assertIn("no inflow/outflow", description)
        self.assertIn("Time-dependent", description)


class TestCSTR(unittest.TestCase):
    """Test cases for the CSTR class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cstr = CSTR(
            name="Test CSTR",
            volume=2.0,
            flow_rate=0.1,
            inlet_concentrations={"A": 2.0, "B": 1.0}
        )

    def test_init(self):
        """Test CSTR initialization."""
        self.assertEqual(self.cstr.name, "Test CSTR")
        self.assertEqual(self.cstr.volume, 2.0)
        self.assertEqual(self.cstr.flow_rate, 0.1)
        self.assertEqual(self.cstr.reactor_type, "CSTR")
        self.assertEqual(self.cstr.inlet_concentrations, {"A": 2.0, "B": 1.0})

    def test_default_init(self):
        """Test CSTR with default values."""
        default_cstr = CSTR()
        self.assertEqual(default_cstr.name, "CSTR")
        self.assertEqual(default_cstr.flow_rate, 0.0)
        self.assertEqual(default_cstr.inlet_concentrations, {})

    def test_set_flow_rate(self):
        """Test setting flow rate."""
        self.cstr.set_flow_rate(0.2)
        self.assertEqual(self.cstr.flow_rate, 0.2)

    def test_set_inlet_concentration(self):
        """Test setting inlet concentration."""
        self.cstr.set_inlet_concentration("C", 0.5)
        self.assertEqual(self.cstr.inlet_concentrations["C"], 0.5)

    def test_get_residence_time(self):
        """Test residence time calculation."""
        # τ = V/Q = 2.0/0.1 = 20.0 s
        self.assertEqual(self.cstr.get_residence_time(), 20.0)

    def test_get_residence_time_zero_flow(self):
        """Test residence time with zero flow rate."""
        self.cstr.set_flow_rate(0.0)
        self.assertEqual(self.cstr.get_residence_time(), float('inf'))

    def test_describe(self):
        """Test CSTR description."""
        description = self.cstr.describe()
        
        self.assertIn("CSTR", description)
        self.assertIn("0.1 m³/s", description)
        self.assertIn("20.00 s", description)  # Residence time
        self.assertIn("Inlet Concentrations", description)
        self.assertIn("A: 2.0", description)
        self.assertIn("B: 1.0", description)


class TestPFR(unittest.TestCase):
    """Test cases for the PFR class."""

    def setUp(self):
        """Set up test fixtures."""
        self.pfr = PFR(
            name="Test PFR",
            volume=5.0,
            length=10.0,
            flow_rate=0.2,
            n_segments=5
        )

    def test_init(self):
        """Test PFR initialization."""
        self.assertEqual(self.pfr.name, "Test PFR")
        self.assertEqual(self.pfr.volume, 5.0)
        self.assertEqual(self.pfr.length, 10.0)
        self.assertEqual(self.pfr.flow_rate, 0.2)
        self.assertEqual(self.pfr.n_segments, 5)
        self.assertEqual(self.pfr.reactor_type, "PFR")

    def test_calculated_properties(self):
        """Test calculated properties."""
        # Cross-sectional area = V/L = 5.0/10.0 = 0.5 m²
        self.assertEqual(self.pfr.cross_sectional_area, 0.5)
        
        # Segment length = L/n = 10.0/5 = 2.0 m
        self.assertEqual(self.pfr.segment_length, 2.0)
        
        # Segment volume = V/n = 5.0/5 = 1.0 m³
        self.assertEqual(self.pfr.segment_volume, 1.0)

    def test_get_superficial_velocity(self):
        """Test superficial velocity calculation."""
        # u = Q/A = 0.2/0.5 = 0.4 m/s
        self.assertEqual(self.pfr.get_superficial_velocity(), 0.4)

    def test_get_residence_time(self):
        """Test residence time calculation."""
        # τ = V/Q = 5.0/0.2 = 25.0 s
        self.assertEqual(self.pfr.get_residence_time(), 25.0)

    def test_get_residence_time_zero_flow(self):
        """Test residence time with zero flow rate."""
        pfr_zero_flow = PFR(volume=5.0, flow_rate=0.0)
        self.assertEqual(pfr_zero_flow.get_residence_time(), float('inf'))

    def test_describe(self):
        """Test PFR description."""
        description = self.pfr.describe()
        
        self.assertIn("PFR", description)
        self.assertIn("10.0 m", description)  # Length
        self.assertIn("0.5000 m²", description)  # Cross-sectional area
        self.assertIn("0.2 m³/s", description)  # Flow rate
        self.assertIn("0.4000 m/s", description)  # Superficial velocity
        self.assertIn("25.00 s", description)  # Residence time
        self.assertIn("5", description)  # Segments
        self.assertIn("2.0000 m", description)  # Segment length


class TestSemiBatchReactor(unittest.TestCase):
    """Test cases for the SemiBatchReactor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.semi_batch = SemiBatchReactor(
            name="Test Semi-Batch",
            initial_volume=1.0
        )

    def test_init(self):
        """Test SemiBatchReactor initialization."""
        self.assertEqual(self.semi_batch.name, "Test Semi-Batch")
        self.assertEqual(self.semi_batch.initial_volume, 1.0)
        self.assertEqual(self.semi_batch.volume, 1.0)
        self.assertEqual(self.semi_batch.reactor_type, "Semi-Batch")
        self.assertEqual(self.semi_batch.feed_schedule, {})

    def test_default_init(self):
        """Test SemiBatchReactor with default values."""
        default_semi = SemiBatchReactor()
        self.assertEqual(default_semi.name, "Semi-Batch Reactor")

    def test_add_feed(self):
        """Test adding feed schedule."""
        self.semi_batch.add_feed(
            species_name="A",
            flow_rate=0.01,
            concentration=5.0,
            start_time=10.0,
            end_time=100.0
        )
        
        expected_feed = {
            'flow_rate': 0.01,
            'concentration': 5.0,
            'start_time': 10.0,
            'end_time': 100.0
        }
        
        self.assertEqual(self.semi_batch.feed_schedule["A"], expected_feed)

    def test_add_feed_default_times(self):
        """Test adding feed with default start/end times."""
        self.semi_batch.add_feed("B", 0.02, 3.0)
        
        feed_info = self.semi_batch.feed_schedule["B"]
        self.assertEqual(feed_info['start_time'], 0.0)
        self.assertEqual(feed_info['end_time'], float('inf'))

    def test_describe(self):
        """Test SemiBatchReactor description."""
        self.semi_batch.add_feed("A", 0.01, 5.0, 10.0, 100.0)
        
        description = self.semi_batch.describe()
        
        self.assertIn("Semi-Batch", description)
        self.assertIn("Initial Volume: 1.0 m³", description)
        self.assertIn("Feed Schedule", description)
        self.assertIn("A: 5.0 mol/L at 0.01 m³/s", description)
        self.assertIn("Time: 10.0 - 100.0 s", description)


class TestReactorIntegration(unittest.TestCase):
    """Integration tests for reactor functionality."""

    def setUp(self):
        """Set up test fixtures for integration tests."""
        # Create a simple batch reactor with reaction
        self.reactor = BatchReactor("Integration Test")
        
        # Add species
        species_A = Species("A", concentration=2.0)
        species_B = Species("B", concentration=1.0)
        species_C = Species("C", concentration=0.0)
        self.reactor.add_species([species_A, species_B, species_C])
        
        # Add reaction
        reaction = Reaction(
            "A + B -> C",
            rate_expression="k * [A] * [B]",
            rate_parameters={'k': 0.1}
        )
        self.reactor.add_reaction(reaction)

    def test_complete_workflow(self):
        """Test complete reactor workflow."""
        # Build ODE system (using simple reaction without rate expression)
        simple_reaction = Reaction("A + B -> C")
        self.reactor.reactions = [simple_reaction]  # Replace the reaction with rate expression
        
        ode_system = self.reactor.build_ode_system()
        self.assertIsNotNone(ode_system)
        
        # Get initial concentrations
        initial_conc = self.reactor.get_initial_concentrations()
        expected_conc = {"A": 2.0, "B": 1.0, "C": 0.0}
        self.assertEqual(initial_conc, expected_conc)
        
        # Check description
        description = self.reactor.describe()
        self.assertIn("Integration Test", description)
        self.assertIn("A + B -> C", description)

    def test_modification_after_setup(self):
        """Test modifying reactor after initial setup."""
        # Add another species
        species_D = Species("D", concentration=0.5)
        self.reactor.add_species(species_D)
        
        # Modify existing species concentration
        self.reactor.set_initial_concentration("A", 3.0)
        
        # Verify changes
        self.assertEqual(len(self.reactor.species), 4)
        self.assertEqual(self.reactor.species["A"].concentration, 3.0)
        self.assertEqual(self.reactor.species["D"].concentration, 0.5)

    def test_multiple_reactions(self):
        """Test reactor with multiple reactions."""
        # Replace with simple reactions
        simple_reaction1 = Reaction("A + B -> C")
        simple_reaction2 = Reaction("C -> A + B")
        self.reactor.reactions = [simple_reaction1, simple_reaction2]
        
        self.assertEqual(len(self.reactor.reactions), 2)
        
        # Should still be able to build ODE system
        ode_system = self.reactor.build_ode_system()
        self.assertIsNotNone(ode_system)


class TestReactorTypes(unittest.TestCase):
    """Test different reactor type behaviors."""

    def test_reactor_type_inheritance(self):
        """Test that all reactor types inherit from base Reactor."""
        batch = BatchReactor()
        cstr = CSTR()
        pfr = PFR()
        semi_batch = SemiBatchReactor()
        
        self.assertIsInstance(batch, Reactor)
        self.assertIsInstance(cstr, Reactor)
        self.assertIsInstance(pfr, Reactor)
        self.assertIsInstance(semi_batch, Reactor)

    def test_reactor_type_attributes(self):
        """Test reactor type specific attributes."""
        batch = BatchReactor()
        cstr = CSTR()
        pfr = PFR(length=5.0, n_segments=10)
        semi_batch = SemiBatchReactor()
        
        self.assertEqual(batch.reactor_type, "Batch")
        self.assertEqual(cstr.reactor_type, "CSTR")
        self.assertEqual(pfr.reactor_type, "PFR")
        self.assertEqual(semi_batch.reactor_type, "Semi-Batch")
        
        # Check PFR specific attributes
        self.assertEqual(pfr.length, 5.0)
        self.assertEqual(pfr.n_segments, 10)


if __name__ == '__main__':
    # Set up test suite
    unittest.main(verbosity=2)