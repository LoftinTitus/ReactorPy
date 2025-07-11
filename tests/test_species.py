import unittest
import sys
import os

# Add the parent directory to the path to import reactorpy modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the Species class directly
from reactorpy.species import Species


class TestSpecies(unittest.TestCase):
    """Test cases for the Species class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Basic species with minimal properties
        self.basic_species = Species(name="Water", concentration=1.0)
        
        # Comprehensive species with all properties
        self.comprehensive_species = Species(
            name="Ethanol",
            concentration=2.5,
            density=789.0,
            volume=0.1,
            molarW=0.046068,  # kg/mol
            molecular_formula="C2H5OH",
            phase="liquid",
            temperature=298.15,
            pressure=101325,
            enthalpy_formation=-277690,  # J/mol
            entropy=160.7,  # J/mol·K
            heat_capacity_coeffs={'a': 61.34, 'b': 0.0015, 'c': -8.749e-6, 'd': 1.188e-8},
            diffusivity=1.2e-9,  # m²/s
            viscosity=1.074e-3,  # Pa·s
            thermal_conductivity=0.167,  # W/m·K
            activity_coefficient=0.95,
            charge=0,
            cas_number="64-17-5"
        )
        
        # Ionic species
        self.ionic_species = Species(
            name="Na+",
            concentration=0.1,
            molecular_formula="Na+",
            phase="aqueous",
            charge=1,
            molarW=0.02299  # kg/mol
        )

    def test_basic_initialization(self):
        """Test basic species initialization."""
        self.assertEqual(self.basic_species.name, "Water")
        self.assertEqual(self.basic_species.concentration, 1.0)
        self.assertEqual(self.basic_species.phase, "liquid")  # default
        self.assertEqual(self.basic_species.temperature, 298.15)  # default
        self.assertEqual(self.basic_species.pressure, 101325)  # default
        self.assertEqual(self.basic_species.activity_coefficient, 1.0)  # default
        self.assertEqual(self.basic_species.charge, 0)  # default

    def test_comprehensive_initialization(self):
        """Test comprehensive species initialization with all properties."""
        species = self.comprehensive_species
        
        # Basic identification
        self.assertEqual(species.name, "Ethanol")
        self.assertEqual(species.molecular_formula, "C2H5OH")
        self.assertEqual(species.cas_number, "64-17-5")
        self.assertEqual(species.molarW, 0.046068)
        
        # State properties
        self.assertEqual(species.concentration, 2.5)
        self.assertEqual(species.density, 789.0)
        self.assertEqual(species.volume, 0.1)
        self.assertEqual(species.phase, "liquid")
        self.assertEqual(species.temperature, 298.15)
        self.assertEqual(species.pressure, 101325)
        self.assertEqual(species.activity_coefficient, 0.95)
        self.assertEqual(species.charge, 0)
        
        # Thermodynamic properties
        self.assertEqual(species.enthalpy_formation, -277690)
        self.assertEqual(species.entropy, 160.7)
        self.assertIsInstance(species.heat_capacity_coeffs, dict)
        
        # Transport properties
        self.assertEqual(species.diffusivity, 1.2e-9)
        self.assertEqual(species.viscosity, 1.074e-3)
        self.assertEqual(species.thermal_conductivity, 0.167)

    def test_default_values(self):
        """Test that default values are properly set."""
        species = Species("TestSpecies")
        
        self.assertEqual(species.name, "TestSpecies")
        self.assertIsNone(species.concentration)
        self.assertIsNone(species.density)
        self.assertIsNone(species.volume)
        self.assertIsNone(species.molarW)
        self.assertIsNone(species.molecular_formula)
        self.assertEqual(species.phase, "liquid")
        self.assertEqual(species.temperature, 298.15)
        self.assertEqual(species.pressure, 101325)
        self.assertIsNone(species.enthalpy_formation)
        self.assertIsNone(species.entropy)
        self.assertEqual(species.heat_capacity_coeffs, {})
        self.assertIsNone(species.diffusivity)
        self.assertIsNone(species.viscosity)
        self.assertIsNone(species.thermal_conductivity)
        self.assertEqual(species.activity_coefficient, 1.0)
        self.assertEqual(species.charge, 0)
        self.assertIsNone(species.cas_number)

    def test_calculate_heat_capacity_no_coefficients(self):
        """Test heat capacity calculation with no coefficients."""
        result = self.basic_species.calculate_heat_capacity()
        self.assertIsNone(result)

    def test_calculate_heat_capacity_with_coefficients(self):
        """Test heat capacity calculation with polynomial coefficients."""
        # Using ethanol at 298.15 K
        cp = self.comprehensive_species.calculate_heat_capacity(298.15)
        
        # Cp = a + bT + cT² + dT³
        expected = (61.34 + 
                   0.0015 * 298.15 + 
                   (-8.749e-6) * (298.15**2) + 
                   1.188e-8 * (298.15**3))
        
        self.assertAlmostEqual(cp, expected, places=5)

    def test_calculate_heat_capacity_different_temperature(self):
        """Test heat capacity calculation at different temperature."""
        cp_300 = self.comprehensive_species.calculate_heat_capacity(300.0)
        cp_350 = self.comprehensive_species.calculate_heat_capacity(350.0)
        
        # Heat capacity should change with temperature
        self.assertNotEqual(cp_300, cp_350)
        self.assertIsInstance(cp_300, float)
        self.assertIsInstance(cp_350, float)

    def test_calculate_heat_capacity_default_temperature(self):
        """Test heat capacity calculation using default temperature."""
        cp_default = self.comprehensive_species.calculate_heat_capacity()
        cp_explicit = self.comprehensive_species.calculate_heat_capacity(298.15)
        
        self.assertEqual(cp_default, cp_explicit)

    def test_calculate_enthalpy_no_formation_enthalpy(self):
        """Test enthalpy calculation with no formation enthalpy."""
        result = self.basic_species.calculate_enthalpy()
        self.assertIsNone(result)

    def test_calculate_enthalpy_at_reference_temperature(self):
        """Test enthalpy calculation at reference temperature."""
        enthalpy = self.comprehensive_species.calculate_enthalpy(298.15)
        self.assertEqual(enthalpy, -277690)  # Should equal formation enthalpy

    def test_calculate_enthalpy_different_temperature(self):
        """Test enthalpy calculation at different temperature."""
        enthalpy_300 = self.comprehensive_species.calculate_enthalpy(300.0)
        enthalpy_350 = self.comprehensive_species.calculate_enthalpy(350.0)
        
        # Should be different from formation enthalpy
        self.assertNotEqual(enthalpy_300, -277690)
        self.assertNotEqual(enthalpy_350, -277690)
        self.assertNotEqual(enthalpy_300, enthalpy_350)

    def test_calculate_enthalpy_without_heat_capacity(self):
        """Test enthalpy calculation without heat capacity coefficients."""
        species = Species(
            "Test",
            enthalpy_formation=-100000,
            temperature=350.0
        )
        
        enthalpy = species.calculate_enthalpy(350.0)
        self.assertEqual(enthalpy, -100000)  # Should return formation enthalpy

    def test_calculate_moles_with_values(self):
        """Test moles calculation with concentration and volume."""
        moles = self.comprehensive_species.calculate_moles()
        expected = 2.5 * 0.1  # concentration * volume
        self.assertEqual(moles, expected)

    def test_calculate_moles_missing_concentration(self):
        """Test moles calculation with missing concentration."""
        species = Species("Test", volume=0.1)
        moles = species.calculate_moles()
        self.assertIsNone(moles)

    def test_calculate_moles_missing_volume(self):
        """Test moles calculation with missing volume."""
        species = Species("Test", concentration=1.0)
        moles = species.calculate_moles()
        self.assertIsNone(moles)

    def test_calculate_mass_with_values(self):
        """Test mass calculation with all required values."""
        mass = self.comprehensive_species.calculate_mass()
        expected_moles = 2.5 * 0.1  # concentration * volume
        expected_mass = expected_moles * 0.046068  # moles * molar weight
        self.assertAlmostEqual(mass, expected_mass, places=8)

    def test_calculate_mass_missing_moles(self):
        """Test mass calculation when moles can't be calculated."""
        species = Species("Test", molarW=0.05)
        mass = species.calculate_mass()
        self.assertIsNone(mass)

    def test_calculate_mass_missing_molar_weight(self):
        """Test mass calculation with missing molar weight."""
        species = Species("Test", concentration=1.0, volume=0.1)
        mass = species.calculate_mass()
        self.assertIsNone(mass)

    def test_update_concentration(self):
        """Test updating concentration."""
        original_concentration = self.comprehensive_species.concentration
        new_concentration = 5.0
        
        self.comprehensive_species.update_concentration(new_concentration)
        
        self.assertNotEqual(self.comprehensive_species.concentration, original_concentration)
        self.assertEqual(self.comprehensive_species.concentration, new_concentration)

    def test_is_ionic_neutral_species(self):
        """Test ionic check for neutral species."""
        self.assertFalse(self.basic_species.is_ionic())
        self.assertFalse(self.comprehensive_species.is_ionic())

    def test_is_ionic_charged_species(self):
        """Test ionic check for charged species."""
        self.assertTrue(self.ionic_species.is_ionic())

    def test_is_ionic_negative_charge(self):
        """Test ionic check for negatively charged species."""
        anion = Species("Cl-", charge=-1)
        self.assertTrue(anion.is_ionic())

    def test_describe_basic_species(self):
        """Test description for basic species."""
        description = self.basic_species.describe()
        
        self.assertIn("Species: Water", description)
        self.assertIn("Phase: liquid", description)
        self.assertIn("Temperature: 298.15 K", description)
        self.assertIn("Pressure: 101325 Pa", description)
        self.assertIn("Concentration: 1.0 mol/L", description)

    def test_describe_comprehensive_species(self):
        """Test description for comprehensive species."""
        description = self.comprehensive_species.describe()
        
        self.assertIn("Species: Ethanol (C2H5OH)", description)
        self.assertIn("Phase: liquid", description)
        self.assertIn("Temperature: 298.15 K", description)
        self.assertIn("Pressure: 101325 Pa", description)
        self.assertIn("Concentration: 2.5 mol/L", description)
        self.assertIn("Density: 789.0 kg/m³", description)
        self.assertIn("Molar Weight: 0.046068 kg/mol", description)
        self.assertIn("Volume: 0.1 m³", description)
        self.assertIn("Moles:", description)
        self.assertIn("Mass:", description)

    def test_describe_ionic_species(self):
        """Test description for ionic species."""
        description = self.ionic_species.describe()
        
        self.assertIn("Species: Na+ (Na+)", description)
        self.assertIn("Phase: aqueous", description)
        self.assertIn("Charge: 1", description)

    def test_phase_types(self):
        """Test different phase types."""
        gas_species = Species("H2", phase="gas")
        solid_species = Species("NaCl", phase="solid")
        aqueous_species = Species("HCl", phase="aqueous")
        
        self.assertEqual(gas_species.phase, "gas")
        self.assertEqual(solid_species.phase, "solid")
        self.assertEqual(aqueous_species.phase, "aqueous")

    def test_heat_capacity_coefficients_structure(self):
        """Test heat capacity coefficients structure."""
        coeffs = self.comprehensive_species.heat_capacity_coeffs
        
        self.assertIn('a', coeffs)
        self.assertIn('b', coeffs)
        self.assertIn('c', coeffs)
        self.assertIn('d', coeffs)
        
        # Test that they are numeric values
        for key, value in coeffs.items():
            self.assertIsInstance(value, (int, float))

    def test_activity_coefficient_range(self):
        """Test activity coefficient values."""
        # Test different activity coefficients
        ideal_species = Species("Ideal", activity_coefficient=1.0)
        non_ideal_species = Species("NonIdeal", activity_coefficient=0.8)
        
        self.assertEqual(ideal_species.activity_coefficient, 1.0)
        self.assertEqual(non_ideal_species.activity_coefficient, 0.8)

    def test_temperature_pressure_effects(self):
        """Test species at different temperatures and pressures."""
        high_temp_species = Species("HighTemp", temperature=500.0)
        high_pressure_species = Species("HighPressure", pressure=1000000)
        
        self.assertEqual(high_temp_species.temperature, 500.0)
        self.assertEqual(high_pressure_species.pressure, 1000000)

    def test_transport_properties(self):
        """Test transport properties."""
        species = self.comprehensive_species
        
        self.assertIsNotNone(species.diffusivity)
        self.assertIsNotNone(species.viscosity)
        self.assertIsNotNone(species.thermal_conductivity)
        
        self.assertGreater(species.diffusivity, 0)
        self.assertGreater(species.viscosity, 0)
        self.assertGreater(species.thermal_conductivity, 0)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Zero concentration
        zero_conc_species = Species("Zero", concentration=0.0)
        self.assertEqual(zero_conc_species.concentration, 0.0)
        
        # Very small molar weight
        small_mw_species = Species("Small", molarW=1e-6)
        self.assertEqual(small_mw_species.molarW, 1e-6)
        
        # Negative charge
        anion = Species("Anion", charge=-2)
        self.assertEqual(anion.charge, -2)
        self.assertTrue(anion.is_ionic())

    def test_species_equality_by_name(self):
        """Test that species with same name are conceptually the same."""
        species1 = Species("Water", concentration=1.0)
        species2 = Species("Water", concentration=2.0)
        
        # They should have the same name even with different properties
        self.assertEqual(species1.name, species2.name)
        self.assertNotEqual(species1.concentration, species2.concentration)

    def test_molar_weight_units_consistency(self):
        """Test molar weight consistency (should be in kg/mol)."""
        # Test common species molar weights in kg/mol
        water = Species("H2O", molarW=0.018015)  # kg/mol
        oxygen = Species("O2", molarW=0.032)     # kg/mol
        
        self.assertAlmostEqual(water.molarW, 0.018015, places=6)
        self.assertAlmostEqual(oxygen.molarW, 0.032, places=3)

    def test_concentration_units_consistency(self):
        """Test concentration consistency (should be in mol/L or mol/m³)."""
        # Standard concentration in mol/L
        species = Species("Test", concentration=1.5)
        
        # Calculate moles for 1 m³
        species.volume = 1.0  # m³
        moles = species.calculate_moles()
        
        self.assertEqual(moles, 1.5)  # mol


if __name__ == '__main__':
    # Set up test suite
    unittest.main(verbosity=2)