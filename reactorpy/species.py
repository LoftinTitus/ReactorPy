class Species:
    def __init__(self, name, concentration=None, density=None, volume=None, molarW=None,
                 molecular_formula=None, phase='liquid', temperature=298.15, pressure=101325,
                 enthalpy_formation=None, entropy=None, heat_capacity_coeffs=None,
                 diffusivity=None, viscosity=None, thermal_conductivity=None,
                 activity_coefficient=1.0, charge=0, cas_number=None):
        """
        Initialize a chemical species with properties needed for reactor calculations.
        
        Args:
            name (str): Species name
            concentration (float): Molar concentration (mol/L or mol/m³)
            density (float): Density (kg/m³)
            volume (float): Volume (m³)
            molarW (float): Molar weight (kg/mol or g/mol)
            molecular_formula (str): Chemical formula (e.g., 'C2H4O')
            phase (str): Physical phase ('gas', 'liquid', 'solid', 'aqueous')
            temperature (float): Temperature (K)
            pressure (float): Pressure (Pa)
            enthalpy_formation (float): Standard enthalpy of formation (J/mol)
            entropy (float): Standard entropy (J/mol·K)
            heat_capacity_coeffs (dict): Heat capacity coefficients for Cp = a + bT + cT² + dT³
            diffusivity (float): Diffusion coefficient (m²/s)
            viscosity (float): Dynamic viscosity (Pa·s)
            thermal_conductivity (float): Thermal conductivity (W/m·K)
            activity_coefficient (float): Activity coefficient for non-ideal solutions
            charge (int): Ionic charge (for electrolytes)
            cas_number (str): CAS registry number for identification
        """
        # Basic identification
        self.name = name
        self.molecular_formula = molecular_formula
        self.cas_number = cas_number
        self.molarW = molarW
        
        # State properties
        self.concentration = concentration
        self.density = density
        self.volume = volume
        self.phase = phase
        self.temperature = temperature
        self.pressure = pressure
        self.activity_coefficient = activity_coefficient
        self.charge = charge
        
        # Thermodynamic properties
        self.enthalpy_formation = enthalpy_formation  # Standard enthalpy of formation
        self.entropy = entropy  # Standard entropy
        self.heat_capacity_coeffs = heat_capacity_coeffs or {}  # For temperature-dependent Cp
        
        # Transport properties
        self.diffusivity = diffusivity
        self.viscosity = viscosity
        self.thermal_conductivity = thermal_conductivity

    def calculate_heat_capacity(self, temperature=None):
        """Calculate heat capacity at given temperature using polynomial coefficients."""
        if not self.heat_capacity_coeffs:
            return None
        
        T = temperature or self.temperature
        coeffs = self.heat_capacity_coeffs
        
        # Cp = a + bT + cT² + dT³
        cp = coeffs.get('a', 0)
        cp += coeffs.get('b', 0) * T
        cp += coeffs.get('c', 0) * T**2
        cp += coeffs.get('d', 0) * T**3
        
        return cp
    
    def calculate_enthalpy(self, temperature=None, reference_temp=298.15):
        """Calculate enthalpy at given temperature."""
        if self.enthalpy_formation is None:
            return None
            
        T = temperature or self.temperature
        
        if self.heat_capacity_coeffs and T != reference_temp:
            # H(T) = H_f + ∫Cp dT from reference_temp to T
            coeffs = self.heat_capacity_coeffs
            dT = T - reference_temp
            
            enthalpy_correction = (
                coeffs.get('a', 0) * dT +
                coeffs.get('b', 0) * (T**2 - reference_temp**2) / 2 +
                coeffs.get('c', 0) * (T**3 - reference_temp**3) / 3 +
                coeffs.get('d', 0) * (T**4 - reference_temp**4) / 4
            )
            
            return self.enthalpy_formation + enthalpy_correction
        
        return self.enthalpy_formation
    
    def calculate_moles(self):
        """Calculate number of moles from concentration and volume."""
        if self.concentration is not None and self.volume is not None:
            return self.concentration * self.volume
        return None
    
    def calculate_mass(self):
        """Calculate mass from moles and molar weight."""
        moles = self.calculate_moles()
        if moles is not None and self.molarW is not None:
            return moles * self.molarW
        return None
    
    def update_concentration(self, new_concentration):
        """Update concentration and recalculate dependent properties."""
        self.concentration = new_concentration
    
    def is_ionic(self):
        """Check if species is ionic."""
        return self.charge != 0
    
    def describe(self):
        """Return detailed description of the species."""
        desc = f"Species: {self.name}"
        if self.molecular_formula:
            desc += f" ({self.molecular_formula})"
        desc += f"\n  Phase: {self.phase}"
        desc += f"\n  Temperature: {self.temperature} K"
        desc += f"\n  Pressure: {self.pressure} Pa"
        
        if self.concentration is not None:
            desc += f"\n  Concentration: {self.concentration} mol/L"
        if self.density is not None:
            desc += f"\n  Density: {self.density} kg/m³"
        if self.molarW is not None:
            desc += f"\n  Molar Weight: {self.molarW} kg/mol"
        if self.volume is not None:
            desc += f"\n  Volume: {self.volume} m³"
        
        moles = self.calculate_moles()
        if moles is not None:
            desc += f"\n  Moles: {moles} mol"
        
        mass = self.calculate_mass()
        if mass is not None:
            desc += f"\n  Mass: {mass} kg"
            
        if self.is_ionic():
            desc += f"\n  Charge: {self.charge}"
            
        return desc