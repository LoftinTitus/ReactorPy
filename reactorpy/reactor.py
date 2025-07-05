import numpy as np
from typing import List, Dict, Optional, Union
from .model_builder import ModelBuilder
from .species import Species
from .reaction import Reaction


class Reactor:
    """
    Generic reactor class that stores species and reactions, interfaces with ModelBuilder,
    and provides simulation capabilities.
    """
    
    def __init__(self, name: str = "Generic Reactor", volume: float = 1.0, temperature: float = 298.15, pressure: float = 101325):
        """
        Initialize a generic reactor.
        
        Args:
            name (str): Reactor name/identifier
            volume (float): Reactor volume (m³)
            temperature (float): Operating temperature (K)
            pressure (float): Operating pressure (Pa)
        """
        self.name = name
        self.volume = volume
        self.temperature = temperature
        self.pressure = pressure
        
        # Storage for species and reactions
        self.species = {}  # Dict: species_name -> Species object
        self.reactions = []  # List of Reaction objects
        
        # Model building
        self.model_builder = ModelBuilder()
        self.ode_system = None
        self.symbolic_system = None
        
        # Simulation results
        self.simulation_results = None
        self.time_points = None
        
    def add_species(self, species: Union[Species, List[Species]]):
        """
        Add one or more species to the reactor.
        
        Args:
            species: Single Species object or list of Species objects
        """
        if isinstance(species, list):
            for s in species:
                self.species[s.name] = s
        else:
            self.species[species.name] = species
    
    def add_reaction(self, reaction: Union[Reaction, List[Reaction]]):
        """
        Add one or more reactions to the reactor.
        
        Args:
            reaction: Single Reaction object or list of Reaction objects
        """
        if isinstance(reaction, list):
            self.reactions.extend(reaction)
        else:
            self.reactions.append(reaction)
    
    def get_species(self, name: str) -> Optional[Species]:
        """Get a species by name."""
        return self.species.get(name)
    
    def get_initial_concentrations(self) -> Dict[str, float]:
        """Get initial concentrations for all species."""
        concentrations = {}
        for name, species in self.species.items():
            concentrations[name] = species.concentration if species.concentration is not None else 0.0
        return concentrations
    
    def set_initial_concentration(self, species_name: str, concentration: float):
        """Set initial concentration for a species."""
        if species_name in self.species:
            self.species[species_name].concentration = concentration
        else:
            raise ValueError(f"Species '{species_name}' not found in reactor")
    
    def build_ode_system(self):
        """
        Build the symbolic ODE system using ModelBuilder.
        This is the core method that interfaces with ModelBuilder.
        """
        if not self.reactions:
            raise ValueError("No reactions defined. Add reactions before building ODE system.")
        
        # Get species list
        species_objects = list(self.species.values())
        
        # Generate symbolic ODE system
        self.symbolic_system = self.model_builder.generate_ode_system_with_species_objects(
            self.reactions, species_objects
        )
        
        # Store for easy access
        self.ode_system = self.symbolic_system
        
        return self.symbolic_system
    
    def get_ode_system(self) -> Dict:
        """Get the current ODE system, building it if necessary."""
        if self.ode_system is None:
            self.build_ode_system()
        return self.ode_system
    
    def simulate(self, time_span: tuple, method: str = 'RK45', **kwargs):
        """
        Simulate the reactor over time.
        
        Args:
            time_span (tuple): (t_start, t_end) for simulation
            method (str): Integration method for solver
            **kwargs: Additional arguments passed to solver
            
        Returns:
            dict: Simulation results containing time points and concentrations
        """
        # Import here to avoid circular imports
        from .solver import solve_ode_system
        
        # Build ODE system if not already done
        if self.ode_system is None:
            self.build_ode_system()
        
        # Get initial conditions
        initial_concentrations = self.get_initial_concentrations()
        
        # Collect all rate parameters
        all_rate_params = {}
        for reaction in self.reactions:
            all_rate_params.update(reaction.rate_parameters)
        
        # Run simulation
        self.simulation_results = solve_ode_system(
            self.ode_system,
            initial_concentrations,
            time_span,
            rate_parameters=all_rate_params,
            method=method,
            **kwargs
        )
        
        # Extract time points for plotting
        if 'time' in self.simulation_results:
            self.time_points = self.simulation_results['time']
        
        return self.simulation_results
    
    def describe(self) -> str:
        """Return a detailed description of the reactor."""
        desc = f"Reactor: {self.name}\n"
        desc += f"Type: {self.__class__.__name__}\n"
        desc += f"Volume: {self.volume} m³\n"
        desc += f"Temperature: {self.temperature} K\n"
        desc += f"Pressure: {self.pressure} Pa\n\n"
        
        desc += f"Species ({len(self.species)}):\n"
        for name, species in self.species.items():
            conc = species.concentration if species.concentration is not None else "Not set"
            desc += f"  {name}: {conc} mol/L\n"
        
        desc += f"\nReactions ({len(self.reactions)}):\n"
        for i, reaction in enumerate(self.reactions, 1):
            desc += f"  {i}. {reaction.reaction_string}\n"
            if reaction.rate_expression:
                desc += f"     Rate: {reaction.rate_expression}\n"
        
        if self.ode_system:
            desc += f"\nODE System:\n"
            for species, ode in self.ode_system.items():
                desc += f"  d[{species}]/dt = {ode}\n"
        
        return desc
    
    def plot(self, species_names: Optional[List[str]] = None, **kwargs):
        """
        Plot simulation results.
        
        Args:
            species_names: List of species to plot. If None, plots all species.
            **kwargs: Additional plotting arguments
        """
        if self.simulation_results is None:
            raise ValueError("No simulation results available. Run simulate() first.")
        
        # Import here to avoid circular imports
        from .plotting import plot_concentration_profiles
        
        if species_names is None:
            species_names = list(self.species.keys())
        
        return plot_concentration_profiles(
            self.simulation_results,
            species_names,
            title=f"{self.name} - Concentration Profiles",
            **kwargs
        )


class BatchReactor(Reactor):
    """
    Batch reactor with no inflow/outflow - purely time-dependent.
    """
    
    def __init__(self, name: str = "Batch Reactor", volume: float = 1.0, **kwargs):
        super().__init__(name, volume, **kwargs)
        self.reactor_type = "Batch"
    
    def describe(self) -> str:
        desc = super().describe()
        desc += f"\nReactor Type: Batch (no inflow/outflow)\n"
        desc += f"Operating Mode: Time-dependent\n"
        return desc


class CSTR(Reactor):
    """
    Continuous Stirred Tank Reactor with inflow/outflow terms.
    """
    
    def __init__(self, name: str = "CSTR", volume: float = 1.0, 
                 flow_rate: float = 0.0, inlet_concentrations: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(name, volume, **kwargs)
        self.reactor_type = "CSTR"
        self.flow_rate = flow_rate  # m³/s
        self.inlet_concentrations = inlet_concentrations or {}
        
    def set_flow_rate(self, flow_rate: float):
        """Set the volumetric flow rate."""
        self.flow_rate = flow_rate
    
    def set_inlet_concentration(self, species_name: str, concentration: float):
        """Set inlet concentration for a species."""
        self.inlet_concentrations[species_name] = concentration
    
    def get_residence_time(self) -> float:
        """Calculate residence time (τ = V/Q)."""
        if self.flow_rate == 0:
            return float('inf')
        return self.volume / self.flow_rate
    
    def build_ode_system(self):
        """
        Build ODE system for CSTR including flow terms.
        For each species: dC/dt = reaction_rate + (C_in - C_out)/τ
        """
        # Get base reaction ODEs
        super().build_ode_system()
        
        # Add flow terms for each species
        if self.flow_rate > 0:
            residence_time = self.get_residence_time()
            
            for species_name in self.species.keys():
                # Get inlet concentration (0 if not specified)
                C_in = self.inlet_concentrations.get(species_name, 0.0)
                
                # Get concentration symbol
                C_symbol = self.model_builder.concentration_symbols[species_name]
                
                # Add flow term: (C_in - C)/τ
                flow_term = (C_in - C_symbol) / residence_time
                self.ode_system[species_name] += flow_term
    
    def describe(self) -> str:
        desc = super().describe()
        desc += f"\nReactor Type: CSTR\n"
        desc += f"Flow Rate: {self.flow_rate} m³/s\n"
        desc += f"Residence Time: {self.get_residence_time():.2f} s\n"
        
        if self.inlet_concentrations:
            desc += f"\nInlet Concentrations:\n"
            for species, conc in self.inlet_concentrations.items():
                desc += f"  {species}: {conc} mol/L\n"
        
        return desc


class PFR(Reactor):
    """
    Plug Flow Reactor with spatial discretization along reactor length.
    """
    
    def __init__(self, name: str = "PFR", volume: float = 1.0, length: float = 1.0,
                 flow_rate: float = 0.0, n_segments: int = 10, **kwargs):
        super().__init__(name, volume, **kwargs)
        self.reactor_type = "PFR"
        self.length = length  # Reactor length (m)
        self.flow_rate = flow_rate  # m³/s
        self.n_segments = n_segments  # Number of spatial discretization segments
        
        # Calculate cross-sectional area
        self.cross_sectional_area = volume / length
        
        # Spatial discretization
        self.segment_length = length / n_segments
        self.segment_volume = volume / n_segments
        
    def get_superficial_velocity(self) -> float:
        """Calculate superficial velocity (u = Q/A)."""
        return self.flow_rate / self.cross_sectional_area
    
    def get_residence_time(self) -> float:
        """Calculate residence time (τ = V/Q)."""
        if self.flow_rate == 0:
            return float('inf')
        return self.volume / self.flow_rate
    
    def build_ode_system(self):
        """
        Build ODE system for PFR with spatial discretization.
        This creates separate concentration variables for each segment.
        """
        if not self.reactions:
            raise ValueError("No reactions defined.")
        
        # Create species for each segment
        segmented_species = {}
        for segment in range(self.n_segments):
            for species_name in self.species.keys():
                seg_name = f"{species_name}_seg{segment}"
                segmented_species[seg_name] = species_name
        
        # Generate base reaction ODEs for each segment
        segment_odes = {}
        velocity = self.get_superficial_velocity()
        
        for segment in range(self.n_segments):
            # Create temporary species list for this segment
            temp_species = []
            for species_name in self.species.keys():
                seg_name = f"{species_name}_seg{segment}"
                temp_species.append(seg_name)
            
            # Generate reaction ODEs for this segment
            segment_reaction_odes = self.model_builder.generate_ode_system(self.reactions, temp_species)
            
            # Add convection terms
            for species_name in self.species.keys():
                seg_name = f"{species_name}_seg{segment}"
                
                # Start with reaction term
                segment_odes[seg_name] = segment_reaction_odes.get(seg_name, 0)
                
                # Add convection term: -u * dC/dz
                if segment == 0:
                    # First segment: inlet concentration
                    C_upstream = 0  # Could be inlet concentration
                else:
                    # Get concentration from upstream segment
                    upstream_seg = f"{species_name}_seg{segment-1}"
                    C_upstream = self.model_builder.concentration_symbols[upstream_seg]
                
                C_current = self.model_builder.concentration_symbols[seg_name]
                
                # Convection term: -u * (C_current - C_upstream) / Δz
                convection_term = -velocity * (C_current - C_upstream) / self.segment_length
                segment_odes[seg_name] += convection_term
        
        self.ode_system = segment_odes
        return segment_odes
    
    def describe(self) -> str:
        desc = super().describe()
        desc += f"\nReactor Type: PFR\n"
        desc += f"Length: {self.length} m\n"
        desc += f"Cross-sectional Area: {self.cross_sectional_area:.4f} m²\n"
        desc += f"Flow Rate: {self.flow_rate} m³/s\n"
        desc += f"Superficial Velocity: {self.get_superficial_velocity():.4f} m/s\n"
        desc += f"Residence Time: {self.get_residence_time():.2f} s\n"
        desc += f"Spatial Segments: {self.n_segments}\n"
        desc += f"Segment Length: {self.segment_length:.4f} m\n"
        
        return desc


class SemiBatchReactor(Reactor):
    """
    Semi-batch reactor with time-dependent feeding.
    """
    
    def __init__(self, name: str = "Semi-Batch Reactor", initial_volume: float = 1.0, **kwargs):
        super().__init__(name, initial_volume, **kwargs)
        self.reactor_type = "Semi-Batch"
        self.initial_volume = initial_volume
        self.feed_schedule = {}  # Dict: species_name -> (flow_rate, concentration, start_time, end_time)
        
    def add_feed(self, species_name: str, flow_rate: float, concentration: float, 
                 start_time: float = 0.0, end_time: float = float('inf')):
        """
        Add a feeding schedule for a species.
        
        Args:
            species_name: Name of species being fed
            flow_rate: Volumetric flow rate (m³/s)
            concentration: Feed concentration (mol/L)
            start_time: Start time for feeding (s)
            end_time: End time for feeding (s)
        """
        self.feed_schedule[species_name] = {
            'flow_rate': flow_rate,
            'concentration': concentration,
            'start_time': start_time,
            'end_time': end_time
        }
    
    def describe(self) -> str:
        desc = super().describe()
        desc += f"\nReactor Type: Semi-Batch\n"
        desc += f"Initial Volume: {self.initial_volume} m³\n"
        
        if self.feed_schedule:
            desc += f"\nFeed Schedule:\n"
            for species, feed_info in self.feed_schedule.items():
                desc += f"  {species}: {feed_info['concentration']} mol/L at {feed_info['flow_rate']} m³/s\n"
                desc += f"    Time: {feed_info['start_time']} - {feed_info['end_time']} s\n"
        
        return desc
