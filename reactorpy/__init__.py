"""
ReactorPy: A Python Library for Chemical Reactor Simulation and Analysis

ReactorPy provides a comprehensive toolkit for modeling, simulating, and analyzing
chemical reactors. It supports various reactor types including batch reactors,
CSTRs, PFRs, and semi-batch reactors with sophisticated kinetics modeling.

Key Features:
- Species and reaction management with thermodynamic properties
- Symbolic ODE generation from reaction networks
- Multiple reactor types (Batch, CSTR, PFR, Semi-batch)
- Numerical solving with scipy integration methods
- Comprehensive plotting and visualization
- Export capabilities (CSV, PDF, Excel, JSON)

Basic Usage:
    >>> from reactorpy import BatchReactor, Species, Reaction
    >>> 
    >>> # Create species
    >>> A = Species("A", concentration=2.0)
    >>> B = Species("B", concentration=1.0) 
    >>> C = Species("C", concentration=0.0)
    >>> 
    >>> # Create reaction
    >>> rxn = Reaction("A + B -> C")
    >>> rxn.set_rate_expression("k * [A] * [B]")
    >>> rxn.add_rate_parameter('k', 0.1)
    >>> 
    >>> # Set up reactor
    >>> reactor = BatchReactor("My Reactor", volume=2.0)
    >>> reactor.add_species([A, B, C])
    >>> reactor.add_reaction(rxn)
    >>> 
    >>> # Simulate
    >>> results = reactor.simulate((0, 50))
    >>> reactor.plot()

Author: ReactorPy Development Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Titus Loftin"
__email__ = "loftintitus@utexas.edu"
__license__ = "MIT"

# Core classes
from .species import Species
from .reaction import Reaction
from .model_builder import ModelBuilder, generate_symbolic_ODE
from .reactor import Reactor, BatchReactor, CSTR, PFR, SemiBatchReactor

# Solver functions
from .solver import (
    ODESolver,
    solve_ode_system,
    solve_steady_state,
    create_time_series,
    validate_solution
)

# Plotting functions
from .plotting import (
    plot_concentration_profiles,
    plot_reaction_rates,
    plot_phase_portrait,
    plot_conversion_selectivity,
    plot_reactor_comparison,
    plot_spatial_profile,
    plot_3d_surface,
    plot_multiple_species_subplots,
    save_all_plots
)

# I/O utilities
from .io_utilities import (
    export_to_csv,
    export_to_excel,
    export_to_pdf_report,
    export_results_json,
    batch_export
)

# Define what gets imported with "from reactorpy import *"
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    
    # Core classes
    'Species',
    'Reaction',
    'ModelBuilder',
    'Reactor',
    'BatchReactor',
    'CSTR',
    'PFR',
    'SemiBatchReactor',
    
    # Solver components
    'ODESolver',
    'solve_ode_system',
    'solve_steady_state',
    'create_time_series',
    'validate_solution',
    'generate_symbolic_ODE',  # Legacy function
    
    # Plotting functions
    'plot_concentration_profiles',
    'plot_reaction_rates',
    'plot_phase_portrait',
    'plot_conversion_selectivity',
    'plot_reactor_comparison',
    'plot_spatial_profile',
    'plot_3d_surface',
    'plot_multiple_species_subplots',
    'save_all_plots',
    
    # I/O utilities
    'export_to_csv',
    'export_to_excel',
    'export_to_pdf_report',
    'export_results_json',
    'batch_export',
]

# Package-level convenience functions
def create_batch_reactor(name="Batch Reactor", volume=1.0, **kwargs):
    """
    Convenience function to create a batch reactor.
    
    Args:
        name (str): Reactor name
        volume (float): Reactor volume in m³
        **kwargs: Additional reactor parameters
        
    Returns:
        BatchReactor: Configured batch reactor instance
    """
    return BatchReactor(name=name, volume=volume, **kwargs)

def create_cstr(name="CSTR", volume=1.0, flow_rate=0.0, **kwargs):
    """
    Convenience function to create a CSTR.
    
    Args:
        name (str): Reactor name
        volume (float): Reactor volume in m³
        flow_rate (float): Volumetric flow rate in m³/s
        **kwargs: Additional reactor parameters
        
    Returns:
        CSTR: Configured CSTR instance
    """
    return CSTR(name=name, volume=volume, flow_rate=flow_rate, **kwargs)

def create_pfr(name="PFR", volume=1.0, length=1.0, flow_rate=0.0, n_segments=10, **kwargs):
    """
    Convenience function to create a PFR.
    
    Args:
        name (str): Reactor name
        volume (float): Reactor volume in m³
        length (float): Reactor length in m
        flow_rate (float): Volumetric flow rate in m³/s
        n_segments (int): Number of spatial discretization segments
        **kwargs: Additional reactor parameters
        
    Returns:
        PFR: Configured PFR instance
    """
    return PFR(name=name, volume=volume, length=length, 
               flow_rate=flow_rate, n_segments=n_segments, **kwargs)

def quick_simulation(reaction_string, species_data, rate_params, time_span=(0, 100), 
                    reactor_type="batch", reactor_params=None, plot=True):
    """
    Quick simulation function for simple cases.
    
    Args:
        reaction_string (str): Reaction string like "A + B -> C"
        species_data (dict): Dict of species_name: initial_concentration
        rate_params (dict): Rate parameters like {'k': 0.1}
        time_span (tuple): Time span for simulation
        reactor_type (str): 'batch', 'cstr', or 'pfr'
        reactor_params (dict): Additional reactor parameters
        plot (bool): Whether to plot results
        
    Returns:
        dict: Simulation results
        
    Example:
        >>> results = quick_simulation(
        ...     "A + B -> C",
        ...     {"A": 2.0, "B": 1.0, "C": 0.0},
        ...     {"k": 0.1},
        ...     time_span=(0, 50),
        ...     plot=True
        ... )
    """
    reactor_params = reactor_params or {}
    
    # Create species
    species_list = []
    for name, conc in species_data.items():
        species = Species(name, concentration=conc)
        species_list.append(species)
    
    # Create reaction
    reaction = Reaction(reaction_string)
    
    # Infer rate expression (simple case)
    reactants = list(reaction.reactants.keys())
    if len(reactants) == 1:
        rate_expr = f"k * [{reactants[0]}]"
    elif len(reactants) == 2:
        rate_expr = f"k * [{reactants[0]}] * [{reactants[1]}]"
    else:
        rate_expr = "k * " + " * ".join([f"[{r}]" for r in reactants])
    
    reaction.set_rate_expression(rate_expr)
    for param, value in rate_params.items():
        reaction.add_rate_parameter(param, value)
    
    # Create reactor
    if reactor_type.lower() == "batch":
        reactor = BatchReactor(**reactor_params)
    elif reactor_type.lower() == "cstr":
        reactor = CSTR(**reactor_params)
    elif reactor_type.lower() == "pfr":
        reactor = PFR(**reactor_params)
    else:
        raise ValueError(f"Unknown reactor type: {reactor_type}")
    
    # Set up simulation
    reactor.add_species(species_list)
    reactor.add_reaction(reaction)
    
    # Run simulation
    results = reactor.simulate(time_span)
    
    # Plot if requested
    if plot:
        reactor.plot()
    
    return results

# Add convenience functions to __all__
__all__.extend([
    'create_batch_reactor',
    'create_cstr', 
    'create_pfr',
    'quick_simulation'
])

# Module-level configuration
def set_default_plot_style(style='seaborn'):
    """
    Set the default plotting style for all ReactorPy plots.
    
    Args:
        style (str): Matplotlib style name
    """
    import matplotlib.pyplot as plt
    plt.style.use(style)

def get_version_info():
    """
    Get detailed version and dependency information.
    
    Returns:
        dict: Version information
    """
    import sys
    
    version_info = {
        'reactorpy': __version__,
        'python': sys.version,
        'platform': sys.platform
    }
    
    # Check for optional dependencies
    try:
        import numpy
        version_info['numpy'] = numpy.__version__
    except ImportError:
        version_info['numpy'] = 'Not installed'
    
    try:
        import scipy
        version_info['scipy'] = scipy.__version__
    except ImportError:
        version_info['scipy'] = 'Not installed'
    
    try:
        import sympy
        version_info['sympy'] = sympy.__version__
    except ImportError:
        version_info['sympy'] = 'Not installed'
    
    try:
        import matplotlib
        version_info['matplotlib'] = matplotlib.__version__
    except ImportError:
        version_info['matplotlib'] = 'Not installed'
    
    try:
        import pandas
        version_info['pandas'] = pandas.__version__
    except ImportError:
        version_info['pandas'] = 'Not installed'
    
    return version_info

def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are available
    """
    required_packages = ['numpy', 'scipy', 'sympy', 'matplotlib']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("All required dependencies are installed!")
    return True

# Add utility functions to __all__
__all__.extend([
    'set_default_plot_style',
    'get_version_info',
    'check_dependencies'
])

# Package initialization message
def _show_welcome_message():
    """Show welcome message when package is imported."""
    import sys
    if hasattr(sys, 'ps1'):  # Only show in interactive mode
        print(f"ReactorPy v{__version__} loaded successfully!")
        print("Quick start: reactorpy.check_dependencies()")
        print("Documentation: help(reactorpy)")

# Show welcome message
try:
    _show_welcome_message()
except:
    pass  # Silently fail if there are issues

