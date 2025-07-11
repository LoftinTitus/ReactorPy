import sympy as sp
import re
from typing import List, Dict, Union

class ModelBuilder:
    """
    A class for building symbolic ODE systems from reactions and species.
    """
    
    def __init__(self):
        self.concentration_symbols = {}  # Maps species name to [species] symbol
        self.parameter_symbols = {}     # Maps parameter name to symbol
        self.time_symbol = sp.Symbol('t')
        
    def create_concentration_symbol(self, species_name):
        """
        Create a SymPy symbol for species concentration [species_name].
        
        Args:
            species_name (str): Name of the species
            
        Returns:
            sympy.Symbol: Symbol representing [species_name]
        """
        if species_name not in self.concentration_symbols:
            # Create symbol with brackets for clarity: [A], [B], etc.
            symbol_name = f"[{species_name}]"
            self.concentration_symbols[species_name] = sp.Symbol(symbol_name, real=True, positive=True)
        
        return self.concentration_symbols[species_name]
    
    def create_parameter_symbol(self, param_name):
        """
        Create a SymPy symbol for a rate parameter.
        
        Args:
            param_name (str): Name of the parameter (e.g., 'k', 'k1', 'Ea')
            
        Returns:
            sympy.Symbol: Symbol representing the parameter
        """
        if param_name not in self.parameter_symbols:
            self.parameter_symbols[param_name] = sp.Symbol(param_name, real=True, positive=True)
        
        return self.parameter_symbols[param_name]
    
    def parse_rate_expression(self, rate_expression, rate_parameters=None):
        """
        Parse a rate expression string into a SymPy expression.

        Args:
            rate_expression (str): Rate expression like "k * [A] * [B]" or "k1 * [A]^2 - k2 * [C]"
            rate_parameters (dict): Dictionary of rate parameters

        Returns:
            sympy.Expr: Symbolic rate expression
        """
        if not rate_expression:
            return sp.sympify(0)

        # Create a local dictionary for sympify to use
        local_dict = {}

        # Find all concentration terms [species] and create symbols for them
        concentration_pattern = r'\[([A-Za-z][A-Za-z0-9_]*)\]'
        species_matches = re.findall(concentration_pattern, rate_expression)

        for species in species_matches:
            concentration_symbol = self.create_concentration_symbol(species)
            # Use a simple variable name for parsing, without brackets
            local_dict[f'conc_{species}'] = concentration_symbol

        # Find all parameter names and create symbols for them
        if rate_parameters:
            for param_name in rate_parameters.keys():
                if param_name in rate_expression:
                    param_symbol = self.create_parameter_symbol(param_name)
                    local_dict[param_name] = param_symbol

        # Replace [species] with simple variable names for parsing
        expr_str = rate_expression
        for species in species_matches:
            expr_str = expr_str.replace(f'[{species}]', f'conc_{species}')

        # Convert ^ to ** for SymPy
        expr_str = expr_str.replace('^', '**')

        try:
            symbolic_expr = sp.sympify(expr_str, locals=local_dict)
            return symbolic_expr
        except Exception as e:
            raise ValueError(f"Could not parse rate expression '{rate_expression}': {e}")
    
    def generate_single_reaction_odes(self, reaction):
        """
        Generate symbolic ODEs for a single reaction.
        
        Args:
            reaction (Reaction): The reaction object
            
        Returns:
            dict: Dictionary mapping species names to their rate contributions
        """
        # Parse the rate expression
        rate_expr = self.parse_rate_expression(reaction.rate_expression, reaction.rate_parameters)
        
        ode_contributions = {}
        
        # Handle reactants (negative contribution)
        for species_name, stoich_coeff in reaction.reactants.items():
            if species_name not in ode_contributions:
                ode_contributions[species_name] = 0
            ode_contributions[species_name] += -stoich_coeff * rate_expr
        
        # Handle products (positive contribution)
        for species_name, stoich_coeff in reaction.products.items():
            if species_name not in ode_contributions:
                ode_contributions[species_name] = 0
            ode_contributions[species_name] += stoich_coeff * rate_expr
        
        return ode_contributions
    
    def generate_ode_system(self, reactions, species_list=None):
        """
        Generate a complete symbolic ODE system from multiple reactions.
        
        Args:
            reactions (list): List of Reaction objects
            species_list (list, optional): List of species to include. If None, includes all species from reactions.
            
        Returns:
            dict: Dictionary mapping species names to their complete ODEs (sum of all reaction contributions)
        """
        # Collect all species if not provided
        if species_list is None:
            all_species = set()
            for reaction in reactions:
                all_species.update(reaction.get_all_species())
            species_list = list(all_species)
        
        # Initialize ODE system
        ode_system = {species: sp.sympify(0) for species in species_list}
        
        # Add contributions from each reaction
        for reaction in reactions:
            reaction_odes = self.generate_single_reaction_odes(reaction)
            
            for species, contribution in reaction_odes.items():
                if species in ode_system:
                    ode_system[species] += contribution
        
        return ode_system
    
    def generate_ode_system_with_species_objects(self, reactions, species_objects):
        """
        Generate ODEs using Species objects for additional constraints/properties.
        
        Args:
            reactions (list): List of Reaction objects
            species_objects (list): List of Species objects
            
        Returns:
            dict: Dictionary with species names mapped to their ODEs
        """
        # Extract species names from Species objects
        species_names = [species.name for species in species_objects]
        
        # Generate basic ODE system
        ode_system = self.generate_ode_system(reactions, species_names)
        
        # Here you could add additional constraints based on Species properties
        # For example, conservation laws, phase equilibria, etc.
        
        return ode_system
    
    def get_system_matrix_form(self, ode_system):
        """
        Convert ODE system to matrix form suitable for numerical solvers.
        
        Args:
            ode_system (dict): Dictionary of species -> ODE expressions
            
        Returns:
            tuple: (concentration_vector, ode_vector, parameter_substitutions)
        """
        species_names = list(ode_system.keys())
        
        # Create concentration vector
        concentration_vector = [self.concentration_symbols[name] for name in species_names]
        
        # Create ODE vector
        ode_vector = [ode_system[name] for name in species_names]
        
        # Get all parameters used
        all_symbols = set()
        for ode in ode_vector:
            all_symbols.update(ode.free_symbols)
        
        # Separate concentration symbols from parameter symbols
        param_symbols = all_symbols - set(concentration_vector)
        
        return concentration_vector, ode_vector, list(param_symbols)
    
    def substitute_numerical_values(self, ode_system, species_objects=None, numerical_params=None):
        """
        Substitute numerical values for parameters and initial conditions.
        
        Args:
            ode_system (dict): Symbolic ODE system
            species_objects (list): List of Species objects with concentration values
            numerical_params (dict): Dictionary of parameter values
            
        Returns:
            dict: ODE system with numerical substitutions
        """
        substitutions = {}
        
        # Add parameter substitutions
        if numerical_params:
            for param_name, value in numerical_params.items():
                if param_name in self.parameter_symbols:
                    substitutions[self.parameter_symbols[param_name]] = value
        
        # Add species concentration substitutions if needed
        if species_objects:
            for species in species_objects:
                if species.concentration is not None:
                    symbol = self.concentration_symbols.get(species.name)
                    if symbol:
                        substitutions[symbol] = species.concentration
        
        # Apply substitutions
        numerical_ode_system = {}
        for species, ode in ode_system.items():
            numerical_ode_system[species] = ode.subs(substitutions)
        
        return numerical_ode_system

def generate_symbolic_ODE(reaction, species):
    """
    Legacy function for backward compatibility.
    Generate symbolic ODEs for a given reaction and species.

    Args:
        reaction (Reaction): The reaction object containing reactants, products, and rate expression.
        species (list): List of species involved in the reaction.

    Returns:
        dict: A dictionary mapping each species to its corresponding ODE.
    """
    builder = ModelBuilder()
    
    # Convert species list to species names if they're Species objects
    if hasattr(species[0], 'name'):
        species_names = [s.name for s in species]
    else:
        species_names = species
    
    return builder.generate_ode_system([reaction], species_names)