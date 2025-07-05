import re

class Reaction(object):
    """
    A class representing a chemical reaction.
    
    Attributes:
        reactants (dict): Dictionary of reactant names and their stoichiometric coefficients.
        products (dict): Dictionary of product names and their stoichiometric coefficients.
        rate_expression (str): String representation of the rate law (e.g., "k * [A] * [B]").
        rate_parameters (dict): Dictionary of rate parameters (e.g., {'k': 0.1, 'Ea': 50000}).
        reaction_string (str): Original reaction string for reference.
        is_reversible (bool): Whether the reaction is reversible.
        equilibrium_constant (float): Equilibrium constant (if reversible).
    """

    def __init__(self, reaction_string=None, reactants=None, products=None, 
                 rate_expression=None, rate_parameters=None, is_reversible=False, 
                 equilibrium_constant=None):
        """
        Initialize a reaction either from a reaction string or explicit reactants/products.
        
        Args:
            reaction_string (str): Reaction string like "2A + B -> C + 3D" or "A + B <-> C"
            reactants (dict): Dict of reactant names and stoichiometric coefficients
            products (dict): Dict of product names and stoichiometric coefficients  
            rate_expression (str): Rate law expression as string
            rate_parameters (dict): Rate parameters like rate constants, activation energies
            is_reversible (bool): Whether reaction is reversible
            equilibrium_constant (float): Equilibrium constant for reversible reactions
        """
        if reaction_string:
            self.reaction_string = reaction_string
            self.reactants, self.products, self.is_reversible = self._parse_reaction_string(reaction_string)
        else:
            self.reactants = reactants or {}
            self.products = products or {}
            self.is_reversible = is_reversible
            self.reaction_string = self._generate_reaction_string()
        
        self.rate_expression = rate_expression
        self.rate_parameters = rate_parameters or {}
        self.equilibrium_constant = equilibrium_constant
        
        # Backward compatibility
        if 'k' in self.rate_parameters:
            self.rate_constant = self.rate_parameters['k']
        else:
            self.rate_constant = None

    def _parse_reaction_string(self, reaction_string):
        """
        Parse a reaction string into reactants and products dictionaries.
        
        Args:
            reaction_string (str): String like "2A + B -> C + 3D" or "A + B <-> C"
            
        Returns:
            tuple: (reactants_dict, products_dict, is_reversible)
        """
        # Clean up the string
        reaction_string = reaction_string.strip()
        
        # Check if reversible
        if '<->' in reaction_string or '⇌' in reaction_string:
            is_reversible = True
            arrow = '<->' if '<->' in reaction_string else '⇌'
        elif '->' in reaction_string or '→' in reaction_string:
            is_reversible = False
            arrow = '->' if '->' in reaction_string else '→'
        else:
            raise ValueError(f"Invalid reaction string: {reaction_string}. Must contain '->' or '<->'")
        
        # Split into reactants and products
        parts = reaction_string.split(arrow)
        if len(parts) != 2:
            raise ValueError(f"Invalid reaction string: {reaction_string}")
        
        reactants_str, products_str = parts
        
        reactants = self._parse_species_string(reactants_str.strip())
        products = self._parse_species_string(products_str.strip())
        
        return reactants, products, is_reversible

    def _parse_species_string(self, species_string):
        """
        Parse a string of species into a dictionary with stoichiometric coefficients.
        
        Args:
            species_string (str): String like "2A + B + 3C"
            
        Returns:
            dict: Dictionary like {'A': 2, 'B': 1, 'C': 3}
        """
        species_dict = {}
        
        # Split by + and process each term
        terms = [term.strip() for term in species_string.split('+')]
        
        for term in terms:
            if not term:
                continue
                
            # Use regex to extract coefficient and species name
            match = re.match(r'^(\d*\.?\d*)\s*([A-Za-z][A-Za-z0-9_]*)$', term)
            
            if match:
                coeff_str, species = match.groups()
                coeff = float(coeff_str) if coeff_str else 1.0
                species_dict[species] = coeff
            else:
                raise ValueError(f"Invalid species term: {term}")
        
        return species_dict

    def _generate_reaction_string(self):
        """Generate a reaction string from reactants and products."""
        def format_species(species_dict):
            terms = []
            for species, coeff in species_dict.items():
                if coeff == 1:
                    terms.append(species)
                else:
                    terms.append(f"{coeff}{species}")
            return " + ".join(terms)
        
        reactants_str = format_species(self.reactants)
        products_str = format_species(self.products)
        arrow = " <-> " if self.is_reversible else " -> "
        
        return reactants_str + arrow + products_str

    def get_all_species(self):
        """Get a set of all species involved in the reaction."""
        all_species = set(self.reactants.keys()) | set(self.products.keys())
        return all_species

    def get_stoichiometric_coefficient(self, species):
        """
        Get the stoichiometric coefficient for a species.
        
        Args:
            species (str): Species name
            
        Returns:
            float: Stoichiometric coefficient (negative for reactants, positive for products)
        """
        if species in self.reactants:
            return -self.reactants[species]
        elif species in self.products:
            return self.products[species]
        else:
            return 0.0

    def set_rate_expression(self, expression):
        """
        Set the rate expression for the reaction.
        
        Args:
            expression (str): Rate expression like "k * [A] * [B]" or "k * [A]^2 * [B]"
        """
        self.rate_expression = expression

    def add_rate_parameter(self, name, value):
        """
        Add or update a rate parameter.
        
        Args:
            name (str): Parameter name (e.g., 'k', 'Ea', 'A')
            value (float): Parameter value
        """
        self.rate_parameters[name] = value
        
        # Maintain backward compatibility
        if name == 'k':
            self.rate_constant = value

    def get_rate_parameter(self, name):
        """
        Get a rate parameter value.
        
        Args:
            name (str): Parameter name
            
        Returns:
            float: Parameter value or None if not found
        """
        return self.rate_parameters.get(name)

    def is_balanced(self):
        """
        Check if the reaction is mass balanced (same atoms on both sides).
        Note: This is a simple check based on species names only.
        """
        # Simple check: same number of each species type
        # For more sophisticated checking, would need molecular formulas
        reactant_species = set(self.reactants.keys())
        product_species = set(self.products.keys())
        
        # This is a placeholder - real mass balance would require molecular formulas
        return len(reactant_species) > 0 and len(product_species) > 0

    def __repr__(self):
        return f"Reaction('{self.reaction_string}')"

    def __str__(self):
        result = f"Reaction: {self.reaction_string}\n"
        if self.rate_expression:
            result += f"Rate law: {self.rate_expression}\n"
        if self.rate_parameters:
            result += f"Parameters: {self.rate_parameters}\n"
        if self.is_reversible and self.equilibrium_constant:
            result += f"Keq: {self.equilibrium_constant}\n"
        return result.strip()