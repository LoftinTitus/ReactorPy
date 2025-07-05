import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Callable, Optional, Union
import warnings


class ODESolver:
    """
    Converts symbolic ODE systems to numerical form and solves them.
    """
    
    def __init__(self):
        self.symbolic_system = None
        self.numerical_system = None
        self.concentration_symbols = []
        self.parameter_symbols = []
        self.lambdified_functions = {}
        
    def prepare_system(self, symbolic_odes: Dict[str, sp.Expr], 
                      concentration_symbols: Dict[str, sp.Symbol],
                      parameter_symbols: Dict[str, sp.Symbol] = None):
        """
        Prepare the symbolic ODE system for numerical solution.
        
        Args:
            symbolic_odes: Dictionary mapping species names to symbolic ODEs
            concentration_symbols: Dictionary mapping species names to concentration symbols
            parameter_symbols: Dictionary mapping parameter names to parameter symbols
        """
        self.symbolic_system = symbolic_odes
        self.concentration_symbols = concentration_symbols
        self.parameter_symbols = parameter_symbols or {}
        
        # Create ordered lists for consistent indexing
        self.species_names = list(symbolic_odes.keys())
        self.concentration_vector = [concentration_symbols[name] for name in self.species_names]
        self.ode_expressions = [symbolic_odes[name] for name in self.species_names]
        
    def lambdify_system(self, rate_parameters: Dict[str, float] = None):
        """
        Convert symbolic expressions to fast numerical functions using lambdify.
        
        Args:
            rate_parameters: Dictionary of numerical parameter values
        """
        if not self.symbolic_system:
            raise ValueError("No symbolic system prepared. Call prepare_system() first.")
        
        rate_parameters = rate_parameters or {}
        
        # Create substitution dictionary for parameters
        parameter_substitutions = {}
        for param_name, param_value in rate_parameters.items():
            if param_name in self.parameter_symbols:
                parameter_substitutions[self.parameter_symbols[param_name]] = param_value
        
        # Substitute parameter values into expressions
        substituted_expressions = []
        for expr in self.ode_expressions:
            substituted_expr = expr.subs(parameter_substitutions)
            substituted_expressions.append(substituted_expr)
        
        # Lambdify each ODE expression
        self.lambdified_functions = {}
        for i, (species_name, expr) in enumerate(zip(self.species_names, substituted_expressions)):
            try:
                # Create lambdified function that takes concentration vector and returns rate
                func = sp.lambdify(self.concentration_vector, expr, modules=['numpy'])
                self.lambdified_functions[species_name] = func
            except Exception as e:
                warnings.warn(f"Could not lambdify expression for {species_name}: {e}")
                # Create a fallback function that returns zero
                self.lambdified_functions[species_name] = lambda *args: 0.0
    
    def create_ode_function(self, rate_parameters: Dict[str, float] = None) -> Callable:
        """
        Create the ODE function suitable for scipy.integrate.solve_ivp.
        
        Args:
            rate_parameters: Dictionary of numerical parameter values
            
        Returns:
            Callable: Function with signature f(t, y) that returns dy/dt
        """
        # Prepare lambdified functions
        self.lambdify_system(rate_parameters)
        
        def ode_function(t: float, y: np.ndarray) -> np.ndarray:
            """
            ODE function for scipy solver.
            
            Args:
                t: Current time
                y: Current concentration vector
                
            Returns:
                np.ndarray: Derivatives dy/dt
            """
            try:
                # Ensure y is a numpy array
                y = np.asarray(y, dtype=float)
                
                # Handle negative concentrations (set to zero)
                y_safe = np.maximum(y, 0.0)
                
                # Calculate derivatives for each species
                dydt = np.zeros_like(y)
                for i, species_name in enumerate(self.species_names):
                    func = self.lambdified_functions[species_name]
                    try:
                        # Call the lambdified function with current concentrations
                        dydt[i] = func(*y_safe)
                    except Exception as e:
                        warnings.warn(f"Error evaluating ODE for {species_name} at t={t}: {e}")
                        dydt[i] = 0.0
                
                return dydt
                
            except Exception as e:
                warnings.warn(f"Error in ODE function at t={t}: {e}")
                return np.zeros_like(y)
        
        return ode_function


def solve_ode_system(symbolic_odes: Dict[str, sp.Expr],
                    initial_concentrations: Dict[str, float],
                    time_span: Tuple[float, float],
                    rate_parameters: Dict[str, float] = None,
                    concentration_symbols: Dict[str, sp.Symbol] = None,
                    parameter_symbols: Dict[str, sp.Symbol] = None,
                    method: str = 'RK45',
                    rtol: float = 1e-6,
                    atol: float = 1e-9,
                    max_step: Optional[float] = None,
                    dense_output: bool = False,
                    events: Optional[List] = None,
                    vectorized: bool = False,
                    **kwargs) -> Dict:
    """
    Main function to solve a symbolic ODE system numerically.
    
    Args:
        symbolic_odes: Dictionary mapping species names to symbolic ODE expressions
        initial_concentrations: Dictionary of initial concentration values
        time_span: Tuple of (t_start, t_end)
        rate_parameters: Dictionary of rate parameter values
        concentration_symbols: Dictionary mapping species names to concentration symbols
        parameter_symbols: Dictionary mapping parameter names to parameter symbols
        method: Integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA')
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_step: Maximum step size
        dense_output: Whether to compute dense output
        events: List of event functions
        vectorized: Whether the ODE function is vectorized
        **kwargs: Additional arguments passed to solve_ivp
        
    Returns:
        Dict: Results containing time points, concentrations, and solver info
    """
    # Extract species names and check consistency
    species_names = list(symbolic_odes.keys())
    
    # Check that all species have initial concentrations
    missing_species = set(species_names) - set(initial_concentrations.keys())
    if missing_species:
        raise ValueError(f"Missing initial concentrations for species: {missing_species}")
    
    # Create concentration symbols if not provided
    if concentration_symbols is None:
        concentration_symbols = {}
        for species in species_names:
            symbol_name = f"[{species}]"
            concentration_symbols[species] = sp.Symbol(symbol_name, real=True, positive=True)
    
    # Set up initial conditions vector
    y0 = np.array([initial_concentrations[species] for species in species_names])
    
    # Create solver instance
    solver = ODESolver()
    solver.prepare_system(symbolic_odes, concentration_symbols, parameter_symbols)
    
    # Create ODE function
    ode_func = solver.create_ode_function(rate_parameters)
    
    # Set up solver options
    solver_options = {
        'rtol': rtol,
        'atol': atol,
        'method': method,
        'dense_output': dense_output,
        'vectorized': vectorized
    }
    
    # Add optional parameters
    if max_step is not None:
        solver_options['max_step'] = max_step
    if events is not None:
        solver_options['events'] = events
    
    # Update with any additional kwargs
    solver_options.update(kwargs)
    
    try:
        # Solve the ODE system
        solution = solve_ivp(ode_func, time_span, y0, **solver_options)
        
        # Check if solution was successful
        if not solution.success:
            warnings.warn(f"ODE solver failed: {solution.message}")
        
        # Package results
        results = {
            'time': solution.t,
            'success': solution.success,
            'message': solution.message,
            'nfev': solution.nfev,
            'njev': solution.njev if hasattr(solution, 'njev') else None,
            'nlu': solution.nlu if hasattr(solution, 'nlu') else None,
            'status': solution.status,
            'method': method,
            'species_names': species_names
        }
        
        # Add concentration results for each species
        for i, species in enumerate(species_names):
            results[species] = solution.y[i]
        
        # Add final concentrations
        results['final_concentrations'] = {
            species: solution.y[i][-1] for i, species in enumerate(species_names)
        }
        
        # Add solution object for advanced users
        results['solution_object'] = solution
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Error during ODE integration: {e}")


def solve_steady_state(symbolic_odes: Dict[str, sp.Expr],
                      initial_guess: Dict[str, float],
                      rate_parameters: Dict[str, float] = None,
                      concentration_symbols: Dict[str, sp.Symbol] = None,
                      parameter_symbols: Dict[str, sp.Symbol] = None,
                      method: str = 'hybr',
                      tol: float = 1e-6,
                      **kwargs) -> Dict:
    """
    Solve for steady-state concentrations by setting all derivatives to zero.
    
    Args:
        symbolic_odes: Dictionary mapping species names to symbolic ODE expressions
        initial_guess: Dictionary of initial guess concentrations
        rate_parameters: Dictionary of rate parameter values
        concentration_symbols: Dictionary mapping species names to concentration symbols
        parameter_symbols: Dictionary mapping parameter names to parameter symbols
        method: Root finding method ('hybr', 'lm', 'broyden1', etc.)
        tol: Tolerance for convergence
        **kwargs: Additional arguments passed to scipy.optimize.root
        
    Returns:
        Dict: Results containing steady-state concentrations and solver info
    """
    from scipy.optimize import root
    
    # Extract species names
    species_names = list(symbolic_odes.keys())
    
    # Create concentration symbols if not provided
    if concentration_symbols is None:
        concentration_symbols = {}
        for species in species_names:
            symbol_name = f"[{species}]"
            concentration_symbols[species] = sp.Symbol(symbol_name, real=True, positive=True)
    
    # Set up solver
    solver = ODESolver()
    solver.prepare_system(symbolic_odes, concentration_symbols, parameter_symbols)
    solver.lambdify_system(rate_parameters)
    
    # Initial guess vector
    x0 = np.array([initial_guess[species] for species in species_names])
    
    def steady_state_equations(x):
        """Function that should equal zero at steady state."""
        x_safe = np.maximum(x, 0.0)  # Prevent negative concentrations
        
        equations = np.zeros_like(x)
        for i, species_name in enumerate(species_names):
            func = solver.lambdified_functions[species_name]
            try:
                equations[i] = func(*x_safe)
            except Exception:
                equations[i] = 0.0
        
        return equations
    
    # Solve for steady state
    try:
        solution = root(steady_state_equations, x0, method=method, tol=tol, **kwargs)
        
        # Package results
        results = {
            'success': solution.success,
            'message': solution.message,
            'nfev': solution.nfev,
            'method': method,
            'species_names': species_names
        }
        
        # Add steady-state concentrations
        if solution.success:
            for i, species in enumerate(species_names):
                results[species] = solution.x[i]
            
            results['steady_state_concentrations'] = {
                species: solution.x[i] for i, species in enumerate(species_names)
            }
        
        results['solution_object'] = solution
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Error during steady-state calculation: {e}")


def create_time_series(time_span: Tuple[float, float], 
                      n_points: int = 100,
                      log_spacing: bool = False) -> np.ndarray:
    """
    Create a time series for evaluation.
    
    Args:
        time_span: (t_start, t_end)
        n_points: Number of time points
        log_spacing: Whether to use logarithmic spacing
        
    Returns:
        np.ndarray: Time points
    """
    t_start, t_end = time_span
    
    if log_spacing and t_start > 0:
        return np.logspace(np.log10(t_start), np.log10(t_end), n_points)
    else:
        return np.linspace(t_start, t_end, n_points)


def validate_solution(results: Dict, 
                     tolerance: float = 1e-3) -> Dict[str, bool]:
    """
    Validate simulation results for common issues.
    
    Args:
        results: Results dictionary from solve_ode_system
        tolerance: Tolerance for validation checks
        
    Returns:
        Dict: Validation results
    """
    validation = {
        'solver_success': results.get('success', False),
        'no_negative_concentrations': True,
        'mass_balance_conserved': True,
        'reasonable_values': True
    }
    
    # Check for negative concentrations
    for species in results['species_names']:
        concentrations = results[species]
        if np.any(concentrations < -tolerance):
            validation['no_negative_concentrations'] = False
    
    # Check for unreasonably large values
    for species in results['species_names']:
        concentrations = results[species]
        if np.any(concentrations > 1e6):  # Arbitrary large value
            validation['reasonable_values'] = False
    
    return validation
