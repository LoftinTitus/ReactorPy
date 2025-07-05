import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


def plot_concentration_profiles(results: Dict, 
                              species_names: Optional[List[str]] = None,
                              title: str = "Concentration Profiles",
                              xlabel: str = "Time (s)",
                              ylabel: str = "Concentration (mol/L)",
                              figsize: Tuple[float, float] = (10, 6),
                              style: str = 'default',
                              colors: Optional[List[str]] = None,
                              linestyles: Optional[List[str]] = None,
                              save_path: Optional[str] = None,
                              show_grid: bool = True,
                              legend_loc: str = 'best',
                              **kwargs) -> plt.Figure:
    """
    Plot concentration profiles over time for specified species.
    
    Args:
        results: Results dictionary from solver containing time and concentration data
        species_names: List of species to plot. If None, plots all species
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        style: Matplotlib style ('default', 'seaborn', 'ggplot', etc.)
        colors: List of colors for each species
        linestyles: List of line styles for each species
        save_path: Path to save the figure
        show_grid: Whether to show grid
        legend_loc: Legend location
        **kwargs: Additional arguments passed to plt.plot()
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Set style
    if style != 'default':
        plt.style.use(style)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get time data
    if 'time' not in results:
        raise ValueError("Results must contain 'time' data")
    time = results['time']
    
    # Determine species to plot
    if species_names is None:
        species_names = results.get('species_names', [])
        if not species_names:
            # Try to infer from results keys
            species_names = [key for key in results.keys() 
                           if key not in ['time', 'success', 'message', 'nfev', 'njev', 'nlu', 
                                        'status', 'method', 'final_concentrations', 'solution_object']]
    
    # Set up colors and line styles
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(species_names)))
    if linestyles is None:
        linestyles = ['-'] * len(species_names)
    
    # Plot each species
    for i, species in enumerate(species_names):
        if species in results:
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            ax.plot(time, results[species], 
                   label=f'[{species}]', 
                   color=color, 
                   linestyle=linestyle,
                   **kwargs)
        else:
            warnings.warn(f"Species '{species}' not found in results")
    
    # Customize plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    if len(species_names) > 0:
        ax.legend(loc=legend_loc)
    
    # Make layout tight
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_reaction_rates(results: Dict,
                       reactions: List,
                       species_names: Optional[List[str]] = None,
                       title: str = "Reaction Rates",
                       xlabel: str = "Time (s)", 
                       ylabel: str = "Reaction Rate (mol/L·s)",
                       figsize: Tuple[float, float] = (10, 6),
                       **kwargs) -> plt.Figure:
    """
    Plot reaction rates over time.
    
    Args:
        results: Results dictionary from solver
        reactions: List of Reaction objects
        species_names: Species involved in reactions
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    time = results['time']
    
    # Calculate reaction rates for each reaction
    for i, reaction in enumerate(reactions):
        # This is a simplified calculation - in practice you'd need to evaluate
        # the rate expression at each time point
        reaction_label = f"R{i+1}: {reaction.reaction_string}"
        
        # Placeholder - would need to implement rate calculation
        # For now, just show the concept
        ax.plot(time, np.zeros_like(time), label=reaction_label, **kwargs)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_phase_portrait(results: Dict,
                       species_x: str,
                       species_y: str,
                       title: Optional[str] = None,
                       xlabel: Optional[str] = None,
                       ylabel: Optional[str] = None,
                       figsize: Tuple[float, float] = (8, 8),
                       show_trajectory: bool = True,
                       show_start_end: bool = True,
                       **kwargs) -> plt.Figure:
    """
    Create a phase portrait plot for two species.
    
    Args:
        results: Results dictionary from solver
        species_x: Species name for x-axis
        species_y: Species name for y-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        show_trajectory: Whether to show trajectory arrows
        show_start_end: Whether to mark start and end points
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if species_x not in results or species_y not in results:
        raise ValueError(f"Species {species_x} or {species_y} not found in results")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x_data = results[species_x]
    y_data = results[species_y]
    
    # Plot trajectory
    ax.plot(x_data, y_data, **kwargs)
    
    # Add trajectory arrows
    if show_trajectory:
        n_arrows = 10
        step = len(x_data) // n_arrows
        for i in range(0, len(x_data) - step, step):
            dx = x_data[i + step] - x_data[i]
            dy = y_data[i + step] - y_data[i]
            ax.arrow(x_data[i], y_data[i], dx * 0.3, dy * 0.3,
                    head_width=max(x_data) * 0.01, head_length=max(x_data) * 0.01,
                    fc='red', ec='red', alpha=0.7)
    
    # Mark start and end points
    if show_start_end:
        ax.plot(x_data[0], y_data[0], 'go', markersize=8, label='Start')
        ax.plot(x_data[-1], y_data[-1], 'ro', markersize=8, label='End')
        ax.legend()
    
    # Labels and title
    xlabel = xlabel or f'[{species_x}] (mol/L)'
    ylabel = ylabel or f'[{species_y}] (mol/L)'
    title = title or f'Phase Portrait: {species_x} vs {species_y}'
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_conversion_selectivity(results: Dict,
                              reactant: str,
                              products: List[str],
                              title: str = "Conversion and Selectivity",
                              figsize: Tuple[float, float] = (12, 5),
                              **kwargs) -> plt.Figure:
    """
    Plot conversion of reactant and selectivity to products.
    
    Args:
        results: Results dictionary from solver
        reactant: Name of the reactant species
        products: List of product species names
        title: Plot title
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if reactant not in results:
        raise ValueError(f"Reactant '{reactant}' not found in results")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    time = results['time']
    reactant_conc = results[reactant]
    initial_reactant = reactant_conc[0]
    
    # Calculate conversion
    conversion = (initial_reactant - reactant_conc) / initial_reactant * 100
    
    # Plot conversion
    ax1.plot(time, conversion, 'b-', linewidth=2, label=f'{reactant} Conversion')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Conversion (%)')
    ax1.set_title('Reactant Conversion')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Calculate and plot selectivity
    for product in products:
        if product in results:
            product_conc = results[product]
            total_products = sum(results[p] for p in products if p in results)
            selectivity = np.divide(product_conc, total_products, 
                                  out=np.zeros_like(product_conc), 
                                  where=total_products != 0) * 100
            ax2.plot(time, selectivity, label=f'{product} Selectivity', **kwargs)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Selectivity (%)')
    ax2.set_title('Product Selectivity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_reactor_comparison(results_list: List[Dict],
                          reactor_names: List[str],
                          species_name: str,
                          title: Optional[str] = None,
                          xlabel: str = "Time (s)",
                          ylabel: str = "Concentration (mol/L)",
                          figsize: Tuple[float, float] = (10, 6),
                          **kwargs) -> plt.Figure:
    """
    Compare results from different reactors for a single species.
    
    Args:
        results_list: List of results dictionaries
        reactor_names: List of reactor names for legend
        species_name: Name of species to compare
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (results, reactor_name) in enumerate(zip(results_list, reactor_names)):
        if 'time' in results and species_name in results:
            ax.plot(results['time'], results[species_name], 
                   label=f'{reactor_name}', **kwargs)
        else:
            warnings.warn(f"Data for '{species_name}' not found in {reactor_name} results")
    
    title = title or f'Reactor Comparison: [{species_name}]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_spatial_profile(results: Dict,
                        species_name: str,
                        reactor_length: float,
                        n_segments: int,
                        time_points: Optional[List[float]] = None,
                        title: Optional[str] = None,
                        xlabel: str = "Reactor Length (m)",
                        ylabel: str = "Concentration (mol/L)",
                        figsize: Tuple[float, float] = (10, 6),
                        **kwargs) -> plt.Figure:
    """
    Plot spatial concentration profiles for PFR reactors.
    
    Args:
        results: Results dictionary from PFR simulation
        species_name: Name of species to plot
        reactor_length: Total reactor length
        n_segments: Number of spatial segments
        time_points: Specific time points to plot (if None, plots final state)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create spatial coordinate
    z_positions = np.linspace(0, reactor_length, n_segments)
    
    # If no specific time points, just plot final state
    if time_points is None:
        time_points = [results['time'][-1]]
    
    for t in time_points:
        # Find closest time index
        time_idx = np.argmin(np.abs(results['time'] - t))
        
        # Extract concentrations for each segment
        concentrations = []
        for segment in range(n_segments):
            seg_name = f"{species_name}_seg{segment}"
            if seg_name in results:
                concentrations.append(results[seg_name][time_idx])
            else:
                concentrations.append(0.0)
        
        ax.plot(z_positions, concentrations, 
               label=f't = {results["time"][time_idx]:.2f} s', **kwargs)
    
    title = title or f'Spatial Profile: [{species_name}]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_3d_surface(results: Dict,
                   species_name: str,
                   reactor_length: float,
                   n_segments: int,
                   title: Optional[str] = None,
                   figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
    """
    Create a 3D surface plot showing concentration evolution in space and time.
    
    Args:
        results: Results dictionary from PFR simulation
        species_name: Name of species to plot
        reactor_length: Total reactor length
        n_segments: Number of spatial segments
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    time = results['time']
    z_positions = np.linspace(0, reactor_length, n_segments)
    T, Z = np.meshgrid(time, z_positions)
    
    # Extract concentration data
    C = np.zeros((n_segments, len(time)))
    for segment in range(n_segments):
        seg_name = f"{species_name}_seg{segment}"
        if seg_name in results:
            C[segment, :] = results[seg_name]
    
    # Create surface plot
    surf = ax.plot_surface(T, Z, C, cmap='viridis', alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reactor Length (m)')
    ax.set_zlabel(f'[{species_name}] (mol/L)')
    title = title or f'3D Concentration Profile: [{species_name}]'
    ax.set_title(title)
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    return fig


def plot_multiple_species_subplots(results: Dict,
                                 species_names: List[str],
                                 n_cols: int = 2,
                                 figsize: Optional[Tuple[float, float]] = None,
                                 title: str = "Species Concentration Profiles",
                                 **kwargs) -> plt.Figure:
    """
    Create subplots for multiple species concentrations.
    
    Args:
        results: Results dictionary from solver
        species_names: List of species names to plot
        n_cols: Number of columns in subplot grid
        figsize: Figure size (auto-calculated if None)
        title: Overall title
        **kwargs: Additional plotting arguments
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_species = len(species_names)
    n_rows = (n_species + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_species == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    time = results['time']
    
    for i, species in enumerate(species_names):
        ax = axes[i]
        
        if species in results:
            ax.plot(time, results[species], **kwargs)
            ax.set_title(f'[{species}]')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Concentration (mol/L)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {species}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Hide empty subplots
    for i in range(n_species, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def save_all_plots(results: Dict,
                  species_names: List[str],
                  output_dir: str = "./plots",
                  prefix: str = "reactor_",
                  format: str = "png",
                  dpi: int = 300) -> None:
    """
    Save multiple standard plots for a simulation.
    
    Args:
        results: Results dictionary from solver
        species_names: List of species names
        output_dir: Directory to save plots
        prefix: File name prefix
        format: Image format ('png', 'pdf', 'svg', etc.)
        dpi: Image resolution
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Concentration profiles
    fig1 = plot_concentration_profiles(results, species_names)
    fig1.savefig(f"{output_dir}/{prefix}concentrations.{format}", dpi=dpi, bbox_inches='tight')
    plt.close(fig1)
    
    # Multiple species subplots
    fig2 = plot_multiple_species_subplots(results, species_names)
    fig2.savefig(f"{output_dir}/{prefix}species_subplots.{format}", dpi=dpi, bbox_inches='tight')
    plt.close(fig2)
    
    # Phase portraits for pairs of species
    if len(species_names) >= 2:
        fig3 = plot_phase_portrait(results, species_names[0], species_names[1])
        fig3.savefig(f"{output_dir}/{prefix}phase_portrait.{format}", dpi=dpi, bbox_inches='tight')
        plt.close(fig3)
    
    print(f"Plots saved to {output_dir}/")


# Set default plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'legend.framealpha': 0.9,
    'figure.autolayout': True
})
