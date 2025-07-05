import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import os
import warnings


def export_to_csv(results: Dict,
                 filename: str,
                 species_names: Optional[List[str]] = None,
                 include_metadata: bool = True,
                 precision: int = 6) -> str:
    """
    Export simulation results to CSV format.
    
    Args:
        results: Results dictionary from solver
        filename: Output CSV filename (with or without .csv extension)
        species_names: List of species to export. If None, exports all species
        include_metadata: Whether to include simulation metadata as comments
        precision: Number of decimal places for numerical data
        
    Returns:
        str: Full path to the created CSV file
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Get time data
    if 'time' not in results:
        raise ValueError("Results must contain 'time' data")
    
    time = results['time']
    
    # Determine species to export
    if species_names is None:
        species_names = results.get('species_names', [])
        if not species_names:
            # Try to infer from results keys
            species_names = [key for key in results.keys() 
                           if key not in ['time', 'success', 'message', 'nfev', 'njev', 'nlu', 
                                        'status', 'method', 'final_concentrations', 'solution_object']]
    
    # Create DataFrame
    data = {'Time_s': time}
    
    # Add species concentrations
    for species in species_names:
        if species in results:
            column_name = f'{species}_mol_per_L'
            data[column_name] = np.round(results[species], precision)
        else:
            warnings.warn(f"Species '{species}' not found in results")
    
    df = pd.DataFrame(data)
    
    # Write to CSV with metadata
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write metadata as comments if requested
        if include_metadata:
            writer.writerow([f'# ReactorPy Simulation Results'])
            writer.writerow([f'# Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
            writer.writerow([f'# Solver method: {results.get("method", "Unknown")}'])
            writer.writerow([f'# Solver success: {results.get("success", "Unknown")}'])
            writer.writerow([f'# Function evaluations: {results.get("nfev", "Unknown")}'])
            
            if 'final_concentrations' in results:
                writer.writerow(['# Final concentrations:'])
                for species, conc in results['final_concentrations'].items():
                    writer.writerow([f'#   {species}: {conc:.{precision}f} mol/L'])
            
            writer.writerow(['#'])
            writer.writerow(['# Column descriptions:'])
            writer.writerow(['#   Time_s: Time in seconds'])
            for species in species_names:
                if species in results:
                    writer.writerow([f'#   {species}_mol_per_L: Concentration of {species} in mol/L'])
            writer.writerow(['#'])
        
        # Write header
        writer.writerow(df.columns.tolist())
        
        # Write data
        for _, row in df.iterrows():
            writer.writerow(row.tolist())
    
    print(f"Results exported to CSV: {filename}")
    return os.path.abspath(filename)


def export_to_excel(results: Dict,
                   filename: str,
                   species_names: Optional[List[str]] = None,
                   include_summary: bool = True,
                   precision: int = 6) -> str:
    """
    Export simulation results to Excel format with multiple sheets.
    
    Args:
        results: Results dictionary from solver
        filename: Output Excel filename (with or without .xlsx extension)
        species_names: List of species to export. If None, exports all species
        include_summary: Whether to include a summary sheet
        precision: Number of decimal places for numerical data
        
    Returns:
        str: Full path to the created Excel file
    """
    # Ensure filename has .xlsx extension
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    
    # Get time data
    if 'time' not in results:
        raise ValueError("Results must contain 'time' data")
    
    time = results['time']
    
    # Determine species to export
    if species_names is None:
        species_names = results.get('species_names', [])
        if not species_names:
            species_names = [key for key in results.keys() 
                           if key not in ['time', 'success', 'message', 'nfev', 'njev', 'nlu', 
                                        'status', 'method', 'final_concentrations', 'solution_object']]
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main data sheet
        data = {'Time_s': time}
        for species in species_names:
            if species in results:
                data[f'{species}_mol_per_L'] = np.round(results[species], precision)
        
        df_data = pd.DataFrame(data)
        df_data.to_excel(writer, sheet_name='Concentration_Data', index=False)
        
        # Summary sheet
        if include_summary:
            summary_data = []
            summary_data.append(['ReactorPy Simulation Summary', ''])
            summary_data.append(['Generated on', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            summary_data.append(['', ''])
            summary_data.append(['Simulation Parameters', ''])
            summary_data.append(['Solver method', results.get('method', 'Unknown')])
            summary_data.append(['Solver success', results.get('success', 'Unknown')])
            summary_data.append(['Function evaluations', results.get('nfev', 'Unknown')])
            summary_data.append(['Integration time span', f"{time[0]:.2f} - {time[-1]:.2f} s"])
            summary_data.append(['Number of time points', len(time)])
            summary_data.append(['', ''])
            
            # Initial and final concentrations
            if 'final_concentrations' in results:
                summary_data.append(['Initial Concentrations (mol/L)', ''])
                for species in species_names:
                    if species in results:
                        initial_conc = results[species][0]
                        summary_data.append([species, f"{initial_conc:.{precision}f}"])
                
                summary_data.append(['', ''])
                summary_data.append(['Final Concentrations (mol/L)', ''])
                for species, conc in results['final_concentrations'].items():
                    if species in species_names:
                        summary_data.append([species, f"{conc:.{precision}f}"])
            
            df_summary = pd.DataFrame(summary_data, columns=['Parameter', 'Value'])
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Results exported to Excel: {filename}")
    return os.path.abspath(filename)


def export_to_pdf_report(results: Dict,
                        filename: str,
                        species_names: Optional[List[str]] = None,
                        reactor_info: Optional[Dict] = None,
                        reactions_info: Optional[List] = None,
                        include_plots: bool = True,
                        include_data_table: bool = True,
                        plot_style: str = 'seaborn',
                        title: str = "Reactor Simulation Report") -> str:
    """
    Export simulation results to a comprehensive PDF report.
    
    Args:
        results: Results dictionary from solver
        filename: Output PDF filename (with or without .pdf extension)
        species_names: List of species to include. If None, includes all species
        reactor_info: Dictionary with reactor information (name, volume, temperature, etc.)
        reactions_info: List of reaction objects or reaction strings
        include_plots: Whether to include plots in the report
        include_data_table: Whether to include data table
        plot_style: Matplotlib style for plots
        title: Report title
        
    Returns:
        str: Full path to the created PDF file
    """
    # Import plotting functions
    from .plotting import (plot_concentration_profiles, plot_conversion_selectivity, 
                          plot_phase_portrait, plot_multiple_species_subplots)
    
    # Ensure filename has .pdf extension
    if not filename.endswith('.pdf'):
        filename += '.pdf'
    
    # Get time data
    if 'time' not in results:
        raise ValueError("Results must contain 'time' data")
    
    # Determine species to export
    if species_names is None:
        species_names = results.get('species_names', [])
        if not species_names:
            species_names = [key for key in results.keys() 
                           if key not in ['time', 'success', 'message', 'nfev', 'njev', 'nlu', 
                                        'status', 'method', 'final_concentrations', 'solution_object']]
    
    # Set plot style
    if plot_style != 'default':
        plt.style.use(plot_style)
    
    with PdfPages(filename) as pdf:
        # Title page
        fig_title = plt.figure(figsize=(8.5, 11))
        fig_title.text(0.5, 0.8, title, fontsize=24, ha='center', weight='bold')
        fig_title.text(0.5, 0.7, 'ReactorPy Simulation Report', fontsize=16, ha='center')
        fig_title.text(0.5, 0.6, f'Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}', 
                      fontsize=12, ha='center')
        
        # Add reactor information if provided
        y_pos = 0.5
        if reactor_info:
            fig_title.text(0.5, y_pos, 'Reactor Information', fontsize=14, ha='center', weight='bold')
            y_pos -= 0.05
            for key, value in reactor_info.items():
                fig_title.text(0.5, y_pos, f'{key}: {value}', fontsize=11, ha='center')
                y_pos -= 0.03
        
        # Add simulation info
        y_pos -= 0.05
        fig_title.text(0.5, y_pos, 'Simulation Information', fontsize=14, ha='center', weight='bold')
        y_pos -= 0.05
        sim_info = [
            f"Solver: {results.get('method', 'Unknown')}",
            f"Success: {results.get('success', 'Unknown')}",
            f"Function Evaluations: {results.get('nfev', 'Unknown')}",
            f"Time Span: {results['time'][0]:.2f} - {results['time'][-1]:.2f} s",
            f"Species: {', '.join(species_names)}"
        ]
        
        for info in sim_info:
            fig_title.text(0.5, y_pos, info, fontsize=11, ha='center')
            y_pos -= 0.03
        
        fig_title.axis('off')
        pdf.savefig(fig_title, bbox_inches='tight')
        plt.close(fig_title)
        
        # Reaction information page
        if reactions_info:
            fig_rxn = plt.figure(figsize=(8.5, 11))
            fig_rxn.text(0.5, 0.9, 'Reaction Network', fontsize=18, ha='center', weight='bold')
            
            y_pos = 0.8
            for i, reaction in enumerate(reactions_info, 1):
                if hasattr(reaction, 'reaction_string'):
                    rxn_text = f"R{i}: {reaction.reaction_string}"
                    if hasattr(reaction, 'rate_expression') and reaction.rate_expression:
                        rxn_text += f"\n     Rate: {reaction.rate_expression}"
                    if hasattr(reaction, 'rate_parameters') and reaction.rate_parameters:
                        params = ', '.join([f"{k}={v}" for k, v in reaction.rate_parameters.items()])
                        rxn_text += f"\n     Parameters: {params}"
                else:
                    rxn_text = f"R{i}: {str(reaction)}"
                
                fig_rxn.text(0.1, y_pos, rxn_text, fontsize=11, va='top', family='monospace')
                y_pos -= 0.1
            
            fig_rxn.axis('off')
            pdf.savefig(fig_rxn, bbox_inches='tight')
            plt.close(fig_rxn)
        
        # Plots
        if include_plots:
            # Main concentration profile
            try:
                fig1 = plot_concentration_profiles(
                    results, species_names, 
                    title="Concentration Profiles Over Time",
                    figsize=(10, 6)
                )
                pdf.savefig(fig1, bbox_inches='tight')
                plt.close(fig1)
            except Exception as e:
                warnings.warn(f"Could not create concentration profile plot: {e}")
            
            # Multiple species subplots
            try:
                fig2 = plot_multiple_species_subplots(
                    results, species_names,
                    title="Individual Species Profiles",
                    figsize=(12, 8)
                )
                pdf.savefig(fig2, bbox_inches='tight')
                plt.close(fig2)
            except Exception as e:
                warnings.warn(f"Could not create species subplots: {e}")
            
            # Phase portrait (if at least 2 species)
            if len(species_names) >= 2:
                try:
                    fig3 = plot_phase_portrait(
                        results, species_names[0], species_names[1],
                        figsize=(8, 8)
                    )
                    pdf.savefig(fig3, bbox_inches='tight')
                    plt.close(fig3)
                except Exception as e:
                    warnings.warn(f"Could not create phase portrait: {e}")
            
            # Conversion plot (assume first species is reactant)
            if len(species_names) >= 2:
                try:
                    products = species_names[1:]  # Assume others are products
                    fig4 = plot_conversion_selectivity(
                        results, species_names[0], products,
                        figsize=(12, 5)
                    )
                    pdf.savefig(fig4, bbox_inches='tight')
                    plt.close(fig4)
                except Exception as e:
                    warnings.warn(f"Could not create conversion plot: {e}")
        
        # Data table
        if include_data_table:
            # Create data table page
            time = results['time']
            n_points = len(time)
            
            # If too many points, subsample for readability
            if n_points > 50:
                step = n_points // 50
                indices = range(0, n_points, step)
                time_subset = time[indices]
            else:
                indices = range(n_points)
                time_subset = time
            
            # Prepare data
            table_data = []
            headers = ['Time (s)'] + [f'[{species}] (mol/L)' for species in species_names]
            table_data.append(headers)
            
            for i in indices:
                row = [f"{time[i]:.3f}"]
                for species in species_names:
                    if species in results:
                        row.append(f"{results[species][i]:.6f}")
                    else:
                        row.append("N/A")
                table_data.append(row)
            
            # Create table plot
            fig_table = plt.figure(figsize=(8.5, 11))
            ax_table = fig_table.add_subplot(111)
            ax_table.axis('tight')
            ax_table.axis('off')
            
            # Create table
            table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                                 cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # Style header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.title('Concentration Data Table', fontsize=14, weight='bold', pad=20)
            pdf.savefig(fig_table, bbox_inches='tight')
            plt.close(fig_table)
        
        # Summary statistics page
        fig_stats = plt.figure(figsize=(8.5, 11))
        fig_stats.text(0.5, 0.9, 'Summary Statistics', fontsize=18, ha='center', weight='bold')
        
        y_pos = 0.8
        for species in species_names:
            if species in results:
                conc_data = results[species]
                stats_text = f"Species: {species}\n"
                stats_text += f"  Initial: {conc_data[0]:.6f} mol/L\n"
                stats_text += f"  Final: {conc_data[-1]:.6f} mol/L\n"
                stats_text += f"  Maximum: {np.max(conc_data):.6f} mol/L\n"
                stats_text += f"  Minimum: {np.min(conc_data):.6f} mol/L\n"
                stats_text += f"  Average: {np.mean(conc_data):.6f} mol/L\n"
                
                fig_stats.text(0.1, y_pos, stats_text, fontsize=11, va='top', family='monospace')
                y_pos -= 0.15
        
        fig_stats.axis('off')
        pdf.savefig(fig_stats, bbox_inches='tight')
        plt.close(fig_stats)
    
    print(f"PDF report exported to: {filename}")
    return os.path.abspath(filename)


def export_results_json(results: Dict,
                       filename: str,
                       species_names: Optional[List[str]] = None,
                       include_metadata: bool = True,
                       precision: int = 6) -> str:
    """
    Export simulation results to JSON format.
    
    Args:
        results: Results dictionary from solver
        filename: Output JSON filename (with or without .json extension)
        species_names: List of species to export. If None, exports all species
        include_metadata: Whether to include simulation metadata
        precision: Number of decimal places for numerical data
        
    Returns:
        str: Full path to the created JSON file
    """
    # Ensure filename has .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Prepare data for JSON export
    export_data = {}
    
    if include_metadata:
        export_data['metadata'] = {
            'generated_on': datetime.now().isoformat(),
            'solver_method': results.get('method', 'Unknown'),
            'solver_success': results.get('success', False),
            'function_evaluations': results.get('nfev', 0),
            'generator': 'ReactorPy'
        }
    
    # Time data
    if 'time' in results:
        export_data['time'] = np.round(results['time'], precision).tolist()
    
    # Species data
    if species_names is None:
        species_names = results.get('species_names', [])
        if not species_names:
            species_names = [key for key in results.keys() 
                           if key not in ['time', 'success', 'message', 'nfev', 'njev', 'nlu', 
                                        'status', 'method', 'final_concentrations', 'solution_object']]
    
    export_data['concentrations'] = {}
    for species in species_names:
        if species in results:
            export_data['concentrations'][species] = np.round(results[species], precision).tolist()
    
    # Final concentrations
    if 'final_concentrations' in results:
        export_data['final_concentrations'] = {
            k: round(v, precision) for k, v in results['final_concentrations'].items()
            if k in species_names
        }
    
    # Write to JSON file
    with open(filename, 'w') as jsonfile:
        json.dump(export_data, jsonfile, indent=2)
    
    print(f"Results exported to JSON: {filename}")
    return os.path.abspath(filename)


def batch_export(results: Dict,
                base_filename: str,
                species_names: Optional[List[str]] = None,
                formats: List[str] = ['csv', 'pdf', 'json'],
                reactor_info: Optional[Dict] = None,
                reactions_info: Optional[List] = None) -> Dict[str, str]:
    """
    Export results to multiple formats at once.
    
    Args:
        results: Results dictionary from solver
        base_filename: Base filename (without extension)
        species_names: List of species to export
        formats: List of formats to export ('csv', 'pdf', 'json', 'excel')
        reactor_info: Reactor information for PDF report
        reactions_info: Reaction information for PDF report
        
    Returns:
        Dict[str, str]: Dictionary mapping format to exported file path
    """
    exported_files = {}
    
    for fmt in formats:
        try:
            if fmt.lower() == 'csv':
                path = export_to_csv(results, f"{base_filename}.csv", species_names)
                exported_files['csv'] = path
            
            elif fmt.lower() == 'pdf':
                path = export_to_pdf_report(
                    results, f"{base_filename}.pdf", species_names,
                    reactor_info=reactor_info, reactions_info=reactions_info
                )
                exported_files['pdf'] = path
            
            elif fmt.lower() == 'json':
                path = export_results_json(results, f"{base_filename}.json", species_names)
                exported_files['json'] = path
            
            elif fmt.lower() == 'excel':
                path = export_to_excel(results, f"{base_filename}.xlsx", species_names)
                exported_files['excel'] = path
            
            else:
                warnings.warn(f"Unknown format: {fmt}")
        
        except Exception as e:
            warnings.warn(f"Failed to export {fmt}: {e}")
    
    return exported_files
