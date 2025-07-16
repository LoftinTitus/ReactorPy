#!/usr/bin/env python3
"""
Example 2: Batch Reactor with Multiple Reactions
=================================================

This example demonstrates a more complex batch reactor system with
multiple parallel and series reactions:

Main reaction: A + B -> C
Side reaction: A + C -> D
Product decomposition: C -> E + F

This shows how to handle complex reaction networks and analyze
selectivity and yield.
"""

import sys
import os

# Add the parent directory to Python path to import reactorpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactorpy import BatchReactor, Species, Reaction
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("ReactorPy Example 2: Multiple Reactions in Batch Reactor")
    print("=" * 60)
    
    # Define chemical species
    A = Species("A", concentration=3.0)  # mol/L
    B = Species("B", concentration=2.0)  # mol/L
    C = Species("C", concentration=0.0)  # mol/L (desired product)
    D = Species("D", concentration=0.0)  # mol/L (unwanted byproduct)
    E = Species("E", concentration=0.0)  # mol/L (decomposition product)
    F = Species("F", concentration=0.0)  # mol/L (decomposition product)
    
    print("Initial concentrations:")
    for species in [A, B, C, D, E, F]:
        print(f"[{species.name}]₀ = {species.concentration} mol/L")
    
    # Define reaction network
    # Main reaction: A + B -> C (fast, desired)
    rxn1 = Reaction("A + B -> C")
    rxn1.set_rate_expression("k1 * [A] * [B]")
    rxn1.add_rate_parameter('k1', 0.15)  # L/(mol·s)
    
    # Side reaction: A + C -> D (slower, undesired)  
    rxn2 = Reaction("A + C -> D")
    rxn2.set_rate_expression("k2 * [A] * [C]")
    rxn2.add_rate_parameter('k2', 0.05)  # L/(mol·s)
    
    # Product decomposition: C -> E + F (slow)
    rxn3 = Reaction("C -> E + F")
    rxn3.set_rate_expression("k3 * [C]")
    rxn3.add_rate_parameter('k3', 0.02)  # 1/s
    
    reactions = [rxn1, rxn2, rxn3]
    
    print(f"\nReaction network:")
    for i, rxn in enumerate(reactions, 1):
        print(f"R{i}: {rxn.reaction_string}")
        print(f"    Rate: {rxn.rate_expression}")
        rate_param = list(rxn.rate_parameters.keys())[0]
        print(f"    {rate_param} = {rxn.rate_parameters[rate_param]}")
    
    # Create batch reactor
    reactor = BatchReactor("Multi-Reaction Batch Reactor", volume=5.0)
    reactor.add_species([A, B, C, D, E, F])
    reactor.add_reaction(reactions)
    
    # Simulate for 100 seconds
    print(f"\nSimulating {reactor.name} for 100 seconds...")
    time_span = (0, 100)
    results = reactor.simulate(time_span)
    
    # Analyze results
    final_concentrations = {name: results.y[i, -1] for i, name in enumerate(reactor.species.keys())}
    
    print(f"\nFinal concentrations:")
    for species, conc in final_concentrations.items():
        print(f"[{species}] = {conc:.4f} mol/L")
    
    # Calculate performance metrics
    A_converted = A.concentration - final_concentrations['A']
    A_conversion = A_converted / A.concentration * 100
    
    C_yield = final_concentrations['C'] / A.concentration * 100
    D_yield = final_concentrations['D'] / A.concentration * 100
    E_yield = final_concentrations['E'] / A.concentration * 100
    
    selectivity_C_over_D = final_concentrations['C'] / (final_concentrations['C'] + final_concentrations['D']) * 100 if (final_concentrations['C'] + final_concentrations['D']) > 0 else 0
    
    print(f"\nPerformance Analysis:")
    print(f"A conversion: {A_conversion:.1f}%")
    print(f"Yield of C: {C_yield:.1f}%")
    print(f"Yield of D: {D_yield:.1f}%") 
    print(f"Yield of E: {E_yield:.1f}%")
    print(f"Selectivity C/(C+D): {selectivity_C_over_D:.1f}%")
    
    # Plot concentration profiles
    reactor.plot(title="Multi-Reaction Batch Reactor",
                figsize=(12, 8),
                save_path="batch_reactor_multi_reaction.png")
    
    # Create custom selectivity plot
    plt.figure(figsize=(10, 6))
    
    # Calculate selectivity over time
    time_points = results.t
    species_names = list(reactor.species.keys())
    C_idx = species_names.index('C')
    D_idx = species_names.index('D')
    
    C_conc = results.y[C_idx, :]
    D_conc = results.y[D_idx, :]
    
    # Avoid division by zero
    selectivity_time = np.where((C_conc + D_conc) > 1e-10, 
                               C_conc / (C_conc + D_conc) * 100,
                               100)
    
    plt.plot(time_points, selectivity_time, 'r-', linewidth=2, label='C/(C+D) Selectivity')
    plt.xlabel('Time (s)')
    plt.ylabel('Selectivity (%)')
    plt.title('Selectivity of Desired Product C over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig('selectivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPlots saved:")
    print("- batch_reactor_multi_reaction.png")
    print("- selectivity_analysis.png")
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()
