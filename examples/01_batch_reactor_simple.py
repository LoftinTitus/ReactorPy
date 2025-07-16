#!/usr/bin/env python3
"""
Example 1: Simple Batch Reactor Simulation
==========================================

This example demonstrates the basic usage of ReactorPy for simulating
a simple batch reactor with a single irreversible reaction:
A + B -> C

The reaction follows second-order kinetics with rate = k * [A] * [B]
"""

import sys
import os

# Add the parent directory to Python path to import reactorpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactorpy import BatchReactor, Species, Reaction
import matplotlib.pyplot as plt

def main():
    print("ReactorPy Example 1: Simple Batch Reactor")
    print("=" * 50)
    
    # Define chemical species with initial concentrations
    A = Species("A", concentration=2.0)  # mol/L
    B = Species("B", concentration=1.5)  # mol/L  
    C = Species("C", concentration=0.0)  # mol/L
    
    print(f"Initial concentrations:")
    print(f"[A]₀ = {A.concentration} mol/L")
    print(f"[B]₀ = {B.concentration} mol/L")
    print(f"[C]₀ = {C.concentration} mol/L")
    
    # Define the reaction: A + B -> C
    reaction = Reaction("A + B -> C")
    reaction.set_rate_expression("k * [A] * [B]")
    reaction.add_rate_parameter('k', 0.1)  # L/(mol·s)
    
    print(f"\nReaction: {reaction.reaction_string}")
    print(f"Rate law: {reaction.rate_expression}")
    print(f"Rate constant k = {reaction.rate_parameters['k']} L/(mol·s)")
    
    # Create batch reactor
    reactor = BatchReactor("Simple Batch Reactor", volume=2.0)  # 2 L volume
    reactor.add_species([A, B, C])
    reactor.add_reaction(reaction)
    
    print(f"\nReactor: {reactor.name}")
    print(f"Volume: {reactor.volume} L")
    print(f"Temperature: {reactor.temperature} K")
    
    # Simulate the reaction for 50 seconds
    print("\nRunning simulation...")
    time_span = (0, 50)  # seconds
    results = reactor.simulate(time_span)
    
    # Display final concentrations
    final_time = results.t[-1]
    final_concentrations = {name: results.y[i, -1] for i, name in enumerate(reactor.species.keys())}
    
    print(f"\nFinal concentrations at t = {final_time:.1f} s:")
    for species, conc in final_concentrations.items():
        print(f"[{species}] = {conc:.4f} mol/L")
    
    # Calculate conversion of limiting reagent (B)
    conversion_B = (B.concentration - final_concentrations['B']) / B.concentration * 100
    print(f"\nConversion of B: {conversion_B:.1f}%")
    
    # Plot results
    reactor.plot(title="Simple Batch Reactor: A + B → C", 
                figsize=(10, 6),
                save_path="batch_reactor_simple.png")
    
    print("\nPlot saved as 'batch_reactor_simple.png'")
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()
