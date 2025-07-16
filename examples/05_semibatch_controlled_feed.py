#!/usr/bin/env python3
"""
Example 5: Semi-Batch Reactor with Time-Varying Feed
=====================================================

This example demonstrates a semi-batch reactor where:
- One reactant is initially charged to the reactor
- Another reactant is fed continuously over time
- Reaction: A + B -> C (second order)
- Demonstrates the effect of feed rate on selectivity

This is common in pharmaceutical and fine chemical manufacturing
where controlling the addition rate is critical for selectivity.
"""

import sys
import os

# Add the parent directory to Python path to import reactorpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactorpy import SemiBatchReactor, Species, Reaction
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("ReactorPy Example 5: Semi-Batch Reactor with Controlled Feed")
    print("=" * 60)
    
    # Initial charge to reactor (Species A)
    A_initial = Species("A", concentration=2.0)  # mol/L
    B_initial = Species("B", concentration=0.0)  # mol/L (will be fed)
    C_initial = Species("C", concentration=0.0)  # mol/L (product)
    D_initial = Species("D", concentration=0.0)  # mol/L (byproduct)
    
    print("Initial reactor charge:")
    print(f"[A]₀ = {A_initial.concentration} mol/L")
    print(f"[B]₀ = {B_initial.concentration} mol/L")
    print(f"[C]₀ = {C_initial.concentration} mol/L")
    print(f"[D]₀ = {D_initial.concentration} mol/L")
    
    # Define reaction network
    # Main reaction: A + B -> C (desired, second order)
    rxn1 = Reaction("A + B -> C")
    rxn1.set_rate_expression("k1 * [A] * [B]")
    rxn1.add_rate_parameter('k1', 0.2)  # L/(mol·s)
    
    # Side reaction: A + A -> D (undesired, happens at high [A])
    rxn2 = Reaction("2A -> D")  
    rxn2.set_rate_expression("k2 * [A]^2")
    rxn2.add_rate_parameter('k2', 0.05)  # L/(mol·s)
    
    reactions = [rxn1, rxn2]
    
    print(f"\nReaction network:")
    print(f"Main: {rxn1.reaction_string}, Rate = {rxn1.rate_expression}")
    print(f"Side: {rxn2.reaction_string}, Rate = {rxn2.rate_expression}")
    print(f"k1 = {rxn1.rate_parameters['k1']} L/(mol·s)")
    print(f"k2 = {rxn2.rate_parameters['k2']} L/(mol·s)")
    
    # Semi-batch reactor setup
    initial_volume = 1.0  # L
    feed_concentration_B = 5.0  # mol/L
    
    print(f"\nReactor setup:")
    print(f"Initial volume: {initial_volume} L") 
    print(f"Feed concentration [B]: {feed_concentration_B} mol/L")
    
    # Case 1: Fast feed (poor selectivity expected)
    print(f"\n" + "="*40)
    print("Case 1: Fast Feed Rate")
    print("="*40)
    
    fast_feed_rate = 0.1  # L/s
    feed_duration = 20    # s
    total_time = 60      # s
    
    print(f"Feed rate: {fast_feed_rate} L/s")
    print(f"Feed duration: {feed_duration} s")
    
    # Create semi-batch reactor
    reactor_fast = SemiBatchReactor("Fast Feed Semi-Batch", 
                                   initial_volume=initial_volume)
    
    # Add species
    A_fast = Species("A", concentration=A_initial.concentration)
    B_fast = Species("B", concentration=B_initial.concentration)
    C_fast = Species("C", concentration=C_initial.concentration)
    D_fast = Species("D", concentration=D_initial.concentration)
    
    reactor_fast.add_species([A_fast, B_fast, C_fast, D_fast])
    reactor_fast.add_reaction(reactions)
    
    # Set up feed schedule
    def fast_feed_schedule(t):
        """Define when and how much B is fed"""
        if t <= feed_duration:
            return fast_feed_rate  # L/s
        else:
            return 0.0
    
    def feed_composition(t):
        """Define composition of feed stream"""
        return {"B": feed_concentration_B}  # Only B in feed
    
    reactor_fast.feed_schedule(fast_feed_schedule)
    reactor_fast.set_feed_composition(feed_composition)
    
    # Simulate
    print("Simulating fast feed case...")
    results_fast = reactor_fast.simulate((0, total_time))
    
    # Case 2: Slow feed (better selectivity expected)
    print(f"\n" + "="*40)
    print("Case 2: Slow Feed Rate")
    print("="*40)
    
    slow_feed_rate = 0.02  # L/s
    slow_feed_duration = 100  # s
    slow_total_time = 150    # s
    
    print(f"Feed rate: {slow_feed_rate} L/s")
    print(f"Feed duration: {slow_feed_duration} s")
    
    # Create second reactor
    reactor_slow = SemiBatchReactor("Slow Feed Semi-Batch",
                                   initial_volume=initial_volume)
    
    A_slow = Species("A", concentration=A_initial.concentration)
    B_slow = Species("B", concentration=B_initial.concentration) 
    C_slow = Species("C", concentration=C_initial.concentration)
    D_slow = Species("D", concentration=D_initial.concentration)
    
    reactor_slow.add_species([A_slow, B_slow, C_slow, D_slow])
    reactor_slow.add_reaction(reactions)
    
    def slow_feed_schedule(t):
        if t <= slow_feed_duration:
            return slow_feed_rate
        else:
            return 0.0
    
    reactor_slow.set_feed_schedule(slow_feed_schedule)
    reactor_slow.set_feed_composition(feed_composition)
    
    print("Simulating slow feed case...")
    results_slow = reactor_slow.simulate((0, slow_total_time))
    
    # Analyze results
    print(f"\n" + "="*50)
    print("Results Comparison")
    print("="*50)
    
    # Fast feed final results
    species_names = list(reactor_fast.species.keys())
    final_fast = {name: results_fast.y[i, -1] for i, name in enumerate(species_names)}
    
    # Slow feed final results  
    final_slow = {name: results_slow.y[i, -1] for i, name in enumerate(species_names)}
    
    print("Fast feed - Final concentrations:")
    for species, conc in final_fast.items():
        print(f"[{species}] = {conc:.4f} mol/L")
    
    print("\nSlow feed - Final concentrations:")
    for species, conc in final_slow.items():
        print(f"[{species}] = {conc:.4f} mol/L")
    
    # Calculate selectivities
    selectivity_fast = final_fast['C'] / (final_fast['C'] + final_fast['D']) * 100 if (final_fast['C'] + final_fast['D']) > 0 else 0
    selectivity_slow = final_slow['C'] / (final_slow['C'] + final_slow['D']) * 100 if (final_slow['C'] + final_slow['D']) > 0 else 0
    
    print(f"\nSelectivity C/(C+D):")
    print(f"Fast feed: {selectivity_fast:.1f}%")
    print(f"Slow feed: {selectivity_slow:.1f}%")
    print(f"Improvement: {selectivity_slow - selectivity_fast:.1f} percentage points")
    
    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Fast feed concentration profiles
    ax1.plot(results_fast.t, results_fast.y[0, :], 'r-', linewidth=2, label='[A]')
    ax1.plot(results_fast.t, results_fast.y[1, :], 'b-', linewidth=2, label='[B]')
    ax1.plot(results_fast.t, results_fast.y[2, :], 'g-', linewidth=2, label='[C]')
    ax1.plot(results_fast.t, results_fast.y[3, :], 'm-', linewidth=2, label='[D]')
    ax1.axvline(feed_duration, color='black', linestyle='--', alpha=0.5, label='Feed stops')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Concentration (mol/L)')
    ax1.set_title('Fast Feed: Concentration Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Slow feed concentration profiles
    ax2.plot(results_slow.t, results_slow.y[0, :], 'r-', linewidth=2, label='[A]')
    ax2.plot(results_slow.t, results_slow.y[1, :], 'b-', linewidth=2, label='[B]')
    ax2.plot(results_slow.t, results_slow.y[2, :], 'g-', linewidth=2, label='[C]')
    ax2.plot(results_slow.t, results_slow.y[3, :], 'm-', linewidth=2, label='[D]')
    ax2.axvline(slow_feed_duration, color='black', linestyle='--', alpha=0.5, label='Feed stops')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Concentration (mol/L)')
    ax2.set_title('Slow Feed: Concentration Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Volume profiles
    # For fast feed
    time_fast = results_fast.t
    volume_fast = np.ones_like(time_fast) * initial_volume
    for i, t in enumerate(time_fast):
        if t <= feed_duration:
            volume_fast[i] = initial_volume + fast_feed_rate * t
        else:
            volume_fast[i] = initial_volume + fast_feed_rate * feed_duration
    
    ax3.plot(time_fast, volume_fast, 'purple', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Volume (L)')
    ax3.set_title('Fast Feed: Reactor Volume')
    ax3.grid(True, alpha=0.3)
    
    # For slow feed
    time_slow = results_slow.t
    volume_slow = np.ones_like(time_slow) * initial_volume
    for i, t in enumerate(time_slow):
        if t <= slow_feed_duration:
            volume_slow[i] = initial_volume + slow_feed_rate * t
        else:
            volume_slow[i] = initial_volume + slow_feed_rate * slow_feed_duration
    
    ax4.plot(time_slow, volume_slow, 'purple', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Volume (L)')
    ax4.set_title('Slow Feed: Reactor Volume')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('semibatch_feed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot selectivity comparison
    plt.figure(figsize=(10, 6))
    
    # Calculate selectivity over time for both cases
    C_fast = results_fast.y[2, :]
    D_fast = results_fast.y[3, :]
    selectivity_time_fast = np.where((C_fast + D_fast) > 1e-10, 
                                   C_fast / (C_fast + D_fast) * 100, 100)
    
    C_slow = results_slow.y[2, :]
    D_slow = results_slow.y[3, :]
    selectivity_time_slow = np.where((C_slow + D_slow) > 1e-10,
                                   C_slow / (C_slow + D_slow) * 100, 100)
    
    plt.plot(results_fast.t, selectivity_time_fast, 'r-', linewidth=3, 
             label=f'Fast feed (final: {selectivity_fast:.1f}%)')
    plt.plot(results_slow.t, selectivity_time_slow, 'b-', linewidth=3,
             label=f'Slow feed (final: {selectivity_slow:.1f}%)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Selectivity C/(C+D) (%)')
    plt.title('Semi-Batch Reactor: Effect of Feed Rate on Selectivity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('semibatch_selectivity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPlots saved:")
    print("- semibatch_feed_comparison.png")
    print("- semibatch_selectivity_comparison.png")
    
    print(f"\nKey insights:")
    print("- Slower feed maintains lower [A], reducing side reaction")
    print("- Better selectivity comes at cost of longer reaction time")
    print("- Semi-batch operation allows optimization of selectivity vs productivity")
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()
