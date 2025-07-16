#!/usr/bin/env python3
"""
Example 3: CSTR (Continuous Stirred Tank Reactor) Steady State
===============================================================

This example demonstrates steady-state operation of a CSTR with:
- Continuous feed and effluent streams
- Single reaction: A -> B
- Finding steady-state concentrations
- Analyzing the effect of residence time

Key concepts:
- Residence time (τ) = V/Q
- Material balance: Accumulation = In - Out + Generation
- Steady state: dC/dt = 0
"""

import sys
import os

# Add the parent directory to Python path to import reactorpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactorpy import CSTR, Species, Reaction
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("ReactorPy Example 3: CSTR Steady State Analysis")
    print("=" * 50)
    
    # Define feed conditions
    A_feed = Species("A", concentration=5.0)  # mol/L in feed
    B_feed = Species("B", concentration=0.0)  # mol/L in feed
    
    print("Feed conditions:")
    print(f"[A]_feed = {A_feed.concentration} mol/L")
    print(f"[B]_feed = {B_feed.concentration} mol/L")
    
    # Define reaction: A -> B (first order)
    reaction = Reaction("A -> B")
    reaction.set_rate_expression("k * [A]")
    reaction.add_rate_parameter('k', 0.3)  # 1/s
    
    print(f"\nReaction: {reaction.reaction_string}")
    print(f"Rate law: {reaction.rate_expression}")
    print(f"Rate constant k = {reaction.rate_parameters['k']} 1/s")
    
    # Base case CSTR conditions
    reactor_volume = 10.0  # L
    flow_rate = 2.0       # L/s
    residence_time = reactor_volume / flow_rate  # s
    
    print(f"\nReactor conditions:")
    print(f"Volume: {reactor_volume} L")
    print(f"Flow rate: {flow_rate} L/s")
    print(f"Residence time: {residence_time} s")
    
    # Create CSTR and set up species
    cstr = CSTR("Steady State CSTR", volume=reactor_volume, flow_rate=flow_rate)
    
    # Add species to reactor (these will be the exit concentrations)
    A_exit = Species("A", concentration=1.0)  # Initial guess
    B_exit = Species("B", concentration=0.0)  # Initial guess
    
    cstr.add_species([A_exit, B_exit])
    cstr.add_reaction(reaction)
    
    # Set feed concentrations
    cstr.set_feed_concentration("A", A_feed.concentration)
    cstr.set_feed_concentration("B", B_feed.concentration)
    
    # Solve for steady state
    print("\nSolving for steady-state concentrations...")
    steady_state = cstr.solve_steady_state()
    
    if steady_state.success:
        A_ss = steady_state.x[0]
        B_ss = steady_state.x[1]
        
        print(f"\nSteady-state concentrations:")
        print(f"[A]_exit = {A_ss:.4f} mol/L")
        print(f"[B]_exit = {B_ss:.4f} mol/L")
        
        # Calculate performance metrics
        conversion = (A_feed.concentration - A_ss) / A_feed.concentration * 100
        yield_B = B_ss / A_feed.concentration * 100
        
        print(f"\nPerformance:")
        print(f"Conversion of A: {conversion:.1f}%")
        print(f"Yield of B: {yield_B:.1f}%")
        
        # Verify material balance
        reaction_rate = reaction.rate_parameters['k'] * A_ss
        print(f"\nMaterial balance check:")
        print(f"Reaction rate: {reaction_rate:.4f} mol/(L·s)")
        print(f"Expected from balance: {flow_rate/reactor_volume * (A_feed.concentration - A_ss):.4f} mol/(L·s)")
    else:
        print("Failed to find steady state solution!")
        return
    
    # Parametric study: Effect of residence time
    print("\n" + "="*50)
    print("Parametric Study: Effect of Residence Time")
    print("="*50)
    
    # Range of flow rates (inverse residence times)
    flow_rates = np.logspace(-1, 1, 20)  # 0.1 to 10 L/s
    residence_times = reactor_volume / flow_rates
    
    conversions = []
    A_concentrations = []
    B_concentrations = []
    
    for Q in flow_rates:
        # Create new CSTR for each flow rate
        cstr_param = CSTR("Parametric CSTR", volume=reactor_volume, flow_rate=Q)
        
        A_param = Species("A", concentration=1.0)
        B_param = Species("B", concentration=0.0)
        
        cstr_param.add_species([A_param, B_param])
        cstr_param.add_reaction(reaction)
        cstr_param.set_feed_concentration("A", A_feed.concentration)
        cstr_param.set_feed_concentration("B", B_feed.concentration)
        
        # Solve steady state
        ss_result = cstr_param.solve_steady_state()
        
        if ss_result.success:
            A_exit_param = ss_result.x[0]
            B_exit_param = ss_result.x[1]
            
            conversion_param = (A_feed.concentration - A_exit_param) / A_feed.concentration * 100
            
            conversions.append(conversion_param)
            A_concentrations.append(A_exit_param)
            B_concentrations.append(B_exit_param)
        else:
            conversions.append(np.nan)
            A_concentrations.append(np.nan)
            B_concentrations.append(np.nan)
    
    # Plot parametric results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Conversion vs Residence Time
    ax1.semilogx(residence_times, conversions, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Residence Time (s)')
    ax1.set_ylabel('Conversion (%)')
    ax1.set_title('CSTR Performance: Conversion vs Residence Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Mark the base case
    ax1.axvline(residence_time, color='red', linestyle='--', alpha=0.7, label=f'Base case (τ = {residence_time:.1f} s)')
    ax1.legend()
    
    # Plot 2: Exit concentrations vs Residence Time
    ax2.semilogx(residence_times, A_concentrations, 'ro-', linewidth=2, markersize=6, label='[A] exit')
    ax2.semilogx(residence_times, B_concentrations, 'bo-', linewidth=2, markersize=6, label='[B] exit')
    ax2.axhline(A_feed.concentration, color='red', linestyle=':', alpha=0.7, label='[A] feed')
    ax2.set_xlabel('Residence Time (s)')
    ax2.set_ylabel('Concentration (mol/L)')
    ax2.set_title('CSTR: Exit Concentrations vs Residence Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Mark the base case
    ax2.axvline(residence_time, color='green', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('cstr_steady_state_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nParametric study completed!")
    print(f"Flow rate range: {flow_rates[0]:.1f} - {flow_rates[-1]:.1f} L/s")
    print(f"Residence time range: {residence_times[-1]:.1f} - {residence_times[0]:.1f} s")
    print("Plot saved as 'cstr_steady_state_analysis.png'")
    
    print("\nKey insights:")
    print("- Higher residence time = higher conversion")
    print("- Diminishing returns at very high residence times")
    print("- Trade-off between conversion and throughput")

if __name__ == "__main__":
    main()
