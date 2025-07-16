#!/usr/bin/env python3
"""
Example 4: PFR (Plug Flow Reactor) with Spatial Profiles
=========================================================

This example demonstrates the simulation of a Plug Flow Reactor (PFR) where:
- Reaction proceeds along the reactor length
- No mixing in the axial direction
- Concentration varies with position
- Material balance: dC/dz = r(C)/u

Reaction: A -> B + C (first order decomposition)
Shows spatial concentration profiles and conversion along reactor length.
"""

import sys
import os

# Add the parent directory to Python path to import reactorpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactorpy import PFR, Species, Reaction
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("ReactorPy Example 4: PFR Spatial Concentration Profiles")
    print("=" * 60)
    
    # Define chemical species with inlet conditions
    A_inlet = Species("A", concentration=4.0)  # mol/L
    B_inlet = Species("B", concentration=0.0)  # mol/L  
    C_inlet = Species("C", concentration=0.0)  # mol/L
    
    print("Inlet conditions:")
    print(f"[A]₀ = {A_inlet.concentration} mol/L")
    print(f"[B]₀ = {B_inlet.concentration} mol/L")
    print(f"[C]₀ = {C_inlet.concentration} mol/L")
    
    # Define reaction: A -> B + C (first order)
    reaction = Reaction("A -> B + C")
    reaction.set_rate_expression("k * [A]")
    reaction.add_rate_parameter('k', 0.5)  # 1/s
    
    print(f"\nReaction: {reaction.reaction_string}")
    print(f"Rate law: {reaction.rate_expression}")  
    print(f"Rate constant k = {reaction.rate_parameters['k']} 1/s")
    
    # PFR design parameters
    reactor_length = 5.0    # m
    reactor_diameter = 0.2  # m
    cross_section_area = np.pi * (reactor_diameter/2)**2  # m²
    reactor_volume = cross_section_area * reactor_length   # m³
    volumetric_flow = 0.01  # m³/s
    velocity = volumetric_flow / cross_section_area        # m/s
    residence_time = reactor_volume / volumetric_flow      # s
    
    print(f"\nPFR design:")
    print(f"Length: {reactor_length} m")
    print(f"Diameter: {reactor_diameter} m") 
    print(f"Volume: {reactor_volume:.4f} m³")
    print(f"Flow rate: {volumetric_flow} m³/s")
    print(f"Velocity: {velocity:.3f} m/s")
    print(f"Residence time: {residence_time:.1f} s")
    
    # Create PFR
    pfr = PFR("Decomposition PFR", 
              length=reactor_length,
              cross_sectional_area=cross_section_area,
              flow_rate=volumetric_flow)
    
    # Add species and reaction
    A = Species("A", concentration=A_inlet.concentration)
    B = Species("B", concentration=B_inlet.concentration)
    C = Species("C", concentration=C_inlet.concentration)
    
    pfr.add_species([A, B, C])
    pfr.add_reaction(reaction)
    
    # Simulate along reactor length
    print("\nSimulating PFR...")
    results = pfr.simulate(length_span=(0, reactor_length), num_points=50)
    
    # Extract results
    positions = results.t  # Position along reactor (m)
    concentrations = results.y  # Concentrations vs position
    
    species_names = list(pfr.species.keys())
    A_profile = concentrations[0, :]  # [A] vs position
    B_profile = concentrations[1, :]  # [B] vs position
    C_profile = concentrations[2, :]  # [C] vs position
    
    # Calculate conversion and selectivity
    conversion_profile = (A_inlet.concentration - A_profile) / A_inlet.concentration * 100
    final_conversion = conversion_profile[-1]
    
    print(f"\nResults at reactor exit (L = {reactor_length} m):")
    print(f"[A] = {A_profile[-1]:.4f} mol/L")
    print(f"[B] = {B_profile[-1]:.4f} mol/L")
    print(f"[C] = {C_profile[-1]:.4f} mol/L")
    print(f"Conversion: {final_conversion:.1f}%")
    
    # Verify stoichiometry (A -> B + C, so B = C and A₀ = A + B + C)
    mass_balance_check = A_profile + B_profile + C_profile
    print(f"\nMass balance check (should be constant = {A_inlet.concentration}):")
    print(f"A + B + C at inlet: {mass_balance_check[0]:.4f}")
    print(f"A + B + C at exit: {mass_balance_check[-1]:.4f}")
    
    # Plot spatial concentration profiles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Concentration profiles
    ax1.plot(positions, A_profile, 'r-', linewidth=2, label='[A]', marker='o', markersize=4)
    ax1.plot(positions, B_profile, 'b-', linewidth=2, label='[B]', marker='s', markersize=4)
    ax1.plot(positions, C_profile, 'g-', linewidth=2, label='[C]', marker='^', markersize=4)
    
    ax1.set_xlabel('Reactor Position (m)')
    ax1.set_ylabel('Concentration (mol/L)')
    ax1.set_title('PFR: Concentration Profiles Along Reactor Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, reactor_length)
    
    # Plot 2: Conversion profile
    ax2.plot(positions, conversion_profile, 'purple', linewidth=3, marker='o', markersize=5)
    ax2.set_xlabel('Reactor Position (m)')
    ax2.set_ylabel('Conversion (%)')
    ax2.set_title('PFR: Conversion of A Along Reactor Length')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, reactor_length)
    ax2.set_ylim(0, 100)
    
    # Add annotations
    ax2.annotate(f'Final conversion: {final_conversion:.1f}%',
                xy=(reactor_length, final_conversion),
                xytext=(reactor_length*0.7, final_conversion*0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig('pfr_spatial_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Comparison with equivalent CSTR
    print(f"\n" + "="*50)
    print("Comparison with equivalent CSTR")
    print("="*50)
    
    # For first-order reaction: X_CSTR = (k*τ)/(1 + k*τ)
    k = reaction.rate_parameters['k']
    tau = residence_time
    X_cstr_theoretical = (k * tau) / (1 + k * tau) * 100
    
    # For PFR: X_PFR = 1 - exp(-k*τ)
    X_pfr_theoretical = (1 - np.exp(-k * tau)) * 100
    
    print(f"Theoretical comparison (k = {k} 1/s, τ = {tau:.1f} s):")
    print(f"CSTR conversion: {X_cstr_theoretical:.1f}%")
    print(f"PFR conversion: {X_pfr_theoretical:.1f}%")
    print(f"Simulated PFR conversion: {final_conversion:.1f}%")
    print(f"PFR advantage: {final_conversion - X_cstr_theoretical:.1f} percentage points")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    # Calculate theoretical profiles for comparison
    z_theory = np.linspace(0, reactor_length, 100)
    tau_local = z_theory / velocity  # Local residence time
    conversion_theory = (1 - np.exp(-k * tau_local)) * 100
    
    plt.plot(positions, conversion_profile, 'ro-', linewidth=2, markersize=6, 
             label='Simulated PFR', markevery=5)
    plt.plot(z_theory, conversion_theory, 'r--', linewidth=2, 
             label='Theoretical PFR')
    plt.axhline(X_cstr_theoretical, color='blue', linestyle='-', linewidth=2,
                label=f'Equivalent CSTR ({X_cstr_theoretical:.1f}%)')
    
    plt.xlabel('Reactor Position (m)')
    plt.ylabel('Conversion (%)')
    plt.title('PFR vs CSTR Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, reactor_length)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('pfr_vs_cstr_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPlots saved:")
    print("- pfr_spatial_profiles.png")
    print("- pfr_vs_cstr_comparison.png")
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()
