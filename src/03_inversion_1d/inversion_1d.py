#!/usr/bin/env python3
"""
Script to perform 1D inversion of magnetotelluric data
Generates a layered subsurface model with real resistivities

1D inversion assumes the subsurface is composed of infinite horizontal layers,
each with its own resistivity.

Processes multiple stations and generates individual reports for each.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mtpy.core.mt import MT
from pathlib import Path

# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))
EDI_DIR = TFG_DIR / "data" / "edi"   # one EDI file per station (see data/README.md)

base_dir = str(EDI_DIR)
output_dir = str(TFG_DIR)

# List of EDI files to process
edi_files = [
    os.path.join(base_dir, 'mall01.edi'),
    os.path.join(base_dir, 'mall10.edi'),
    os.path.join(base_dir, 'mall28.edi'),
    os.path.join(base_dir, 'mall34.edi'),
    os.path.join(base_dir, 'mall55.edi'),
    os.path.join(base_dir, 'mall67.edi'),
]

print("="*80)
print("1D INVERSION OF MAGNETOTELLURIC DATA - Multiple Stations")
print("="*80)
print(f"\nProcessing {len(edi_files)} stations:")
for edi_file in edi_files:
    print(f"  - {os.path.basename(edi_file)}")
print("="*80)

# Function to calculate 1D MT forward response using Wait's recursion
def mt_1d_forward(periods, resistivities, thicknesses):
    """
    Calculate 1D MT forward response using Wait's recursion

    Parameters:
        periods: array of periods (s)
        resistivities: array of layer resistivities (Ohm*m)
        thicknesses: array of layer thicknesses (m)

    Returns:
        rho_app: apparent resistivities
    """
    mu = 4 * np.pi * 1e-7  # Magnetic permeability
    rho_app = np.zeros(len(periods))

    for i, T in enumerate(periods):
        omega = 2 * np.pi / T

        # Start from bottom layer (half-space)
        n_layers = len(resistivities)
        Z = np.sqrt(1j * omega * mu * resistivities[-1])

        # Recursion from bottom to top
        for j in range(n_layers - 2, -1, -1):
            rho_j = resistivities[j]
            h_j = thicknesses[j]

            # Propagation constant
            k_j = np.sqrt(1j * omega * mu / rho_j)

            # Intrinsic impedance
            Z0_j = np.sqrt(1j * omega * mu * rho_j)

            # Reflection coefficient
            r = (Z - Z0_j) / (Z + Z0_j)

            # Update impedance
            Z = Z0_j * (1 + r * np.exp(-2 * k_j * h_j)) / \
                       (1 - r * np.exp(-2 * k_j * h_j))

        # Calculate apparent resistivity
        rho_app[i] = (1 / (omega * mu)) * np.abs(Z)**2

    return rho_app


# Function to process a single station
def process_station(edi_file, output_dir):
    """Process one EDI file and generate 1D inversion results"""

    station_name = os.path.splitext(os.path.basename(edi_file))[0]

    print("\n" + "="*80)
    print(f"PROCESSING STATION: {station_name}")
    print("="*80)

    # Read EDI file with MTpy
    print("\n1. Reading EDI file with MTpy...")
    mt_obj = MT()
    mt_obj.read(edi_file)

    print(f"   Station: {mt_obj.station}")
    print(f"   Coordinates: {mt_obj.latitude:.6f}N, {mt_obj.longitude:.6f}E")
    print(f"   Number of frequencies: {len(mt_obj.frequency)}")
    print(f"   Frequency range: {mt_obj.frequency.min():.4f} - {mt_obj.frequency.max():.1f} Hz")

    # Method: Bostick transformation (fast 1D model approximation)
    print("\n2. Using Bostick transformation to approximate 1D model...")
    print("   (Fast method without need for external inversion software)")

    # Alternative method: Analyze apparent resistivity vs depth
    # This gives a 1D model approximation without formal inversion
    print("\n" + "="*80)
    print("ALTERNATIVE ANALYSIS: Apparent resistivity vs depth")
    print("="*80)

    # Calculate apparent resistivities
    freq = mt_obj.frequency
    periodo = 1.0 / freq

    # Get impedances from mtpy
    # mtpy handles units internally
    Z = mt_obj.Z

    # Use mtpy methods to calculate apparent resistivity
    # These already handle unit conversions correctly
    rho_xy = Z.resistivity[:, 0, 1]  # TE mode (xy)
    rho_yx = Z.resistivity[:, 1, 0]  # TM mode (yx)

    # Average (Bostick approximation)
    rho_avg = np.sqrt(rho_xy * rho_yx)

    # ============ FILTER ANOMALOUS FREQUENCIES ============
    print("\n3. Filtering anomalous frequencies...")

# Criteria to detect outliers:
# 1. Large difference between TE and TM modes
# 2. Extreme values outside reasonable range
# 3. NaN or Inf values

# Create mask for valid data
valid_mask = np.ones(len(freq), dtype=bool)

# Remove NaN or Inf
valid_mask &= np.isfinite(rho_xy) & np.isfinite(rho_yx)

# Remove extreme values (outside 0.1 to 10000 Ohm*m)
valid_mask &= (rho_avg > 0.1) & (rho_avg < 10000)

# Remove frequencies where TE and TM differ by more than factor of 10
ratio_te_tm = rho_xy / rho_yx
valid_mask &= (ratio_te_tm > 0.1) & (ratio_te_tm < 10)

# Apply filter
freq_filtered = freq[valid_mask]
periodo_filtered = periodo[valid_mask]
rho_xy_filtered = rho_xy[valid_mask]
rho_yx_filtered = rho_yx[valid_mask]
rho_avg_filtered = rho_avg[valid_mask]

n_removed = np.sum(~valid_mask)
print(f"   Removed {n_removed} anomalous frequencies out of {len(freq)}")
print(f"   Valid frequencies: {len(freq_filtered)}")

# Use filtered data for further calculations
freq = freq_filtered
periodo = periodo_filtered
rho_xy = rho_xy_filtered
rho_yx = rho_yx_filtered
rho_avg = rho_avg_filtered

# Calculate penetration depth (skin depth)
# Approximation: characteristic depth where signal has decreased to 1/e
depth = 503 * np.sqrt(rho_avg / freq)

print(f"\nAverage apparent resistivity statistics:")
print(f"  Minimum: {rho_avg.min():.2f} Ohm*m")
print(f"  Maximum: {rho_avg.max():.2f} Ohm*m")
print(f"  Mean: {rho_avg.mean():.2f} Ohm*m")
print(f"  Median: {np.median(rho_avg):.2f} Ohm*m")

print(f"\nInvestigated depth range:")
print(f"  Minimum: {depth.min():.1f} m")
print(f"  Maximum: {depth.max():.1f} m ({depth.max()/1000:.2f} km)")

# Simple Bostick transformation
# Fast 1D model approximation from apparent resistivity
print("\n" + "="*80)
print("BOSTICK TRANSFORMATION (1D model approximation)")
print("="*80)

# Bostick transformation relates derivative of rho_app with depth
# rho(z) ≈ rho_app(T) * [1 + (2/rho_app) * d(rho_app)/d(ln(T))]

# Calculate logarithmic derivative
d_log_rho = np.gradient(np.log(rho_avg))
d_log_T = np.gradient(np.log(periodo))
deriv = d_log_rho / d_log_T

# Bostick resistivity
# Limit derivative to avoid negative or extreme values
deriv_limited = np.clip(deriv, -0.8, 2.0)
rho_bostick = rho_avg * (1 + deriv_limited)

# Ensure positive values
rho_bostick = np.maximum(rho_bostick, 1.0)

# Bostick depth
depth_bostick = depth / np.sqrt(2)

print("\nApproximate 1D model (Bostick):")
print("\nDepth [m]  |  Resistivity [Ohm*m]")
print("-" * 45)

# Sort by depth
sort_idx = np.argsort(depth_bostick)
for i in sort_idx[::3]:  # Show every 3 points to avoid saturation
    print(f"{depth_bostick[i]:10.1f}      |  {rho_bostick[i]:10.2f}")

# ============ CALCULATE FORWARD RESPONSE ============
print("\n" + "="*80)
print("CALCULATING FORWARD RESPONSE")
print("="*80)

# Step 1: Create a discretized layered model from Bostick transformation
# Sort by depth and create layers
sort_idx = np.argsort(depth_bostick)
depth_sorted_model = depth_bostick[sort_idx]
rho_sorted_model = rho_bostick[sort_idx]

# Create layer model: define layer thicknesses and resistivities
# Use a subset of points to create a reasonable number of layers (max 30)
n_layers = min(30, len(depth_sorted_model))
indices = np.linspace(0, len(depth_sorted_model)-1, n_layers, dtype=int)

layer_depths = depth_sorted_model[indices]
layer_resistivities = rho_sorted_model[indices]

# Calculate layer thicknesses
layer_thicknesses = np.diff(layer_depths)
# Add a thick bottom layer (half-space)
layer_thicknesses = np.append(layer_thicknesses, 100000)  # Very thick bottom layer

print(f"\nCreated layered model with {n_layers} layers")
print(f"Depth range: 0 - {layer_depths[-1]:.1f} m")
print(f"Resistivity range: {layer_resistivities.min():.2f} - {layer_resistivities.max():.2f} Ohm*m")

# Step 2: Calculate forward response using 1D MT modeling
# Use Wait's recursion algorithm for 1D MT
def mt_1d_forward(periods, resistivities, thicknesses):
    """
    Calculate 1D MT forward response using Wait's recursion

    Parameters:
        periods: array of periods (s)
        resistivities: array of layer resistivities (Ohm*m)
        thicknesses: array of layer thicknesses (m), last value is ignored (half-space)

    Returns:
        rho_app: apparent resistivities
    """
    mu = 4 * np.pi * 1e-7  # Magnetic permeability
    rho_app = np.zeros(len(periods))

    for i, T in enumerate(periods):
        omega = 2 * np.pi / T

        # Start from bottom layer (half-space)
        n_layers = len(resistivities)
        Z = np.sqrt(1j * omega * mu * resistivities[-1])  # Impedance of half-space

        # Recursion from bottom to top
        for j in range(n_layers - 2, -1, -1):
            rho_j = resistivities[j]
            h_j = thicknesses[j]

            # Propagation constant
            k_j = np.sqrt(1j * omega * mu / rho_j)

            # Intrinsic impedance
            Z0_j = np.sqrt(1j * omega * mu * rho_j)

            # Reflection coefficient
            r = (Z - Z0_j) / (Z + Z0_j)

            # Update impedance for next layer up
            Z = Z0_j * (1 + r * np.exp(-2 * k_j * h_j)) / (1 - r * np.exp(-2 * k_j * h_j))

        # Calculate apparent resistivity from surface impedance
        rho_app[i] = (1 / (omega * mu)) * np.abs(Z)**2

    return rho_app

# Calculate synthetic response
print("\nCalculating 1D MT forward response...")
rho_synthetic = mt_1d_forward(periodo, layer_resistivities, layer_thicknesses)

print(f"Forward response calculated for {len(rho_synthetic)} periods")
print(f"Synthetic resistivity range: {rho_synthetic.min():.2f} - {rho_synthetic.max():.2f} Ohm*m")

# Calculate RMS misfit between observed and synthetic
rms_misfit = np.sqrt(np.mean(((np.log10(rho_avg) - np.log10(rho_synthetic))**2)))
print(f"RMS misfit (log scale): {rms_misfit:.3f}")

# Plot 1D model approximation
print("\n" + "="*80)
print("GENERATING 1D MODEL GRAPHICS")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Graph 1: Apparent Resistivity TE and TM modes
ax1 = axes[0]
ax1.loglog(periodo, rho_xy, 'o-', color='blue', label='rho_xy (TE)',
           markersize=5, linewidth=1.5, markeredgecolor='darkblue')
ax1.loglog(periodo, rho_yx, 's-', color='red', label='rho_yx (TM)',
           markersize=5, linewidth=1.5, markeredgecolor='darkred')

ax1.set_xlabel('Period (s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Apparent Resistivity (Ohm*m)', fontsize=12, fontweight='bold')
ax1.set_title('Observed Data: TE and TM Modes', fontsize=13, fontweight='bold')
ax1.grid(True, which='both', alpha=0.4)
ax1.legend(loc='best', fontsize=10)

# Graph 2: Observed vs Synthetic (Forward Response)
ax2 = axes[1]
ax2.loglog(periodo, rho_avg, 'd-', color='green', label='rho_average (observed)',
           markersize=6, linewidth=2.5, markeredgecolor='darkgreen')
ax2.loglog(periodo, rho_synthetic, 'x--', color='magenta', label='rho_synthetic (forward)',
           markersize=7, linewidth=2.5, markeredgecolor='darkmagenta')

ax2.set_xlabel('Period (s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Apparent Resistivity (Ohm*m)', fontsize=12, fontweight='bold')
ax2.set_title(f'Model Fit (RMS: {rms_misfit:.3f})', fontsize=13, fontweight='bold')
ax2.grid(True, which='both', alpha=0.4)
ax2.legend(loc='best', fontsize=10)

# Graph 3: 1D Model (Resistivity vs Depth) - Professional style
ax3 = axes[2]

# Plot as step function (layered model)
# Sort by depth
sort_idx = np.argsort(depth_bostick)
depth_sorted = depth_bostick[sort_idx]
rho_sorted = rho_bostick[sort_idx]

ax3.step(rho_sorted, depth_sorted, where='post', color='darkblue',
         linewidth=2.5, label='Bostick 1D Model')
ax3.scatter(rho_sorted, depth_sorted, color='red', s=30,
            zorder=5, alpha=0.6, edgecolor='darkred')

ax3.set_xscale('log')
ax3.set_xlabel('Resistivity (Ohm*m)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax3.set_title('1D Subsurface Model (Bostick Transformation)',
              fontsize=13, fontweight='bold')
ax3.set_ylim([5000, 0])  # Limit depth to 5000 m, with 0 at top
ax3.grid(True, which='both', alpha=0.4)
ax3.legend(loc='best', fontsize=10)

# Add interpreted layer annotations
# Identify significant resistivity changes
rho_ratio = rho_sorted[1:] / rho_sorted[:-1]
significant_changes = np.where(np.abs(np.log10(rho_ratio)) > 0.3)[0]

if len(significant_changes) > 0:
    print("\nMain layers detected (significant changes):")
    print("\nDepth [m]  |  Resistivity [Ohm*m]  |  Interpretation")
    print("-" * 70)

    for idx in significant_changes[:5]:  # Show up to 5 main layers
        d = depth_sorted[idx]
        r = rho_sorted[idx]

        # Basic geological interpretation
        if r < 10:
            interp = "Saturated clays / Conductive sediments"
        elif r < 50:
            interp = "Sediments / Altered rocks"
        elif r < 200:
            interp = "Sedimentary rocks / Limestones"
        elif r < 1000:
            interp = "Consolidated rocks"
        else:
            interp = "Resistive basement"

        print(f"{d:10.1f}      |  {r:15.2f}       |  {interp}")

plt.tight_layout()

# Save figure
output_file = os.path.join(output_dir, '1d_bostick_model.png')
fig.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nGraph saved at: {output_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETED")
print("="*80)
print(f"\nModel quality (RMS misfit): {rms_misfit:.3f}")
print("  (Lower is better. < 0.3 is good, < 0.5 is acceptable)")
print("\nNOTE: This is an approximate 1D model using Bostick transformation.")
print("For a more precise model, formal inversion with Occam1D or ModEM")
print("is required, which needs additional software.")
print("\nThe model shows how resistivity varies with depth,")
print("assuming horizontal layers. Low frequencies penetrate deeper.")
print("\nThe magenta curve (synthetic) shows the forward response from the")
print("Bostick model. If it matches the green curve (observed), the model is good.")

if __name__ == '__main__':
    # Don't show interactive plots (only save)
    # plt.show()
    pass
