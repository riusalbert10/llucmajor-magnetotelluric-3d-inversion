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

# Detect operating system for paths
if sys.platform == 'win32':
    base_dir = r'C:\Users\alber\TFG\LLUCMAJOR_DADES_edi'
    output_dir = r'C:\Users\alber\TFG\1D_Inversion_Results'
else:
    base_dir = '/mnt/c/Users/alber/TFG/LLUCMAJOR_DADES_edi'
    output_dir = '/home/alber/1D_Inversion_Results'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List of EDI files to process
edi_files = [
    os.path.join(base_dir, 'mall01.edi'),
    os.path.join(base_dir, 'mall10.edi'),
    os.path.join(base_dir, 'mall28.edi'),
    os.path.join(base_dir, 'mall34.edi'),
    os.path.join(base_dir, 'mall55.edi'),
    os.path.join(base_dir, 'mall67.edi'),
]


def calculate_depth_limit(max_skin_depth):
    """
    Calculate the depth limit for plotting based on maximum skin depth.

    Rules:
    - If max_skin_depth > 5000m: limit to 5000m
    - If max_skin_depth < 5000m: limit to max_skin_depth + 200m

    This ensures the plot shows relevant data without excessive empty space.
    """
    if max_skin_depth > 5000:
        return 5000
    else:
        return max_skin_depth + 200


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


def process_station(edi_file, output_dir):
    """Process one EDI file and generate 1D inversion results"""

    station_name = os.path.splitext(os.path.basename(edi_file))[0]

    print("\n" + "="*80)
    print(f"PROCESSING STATION: {station_name}")
    print("="*80)

    try:
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

        print("\n" + "="*80)
        print("ALTERNATIVE ANALYSIS: Apparent resistivity vs depth")
        print("="*80)

        # Calculate apparent resistivities
        freq = mt_obj.frequency
        periodo = 1.0 / freq

        # Get impedances from mtpy
        Z = mt_obj.Z

        # Use mtpy methods to calculate apparent resistivity
        rho_xy = Z.resistivity[:, 0, 1]  # TE mode (xy)
        rho_yx = Z.resistivity[:, 1, 0]  # TM mode (yx)

        # Average (Bostick approximation)
        rho_avg = np.sqrt(rho_xy * rho_yx)

        # ============ FILTER ANOMALOUS FREQUENCIES ============
        print("\n3. Filtering anomalous frequencies...")

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

        # Check if we have enough data left
        if len(freq_filtered) < 5:
            print(f"   WARNING: Too few valid frequencies ({len(freq_filtered)}). Skipping station.")
            return

        # Use filtered data for further calculations
        freq = freq_filtered
        periodo = periodo_filtered
        rho_xy = rho_xy_filtered
        rho_yx = rho_yx_filtered
        rho_avg = rho_avg_filtered

        # Calculate penetration depth (skin depth)
        depth = 503 * np.sqrt(rho_avg / freq)

        # Calculate automatic depth limit based on maximum skin depth
        max_skin_depth = depth.max()
        depth_limit = calculate_depth_limit(max_skin_depth)

        print(f"\nAverage apparent resistivity statistics:")
        print(f"  Minimum: {rho_avg.min():.2f} Ohm*m")
        print(f"  Maximum: {rho_avg.max():.2f} Ohm*m")
        print(f"  Mean: {rho_avg.mean():.2f} Ohm*m")
        print(f"  Median: {np.median(rho_avg):.2f} Ohm*m")

        print(f"\nInvestigated depth range:")
        print(f"  Minimum: {depth.min():.1f} m")
        print(f"  Maximum: {depth.max():.1f} m ({depth.max()/1000:.2f} km)")
        print(f"  Plot depth limit: {depth_limit:.1f} m (auto-adjusted)")

        # Simple Bostick transformation
        print("\n" + "="*80)
        print("BOSTICK TRANSFORMATION (1D model approximation)")
        print("="*80)

        # Calculate logarithmic derivative
        d_log_rho = np.gradient(np.log(rho_avg))
        d_log_T = np.gradient(np.log(periodo))
        deriv = d_log_rho / d_log_T

        # Apply Bostick transformation with derivative limiting
        deriv_limited = np.clip(deriv, -0.8, 2.0)
        rho_bostick = rho_avg * (1 + deriv_limited)

        # Ensure positive values
        rho_bostick = np.maximum(rho_bostick, 1.0)

        # Calculate Bostick depth
        depth_bostick = depth / np.sqrt(2)

        print("\nApproximate 1D model (Bostick):")
        print("\nDepth [m]  |  Resistivity [Ohm*m]")
        print("-" * 45)

        # Sort by depth
        sort_idx = np.argsort(depth_bostick)
        for i in sort_idx[::3]:  # Show every 3 points
            print(f"{depth_bostick[i]:10.1f}      |  {rho_bostick[i]:10.2f}")

        # ============ CALCULATE FORWARD RESPONSE ============
        print("\n" + "="*80)
        print("CALCULATING FORWARD RESPONSE")
        print("="*80)

        # Sort by depth and create layers
        sort_idx = np.argsort(depth_bostick)
        depth_sorted_model = depth_bostick[sort_idx]
        rho_sorted_model = rho_bostick[sort_idx]

        # Create layer model
        n_layers = min(30, len(depth_sorted_model))
        indices = np.linspace(0, len(depth_sorted_model)-1, n_layers, dtype=int)

        layer_depths = depth_sorted_model[indices]
        layer_resistivities = rho_sorted_model[indices]

        # Calculate layer thicknesses
        layer_thicknesses = np.diff(layer_depths)
        layer_thicknesses = np.append(layer_thicknesses, 100000)  # Bottom layer

        print(f"\nCreated layered model with {n_layers} layers")
        print(f"Depth range: 0 - {layer_depths[-1]:.1f} m")
        print(f"Resistivity range: {layer_resistivities.min():.2f} - {layer_resistivities.max():.2f} Ohm*m")

        # Calculate synthetic response
        print("\nCalculating 1D MT forward response...")
        rho_synthetic = mt_1d_forward(periodo, layer_resistivities, layer_thicknesses)

        print(f"Forward response calculated for {len(rho_synthetic)} periods")
        print(f"Synthetic resistivity range: {rho_synthetic.min():.2f} - {rho_synthetic.max():.2f} Ohm*m")

        # Calculate RMS misfit
        rms_misfit = np.sqrt(np.mean(((np.log10(rho_avg) - np.log10(rho_synthetic))**2)))
        print(f"RMS misfit (log scale): {rms_misfit:.3f}")

        # ============ GENERATE PLOTS ============
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

        # Graph 2: Observed vs Synthetic
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

        # Graph 3: 1D Model
        ax3 = axes[2]

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

        # Use the automatically calculated depth limit
        ax3.set_ylim([depth_limit, 0])

        ax3.grid(True, which='both', alpha=0.4)
        ax3.legend(loc='best', fontsize=10)

        # Add station info to title
        fig.suptitle(f'1D Inversion - Station {station_name} | RMS Misfit: {rms_misfit:.3f}',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save figure
        output_file = os.path.join(output_dir, f'1d_bostick_model_{station_name}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nGraph saved at: {output_file}")

        # Close figure to free memory
        plt.close(fig)

        # Print layer interpretation
        rho_ratio = rho_sorted[1:] / rho_sorted[:-1]
        significant_changes = np.where(np.abs(np.log10(rho_ratio)) > 0.3)[0]

        if len(significant_changes) > 0:
            print("\nMain layers detected (significant changes):")
            print("\nDepth [m]  |  Resistivity [Ohm*m]  |  Interpretation")
            print("-" * 70)

            for idx in significant_changes[:5]:
                d = depth_sorted[idx]
                r = rho_sorted[idx]

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

        print("\n" + "="*80)
        print(f"STATION {station_name} COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nModel quality (RMS misfit): {rms_misfit:.3f}")
        print("  (Lower is better. < 0.3 is good, < 0.5 is acceptable)")

        return True

    except Exception as e:
        print(f"\n*** ERROR processing {station_name}: {str(e)}")
        print(f"*** Skipping this station and continuing with others...")
        return False


# ============ MAIN EXECUTION ============
if __name__ == '__main__':
    print("="*80)
    print("1D MAGNETOTELLURIC INVERSION - MULTIPLE STATIONS")
    print("="*80)
    print(f"\nProcessing {len(edi_files)} stations:")
    for edi_file in edi_files:
        print(f"  - {os.path.basename(edi_file)}")
    print(f"\nDepth limits will be automatically adjusted based on each station's skin depth.")
    print(f"  Rule: If max_skin_depth > 5000m → limit to 5000m")
    print(f"        If max_skin_depth < 5000m → limit to max_skin_depth + 200m")
    print(f"\nResults will be saved in: {output_dir}")
    print("="*80)

    # Process each station
    successful = 0
    failed = 0

    for edi_file in edi_files:
        if not os.path.exists(edi_file):
            print(f"\n*** WARNING: File not found: {edi_file}")
            print("*** Skipping...")
            failed += 1
            continue

        result = process_station(edi_file, output_dir)
        if result:
            successful += 1
        else:
            failed += 1

    # Final summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*80)
    print(f"\nTotal stations processed: {len(edi_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nAll results saved in: {output_dir}")
    print("="*80)
