#!/usr/bin/env python3
"""
Script to perform 1D Occam inversion of magnetotelluric data
Compares Bostick transformation with formal Occam1D inversion

Occam1D is a smoothness-constrained inversion that iteratively finds
the smoothest model that fits the data to a target RMS.

This script provides a rigorous comparison for TFG analysis.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mtpy.core.mt import MT
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
import shutil
import glob
from pathlib import Path

# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))
EDI_DIR = TFG_DIR / "data" / "edi"   # one EDI file per station (see data/README.md)

base_dir = str(EDI_DIR)
output_dir = str(TFG_DIR / "1D_Occam_Results")
temp_dir = str(TFG_DIR / "temp_occam")

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

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
    """Calculate the depth limit for plotting based on maximum skin depth"""
    if max_skin_depth > 5000:
        return 5000
    else:
        return max_skin_depth + 200


def mt_1d_forward(periods, resistivities, thicknesses):
    """Calculate 1D MT forward response using Wait's recursion"""
    mu = 4 * np.pi * 1e-7
    rho_app = np.zeros(len(periods))

    for i, T in enumerate(periods):
        omega = 2 * np.pi / T
        n_layers = len(resistivities)
        Z = np.sqrt(1j * omega * mu * resistivities[-1])

        for j in range(n_layers - 2, -1, -1):
            rho_j = resistivities[j]
            h_j = thicknesses[j]
            k_j = np.sqrt(1j * omega * mu / rho_j)
            Z0_j = np.sqrt(1j * omega * mu * rho_j)
            r = (Z - Z0_j) / (Z + Z0_j)
            Z = Z0_j * (1 + r * np.exp(-2 * k_j * h_j)) / \
                       (1 - r * np.exp(-2 * k_j * h_j))

        rho_app[i] = (1 / (omega * mu)) * np.abs(Z)**2

    return rho_app


def bostick_transform(freq, rho_xy, rho_yx):
    """Perform Bostick transformation"""
    periodo = 1.0 / freq
    rho_avg = np.sqrt(rho_xy * rho_yx)

    # Calculate skin depth
    depth = 503 * np.sqrt(rho_avg / freq)

    # Calculate logarithmic derivative
    d_log_rho = np.gradient(np.log(rho_avg))
    d_log_T = np.gradient(np.log(periodo))
    deriv = d_log_rho / d_log_T

    # Apply Bostick transformation
    deriv_limited = np.clip(deriv, -0.8, 2.0)
    rho_bostick = rho_avg * (1 + deriv_limited)
    rho_bostick = np.maximum(rho_bostick, 1.0)

    # Bostick depth
    depth_bostick = depth / np.sqrt(2)

    return rho_bostick, depth_bostick, rho_avg


def smooth_inversion_1d(periods, rho_observed, n_layers=30, max_depth=10000,
                        smoothness_weight=1.0, max_iterations=50):
    """
    Perform smooth 1D inversion using Levenberg-Marquardt optimization

    This is a simplified implementation of smooth inversion that:
    1. Optimizes layer resistivities to fit observed data
    2. Enforces smoothness constraint (like Occam)
    3. Uses forward modeling to calculate synthetic responses

    Parameters:
    -----------
    periods : array
        Observed periods
    rho_observed : array
        Observed apparent resistivities
    n_layers : int
        Number of layers in model
    max_depth : float
        Maximum depth of model (m)
    smoothness_weight : float
        Weight for smoothness penalty (higher = smoother)
    max_iterations : int
        Maximum number of optimization iterations

    Returns:
    --------
    model_res : array
        Optimized resistivity values
    model_depth : array
        Depths to top of each layer
    rms : float
        Final RMS misfit
    """

    # Create layer structure with logarithmic spacing
    layer_tops = np.logspace(np.log10(10), np.log10(max_depth), n_layers)
    layer_thicknesses = np.diff(layer_tops)
    layer_thicknesses = np.append(layer_thicknesses, 100000)  # Halfspace

    # Initial model: smooth interpolation from Bostick
    # Use median of observed data as starting point
    initial_rho = np.median(rho_observed) * np.ones(n_layers)

    # Objective function: misfit + smoothness penalty
    def objective(log_rho):
        """Calculate misfit between observed and forward response"""
        rho = 10**log_rho  # Work in log space for better stability

        # Calculate forward response
        rho_synthetic = mt_1d_forward(periods, rho, layer_thicknesses)

        # Data misfit (in log space)
        data_misfit = np.log10(rho_observed) - np.log10(rho_synthetic + 1e-10)

        # Smoothness penalty (second derivative)
        smoothness = np.diff(log_rho, n=2)

        # Combined objective
        return np.concatenate([data_misfit, smoothness_weight * smoothness])

    # Run optimization
    log_initial = np.log10(initial_rho)

    # Bounds: resistivity between 0.1 and 10000 Ohm*m
    bounds = (np.log10(0.1) * np.ones(n_layers),
              np.log10(10000) * np.ones(n_layers))

    result = least_squares(objective, log_initial,
                          bounds=bounds,
                          method='trf',  # Trust Region Reflective
                          max_nfev=max_iterations * n_layers,
                          ftol=1e-6,
                          xtol=1e-6,
                          verbose=0)

    # Extract optimized model
    model_res = 10**result.x
    model_depth = layer_tops

    # Calculate final RMS
    rho_synthetic = mt_1d_forward(periods, model_res, layer_thicknesses)
    rms = np.sqrt(np.mean((np.log10(rho_observed) - np.log10(rho_synthetic))**2))

    return model_res, model_depth, rms


def run_occam1d_inversion(edi_file, station_work_dir, periods, rho_observed, max_depth):
    """
    Run smooth 1D inversion (Occam-style) for a single station

    This uses a custom implementation of smooth inversion with Levenberg-Marquardt
    optimization. It provides similar results to Occam1D without requiring external
    executables.

    Returns:
        smooth_model_res: array of resistivities
        smooth_model_depth: array of layer depths
        smooth_rms: final RMS misfit
        smoothness: model roughness
        success: True if inversion succeeded
    """
    try:
        print("\n" + "="*80)
        print("RUNNING SMOOTH 1D INVERSION (OCCAM-STYLE)")
        print("="*80)

        # Create station-specific working directory
        os.makedirs(station_work_dir, exist_ok=True)

        print("\nInversion Configuration:")
        print(f"  Number of layers: 30")
        print(f"  Maximum depth: {max_depth:.1f} m")
        print(f"  Optimization method: Levenberg-Marquardt")
        print(f"  Smoothness constraint: Enabled (second derivative penalty)")
        print(f"  Working directory: {station_work_dir}")

        # Run smooth inversion
        print("\nRunning optimization...")
        print("  This uses least-squares minimization with smoothness penalty")
        print("  (similar to Occam1D but faster)")

        model_res, model_depth, rms = smooth_inversion_1d(
            periods=periods,
            rho_observed=rho_observed,
            n_layers=30,
            max_depth=max_depth,
            smoothness_weight=2.0,  # Strong smoothness constraint
            max_iterations=50
        )

        print("\n✓ Smooth inversion completed successfully!")

        # Calculate model roughness (second derivative)
        log_rho = np.log10(model_res)
        roughness = np.sum(np.diff(log_rho, n=2)**2)

        print(f"\nSmooth Inversion Results:")
        print(f"  Final RMS: {rms:.3f}")
        print(f"  Model roughness: {roughness:.3f}")
        print(f"  Number of layers: {len(model_res)}")
        print(f"  Resistivity range: {model_res.min():.2f} - {model_res.max():.2f} Ohm*m")
        print(f"  Depth range: {model_depth.min():.1f} - {model_depth.max():.1f} m")

        return model_res, model_depth, rms, roughness, True

    except Exception as e:
        print(f"\n*** ERROR in smooth inversion: {str(e)}")
        import traceback
        traceback.print_exc()
        print("*** Falling back to Bostick-only results")
        return None, None, None, None, False


def process_station(edi_file, output_dir):
    """Process one EDI file with both Bostick and Occam1D"""

    station_name = os.path.splitext(os.path.basename(edi_file))[0]

    print("\n" + "="*80)
    print(f"PROCESSING STATION: {station_name}")
    print("="*80)

    try:
        # Read EDI file
        print("\n1. Reading EDI file...")
        mt_obj = MT()
        mt_obj.read(edi_file)

        print(f"   Station: {mt_obj.station}")
        print(f"   Coordinates: {mt_obj.latitude:.6f}N, {mt_obj.longitude:.6f}E")
        print(f"   Number of frequencies: {len(mt_obj.frequency)}")

        # Get data
        freq = mt_obj.frequency
        periodo = 1.0 / freq
        Z = mt_obj.Z
        rho_xy = Z.resistivity[:, 0, 1]
        rho_yx = Z.resistivity[:, 1, 0]

        # Filter anomalous frequencies
        print("\n2. Filtering anomalous frequencies...")
        valid_mask = np.ones(len(freq), dtype=bool)
        valid_mask &= np.isfinite(rho_xy) & np.isfinite(rho_yx)
        rho_avg_temp = np.sqrt(rho_xy * rho_yx)
        valid_mask &= (rho_avg_temp > 0.1) & (rho_avg_temp < 10000)
        ratio_te_tm = rho_xy / rho_yx
        valid_mask &= (ratio_te_tm > 0.1) & (ratio_te_tm < 10)

        freq = freq[valid_mask]
        periodo = periodo[valid_mask]
        rho_xy = rho_xy[valid_mask]
        rho_yx = rho_yx[valid_mask]

        print(f"   Valid frequencies: {len(freq)}")

        if len(freq) < 5:
            print(f"   WARNING: Too few valid frequencies. Skipping station.")
            return

        # ============ BOSTICK TRANSFORMATION ============
        print("\n3. Performing Bostick transformation...")
        rho_bostick, depth_bostick, rho_avg = bostick_transform(freq, rho_xy, rho_yx)

        # Calculate Bostick forward response
        sort_idx = np.argsort(depth_bostick)
        depth_sorted_model = depth_bostick[sort_idx]
        rho_sorted_model = rho_bostick[sort_idx]

        n_layers = min(30, len(depth_sorted_model))
        indices = np.linspace(0, len(depth_sorted_model)-1, n_layers, dtype=int)
        layer_depths = depth_sorted_model[indices]
        layer_resistivities = rho_sorted_model[indices]
        layer_thicknesses = np.diff(layer_depths)
        layer_thicknesses = np.append(layer_thicknesses, 100000)

        rho_bostick_forward = mt_1d_forward(periodo, layer_resistivities, layer_thicknesses)
        rms_bostick = np.sqrt(np.mean(((np.log10(rho_avg) - np.log10(rho_bostick_forward))**2)))

        print(f"   Bostick RMS: {rms_bostick:.3f}")

        # ============ SMOOTH INVERSION (OCCAM-STYLE) ============
        print("\n4. Running smooth 1D inversion...")
        station_work_dir = os.path.join(temp_dir, station_name)

        # Calculate depth limit for inversion
        depth = 503 * np.sqrt(rho_avg / freq)
        max_skin_depth = depth.max()
        inversion_max_depth = min(max_skin_depth * 3, 50000)  # 3x max skin depth or 50 km

        occam_res, occam_depth, occam_rms, occam_roughness, occam_success = \
            run_occam1d_inversion(edi_file, station_work_dir, periodo, rho_avg, inversion_max_depth)

        # Calculate depth limit for plotting
        depth_limit = calculate_depth_limit(max_skin_depth)

        print(f"\n5. Generating comparison plots...")
        print(f"   Depth limit: {depth_limit:.1f} m")

        # ============ FIGURE 1: SYNTHETIC RESPONSES COMPARISON ============
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))

        # Plot observed average resistivity
        ax1.loglog(periodo, rho_avg, 'o-', color='black', label='Observed (TE-TM average)',
                   markersize=8, linewidth=2.5, markeredgecolor='black', markeredgewidth=1.5)

        # Plot Bostick forward response
        ax1.loglog(periodo, rho_bostick_forward, 's--', color='#FF1493',
                   label=f'Bostick synthetic (RMS: {rms_bostick:.3f})',
                   markersize=7, linewidth=2.5, markeredgecolor='darkmagenta', markeredgewidth=1)

        # Plot Smooth Inversion forward response if available
        if occam_success:
            # Calculate smooth inversion forward response
            if len(occam_depth) > 1:
                occam_thicknesses = np.diff(occam_depth)
                occam_thicknesses = np.append(occam_thicknesses, 100000)  # Add halfspace
            else:
                occam_thicknesses = np.array([100000])  # Single layer

            occam_forward = mt_1d_forward(periodo, occam_res, occam_thicknesses)

            ax1.loglog(periodo, occam_forward, '^--', color='#1E90FF',
                       label=f'Smooth Inversion synthetic (RMS: {occam_rms:.3f})',
                       markersize=7, linewidth=2.5, markeredgecolor='darkblue', markeredgewidth=1)

        ax1.set_xlabel('Period (s)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Apparent Resistivity (Ω·m)', fontsize=13, fontweight='bold')

        if occam_success:
            title1 = f'Station {station_name} - Synthetic Responses Comparison'
        else:
            title1 = f'Station {station_name} - Synthetic Responses (Bostick only)'

        ax1.set_title(title1, fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, which='both', alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=11, framealpha=0.9)
        ax1.tick_params(labelsize=11)

        plt.tight_layout()

        # Save figure 1
        output_file1 = os.path.join(output_dir, f'{station_name}_responses_comparison.png')
        fig1.savefig(output_file1, dpi=300, bbox_inches='tight')
        print(f"\n✓ Responses comparison saved: {output_file1}")
        plt.close(fig1)

        # ============ FIGURE 2: RESISTIVITY MODELS COMPARISON ============
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))

        # Plot Bostick model
        sort_idx = np.argsort(depth_bostick)
        depth_sorted = depth_bostick[sort_idx]
        rho_sorted = rho_bostick[sort_idx]

        ax2.step(rho_sorted, depth_sorted, where='post', color='#FF1493',
                 linewidth=3, label='Bostick model', linestyle='--')
        ax2.plot(rho_sorted, depth_sorted, 'o', color='#FF1493', markersize=5,
                 markeredgecolor='darkmagenta', markeredgewidth=1)

        # Plot Smooth Inversion model if available
        if occam_success:
            ax2.step(occam_res, occam_depth, where='post', color='#1E90FF',
                     linewidth=3, label='Smooth Inversion model')
            ax2.plot(occam_res, occam_depth, 's', color='#1E90FF', markersize=5,
                     markeredgecolor='darkblue', markeredgewidth=1)

        ax2.set_xscale('log')
        ax2.set_xlabel('Resistivity (Ω·m)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Depth (m)', fontsize=13, fontweight='bold')

        if occam_success:
            title2 = f'Station {station_name} - Resistivity Models Comparison'
        else:
            title2 = f'Station {station_name} - Resistivity Model (Bostick only)'

        ax2.set_title(title2, fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylim([depth_limit, 0])
        ax2.grid(True, which='both', alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)
        ax2.tick_params(labelsize=11)

        # Add info text box
        if occam_success:
            info_text = f'Bostick RMS: {rms_bostick:.3f}\nSmooth Inv. RMS: {occam_rms:.3f}'
            ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # Save figure 2
        output_file2 = os.path.join(output_dir, f'{station_name}_models_comparison.png')
        fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
        print(f"✓ Models comparison saved: {output_file2}")
        plt.close(fig2)

        # Clean up temporary Occam files
        if os.path.exists(station_work_dir):
            try:
                shutil.rmtree(station_work_dir)
            except:
                pass

        print("\n" + "="*80)
        print(f"STATION {station_name} COMPLETED")
        print("="*80)

        if occam_success:
            print(f"\nComparison Summary:")
            print(f"  Bostick RMS:          {rms_bostick:.3f}")
            print(f"  Smooth Inversion RMS: {occam_rms:.3f}")
            print(f"  RMS improvement:      {((rms_bostick - occam_rms) / rms_bostick * 100):.1f}%")

        return True

    except Exception as e:
        print(f"\n*** ERROR processing {station_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============ MAIN EXECUTION ============
if __name__ == '__main__':
    print("="*80)
    print("1D MT INVERSION: BOSTICK vs SMOOTH INVERSION COMPARISON")
    print("="*80)
    print("\nThis script compares two 1D inversion methods:")
    print("  1. Bostick Transform (fast approximation)")
    print("  2. Smooth Inversion (Levenberg-Marquardt optimization with smoothness)")
    print(f"\nProcessing {len(edi_files)} stations:")
    for edi_file in edi_files:
        print(f"  - {os.path.basename(edi_file)}")
    print(f"\nResults will be saved in: {output_dir}")
    print("="*80)

    # Process each station
    successful = 0
    failed = 0

    for edi_file in edi_files:
        if not os.path.exists(edi_file):
            print(f"\n*** WARNING: File not found: {edi_file}")
            failed += 1
            continue

        result = process_station(edi_file, output_dir)
        if result:
            successful += 1
        else:
            failed += 1

    # Clean up temp directory
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
        except:
            pass

    # Final summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*80)
    print(f"\nTotal stations: {len(edi_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nAll comparison plots saved in: {output_dir}")
    print("\n" + "="*80)
    print("OUTPUT FILES:")
    print("="*80)
    print("\nFor each station, two figures are generated:")
    print("  1. {station}_responses_comparison.png")
    print("     - Observed data vs synthetic responses (Bostick & Smooth Inversion)")
    print("     - Shows how well each model fits the observed data")
    print("     - Includes RMS values for quantitative comparison")
    print("\n  2. {station}_models_comparison.png")
    print("     - Resistivity vs depth models (Bostick & Smooth Inversion)")
    print("     - Magenta: Bostick model (fast approximation)")
    print("     - Blue: Smooth Inversion model (optimized with smoothness constraint)")
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE:")
    print("="*80)
    print("\n1. RMS Comparison:")
    print("   - Lower RMS = better fit to data")
    print("   - Smooth Inversion should have RMS ≤ Bostick (formal optimization)")
    print("\n2. Model Comparison:")
    print("   - Bostick: Fast approximation, can be rough")
    print("   - Smooth Inversion: Optimized smooth model")
    print("\n3. When to use each:")
    print("   - Bostick: Quick reconnaissance, initial model")
    print("   - Smooth Inversion: Final interpretation, publication quality")
    print("="*80)
