"""
Plot RMS convergence curve from ModEM inversion log file
"""
import numpy as np
import matplotlib.pyplot as plt
import re

# Path to log file
log_file = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Smooth1\Run_472_results\mallorca_coast_smooth_inv_NLCG.log"

# Parse the log file to extract iteration numbers and RMS values
iterations = []
rms_values = []

with open(log_file, 'r') as f:
    for line in f:
        # Look for lines with "Completed NLCG iteration" or "START"
        if 'START:' in line:
            # Extract initial RMS
            match = re.search(r'rms=\s+([\d.]+)', line)
            if match:
                iterations.append(0)
                rms_values.append(float(match.group(1)))
        elif 'Completed NLCG iteration' in line:
            # Extract iteration number
            match_iter = re.search(r'iteration\s+(\d+)', line)
            # Read next line for RMS value
            next_line = next(f)
            match_rms = re.search(r'rms=\s+([\d.]+)', next_line)

            if match_iter and match_rms:
                iterations.append(int(match_iter.group(1)))
                rms_values.append(float(match_rms.group(1)))

# Convert to numpy arrays
iterations = np.array(iterations)
rms_values = np.array(rms_values)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, rms_values, 'b-o', linewidth=2, markersize=4)
plt.xlabel('Iteration Number', fontsize=12)
plt.ylabel('RMS Misfit', fontsize=12)
plt.title('ModEM Inversion Convergence - Mallorca MT Survey', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(-1, max(iterations) + 1)

# Add text showing initial and final RMS
plt.text(0.98, 0.95, f'Initial RMS: {rms_values[0]:.2f}\nFinal RMS: {rms_values[-1]:.2f}\nReduction: {(1 - rms_values[-1]/rms_values[0])*100:.1f}%',
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save the figure
output_file = r'C:\Users\alber\TFG\visualizations\rms_convergence.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'RMS convergence plot saved to: {output_file}')

# Show statistics
print(f'\nInversion Statistics:')
print(f'  Total iterations: {len(iterations)-1}')
print(f'  Initial RMS: {rms_values[0]:.3f}')
print(f'  Final RMS: {rms_values[-1]:.3f}')
print(f'  RMS reduction: {(1 - rms_values[-1]/rms_values[0])*100:.1f}%')
print(f'  Target RMS: 1.0 (typical goal)')
print(f'  Status: {"Achieved" if rms_values[-1] <= 1.0 else f"Not achieved (consider more iterations or adjust regularization)"}')

plt.show()
