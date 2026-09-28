"""
Compare original (no tippers) vs filtered (>50% rel.err) MT data.

For each station: two subplots (off-diagonal and diagonal components),
showing apparent resistivity vs period in log-log scale.
- Kept points: solid marker with error bars
- Removed points: 'x' marker, semi-transparent

Outputs:
  - comparison_per_station.pdf  : 40 pages, one per site
  - comparison_summary.png      : grid with most-affected stations
  - removed_summary.txt         : count of removed points per station
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict, Counter
import os
from pathlib import Path

# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))

DAT_ORIG = str(TFG_DIR / "depurar/Mall_Z_no_tippers.dat")
DAT_CLEAN = str(TFG_DIR / "depurar/Mall_Z_clean.dat")
PDF_OUT = str(TFG_DIR / "compare/comparison_per_station.pdf")
PNG_OUT = str(TFG_DIR / "compare/comparison_summary.png")

# Apparent resistivity constant (Z in mV/km/nT, T in s -> rho_a in Ohm·m)
APP_RES_K = 0.2

# Color scheme
COLORS = {
    'ZXY': '#C0392B',  # red - off-diagonal
    'ZYX': '#2E5C8A',  # blue - off-diagonal
    'ZXX': '#D35400',  # orange - diagonal
    'ZYY': '#27AE60',  # green - diagonal
}


def parse_dat(path):
    """Parse a ModEM .dat file. Returns dict[site] -> list of dicts."""
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', '>')):
                continue
            toks = line.split()
            if len(toks) < 11:
                continue
            try:
                T = float(toks[0])
                site = toks[1]
                comp = toks[7]
                re_v = float(toks[8])
                im_v = float(toks[9])
                err = float(toks[10])
            except ValueError:
                continue
            data[site].append({
                'T': T, 'comp': comp, 'real': re_v, 'imag': im_v, 'err': err
            })
    return data


def app_res(re_v, im_v, T):
    """Apparent resistivity in Ohm·m from impedance in mV/km/nT and T in s."""
    Z2 = re_v * re_v + im_v * im_v
    return APP_RES_K * T * Z2


def app_res_err(re_v, im_v, err, T):
    """Error in apparent resistivity by error propagation."""
    Zmod = math.hypot(re_v, im_v)
    return 2.0 * APP_RES_K * T * Zmod * err


def make_key(d):
    """Unique key for a datum: (period, component) rounded to avoid float issues."""
    return (round(d['T'], 12), d['comp'])


def split_kept_removed(orig_records, clean_records):
    """Given lists of records (from one station), classify which are kept vs removed."""
    clean_keys = {make_key(d) for d in clean_records}
    kept = []
    removed = []
    for d in orig_records:
        if make_key(d) in clean_keys:
            kept.append(d)
        else:
            removed.append(d)
    return kept, removed


def plot_station(ax, kept, removed, comps_to_show, title=""):
    """Plot apparent resistivity for given components, kept and removed points."""
    has_data = False
    for comp in comps_to_show:
        # Kept points: filled markers with error bars
        kept_c = [d for d in kept if d['comp'] == comp]
        if kept_c:
            T_k = np.array([d['T'] for d in kept_c])
            rho_k = np.array([app_res(d['real'], d['imag'], d['T']) for d in kept_c])
            err_k = np.array([app_res_err(d['real'], d['imag'], d['err'], d['T']) for d in kept_c])
            # Avoid zero/negative for log scale
            rho_k = np.where(rho_k <= 0, 1e-3, rho_k)
            ax.errorbar(T_k, rho_k, yerr=err_k, fmt='o', ms=5,
                        color=COLORS[comp], ecolor=COLORS[comp],
                        capsize=2, alpha=0.85,
                        label=f'{comp} kept ({len(kept_c)})')
            has_data = True
        # Removed points: 'x' marker, semi-transparent
        rem_c = [d for d in removed if d['comp'] == comp]
        if rem_c:
            T_r = np.array([d['T'] for d in rem_c])
            rho_r = np.array([app_res(d['real'], d['imag'], d['T']) for d in rem_c])
            rho_r = np.where(rho_r <= 0, 1e-3, rho_r)
            ax.plot(T_r, rho_r, 'x', ms=8, mew=2,
                    color=COLORS[comp], alpha=0.4,
                    label=f'{comp} removed ({len(rem_c)})')
            has_data = True

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Period (s)')
    ax.set_ylabel(r'Apparent resistivity $\rho_a$ (Ω·m)')
    ax.set_title(title, fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    if has_data:
        ax.legend(loc='best', fontsize=8, ncol=2)


def main():
    print("Parsing original file...")
    orig = parse_dat(DAT_ORIG)
    n_orig = sum(len(v) for v in orig.values())
    print(f"  {len(orig)} sites, {n_orig} measurements")

    print("Parsing clean file...")
    clean = parse_dat(DAT_CLEAN)
    n_clean = sum(len(v) for v in clean.values())
    print(f"  {len(clean)} sites, {n_clean} measurements")

    print(f"Removed: {n_orig - n_clean} measurements ({100*(n_orig-n_clean)/n_orig:.1f}%)")

    sites = sorted(orig.keys())

    # Compute removed counts per station
    removed_counts = {}
    for site in sites:
        kept, removed = split_kept_removed(orig[site], clean.get(site, []))
        removed_counts[site] = (len(kept), len(removed))

    # Save summary text
    with open(str(TFG_DIR / "compare/removed_summary.txt"), 'w') as f:
        f.write("Site     Kept   Removed   % removed\n")
        f.write("------   ----   -------   ---------\n")
        for site in sorted(sites, key=lambda s: -removed_counts[s][1]):
            k, r = removed_counts[site]
            tot = k + r
            f.write(f"{site:8s} {k:4d}    {r:4d}     {100*r/tot:5.1f}%\n")

    # ---- Generate PDF with one page per station
    print(f"Generating PDF: {PDF_OUT}")
    with PdfPages(PDF_OUT) as pdf:
        # Cover page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'Comparación de datos MT',
                 ha='center', va='center', fontsize=22, weight='bold')
        fig.text(0.5, 0.62, 'Sin depurar  vs  Depurado >50% error relativo',
                 ha='center', va='center', fontsize=14)
        fig.text(0.5, 0.50,
                 f'Original: {n_orig} datums · Depurado: {n_clean} datums · '
                 f'Eliminados: {n_orig - n_clean} ({100*(n_orig-n_clean)/n_orig:.1f}%)',
                 ha='center', va='center', fontsize=12)
        fig.text(0.5, 0.40,
                 'Cada página: una estación.  Subplot izquierdo = off-diagonal (Zxy, Zyx).\n'
                 'Subplot derecho = diagonal (Zxx, Zyy).\n'
                 'Marcador círculo lleno = punto conservado (con barra de error).\n'
                 'Marcador "×" semitransparente = punto eliminado por filtro.',
                 ha='center', va='center', fontsize=11)
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        for site in sites:
            kept, removed = split_kept_removed(orig[site], clean.get(site, []))
            n_k, n_r = len(kept), len(removed)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            plot_station(axes[0], kept, removed, ['ZXY', 'ZYX'],
                         title=f'{site} — Off-diagonal (Zxy, Zyx)')
            plot_station(axes[1], kept, removed, ['ZXX', 'ZYY'],
                         title=f'{site} — Diagonal (Zxx, Zyy)')

            fig.suptitle(f'{site}  —  conservados {n_k}, eliminados {n_r} '
                         f'({100*n_r/(n_k+n_r):.1f}%)',
                         fontsize=13, y=1.00)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Saved {PDF_OUT}")

    # ---- Generate summary PNG with most-affected stations
    print(f"Generating summary PNG: {PNG_OUT}")
    most_affected = sorted(sites, key=lambda s: -removed_counts[s][1])[:6]
    print(f"  Most affected: {most_affected}")

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    for i, site in enumerate(most_affected):
        kept, removed = split_kept_removed(orig[site], clean.get(site, []))
        plot_station(axes[i//2, (i%2)*2], kept, removed, ['ZXY', 'ZYX'],
                     title=f'{site} — Off-diag')
        plot_station(axes[i//2, (i%2)*2 + 1], kept, removed, ['ZXX', 'ZYY'],
                     title=f'{site} — Diag')

    fig.suptitle('Las 6 estaciones más afectadas por el filtro >50% — Off-diagonal (izq) vs Diagonal (der)',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=110, bbox_inches='tight')
    print(f"  Saved {PNG_OUT}")

    # Console summary
    print("\nTop 10 stations by removed points:")
    for site in sorted(sites, key=lambda s: -removed_counts[s][1])[:10]:
        k, r = removed_counts[site]
        print(f"  {site}: kept {k}, removed {r} ({100*r/(k+r):.1f}%)")


if __name__ == '__main__':
    main()
