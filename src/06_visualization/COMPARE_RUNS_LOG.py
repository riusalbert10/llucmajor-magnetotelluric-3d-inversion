"""
compare_runs.py
---------------
Parse ModEM NLCG log files and produce separate comparison figures 
independently for two blocks: Base runs (with zooms) and Smoothed runs (RMS full plot only).
"""

import re
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================ CONFIG ============================

# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))

RUNS_BASE = [
    {
        'label': 'Raw data',
        'log'  : str(TFG_DIR / "MT_Llucmajor_results/Model_Costa_Abrupte/Run_473_results/mallorca_coast_inv_NLCG.log"),
        'color': "#CD6464",
        'marker': 'o',
    },
    {
        'label': 'DM 0',
        'log'  : str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_0/mallorca_coast_Bat_NO_otliers_inv_NLCG.log"),
        'color': "#781494",
        'marker': 'o',
    },
    {
        'label': 'DM 0 + ef',
        'log'  : str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_0_ef/mallorca_bat_seafixed_ef5_cov_inv_NLCG.log"),
        'color': '#7f7f7f',
        'marker': 'o',
    },
    {
        'label': 'DM 1',
        'log'  : str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_1/Run_664/mallorca_10s_dep5_NLCG.log"),
        'color': "#fbff00",
        'marker': 'x',
    },
    {
        'label': 'DM 2',
        'log'  : str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_2/Run_635/mallorca_recond_inv_NLCG.log"),
        'color': "#1F08EFFF",
        'marker': 'x',
    },
    {
        'label': 'DM 3',
        'log'  : str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_3/Depuracio_manual_6_NLCG.log"),
        'color': "#62eb0086",
        'marker': 'x',
    },
    {
        'label': 'DM 4',
        'log'  : str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_4/Run_667/Depuracio_manual_7_NLCG.log"),
        'color': "#720202FF",
        'marker': 'x',
    }, 
]         

RUNS_SMOOTH = [
    {
        'label': 'Suavitzat LOWESS 1',
        'log'  : str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth1/Run_472_results/mallorca_coast_smooth_inv_NLCG.log"),
        'color': "#e714d9",
        'marker': 's',
    },
    {
        'label': 'Suavitzat LOWESS 2',
        'log'  : str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth1_Bat/Run_474_results/job_474_Mallorca_Inv_coast_bat_sm1/mallorca_coast_Bat_sm1_inv_NLCG.log"),
        'color': "#55453d",
        'marker': '^',
    },
    {
        'label': 'Suavitzat Hampel',
        'log'  : str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat/Run_475_results/job_475_Mallorca_Inv_coast_bat_sm3/mallorca_coast_Bat_sm3_inv_NLCG.log"),
        'color': '#ff7f0e',
        'marker': 'x',
    },
]

OUT_DIR = Path(str(TFG_DIR / "visualizations"))


# ============================ PARSER ============================
LINE_RE = re.compile(
    r'^\s+with: f=([\d.E+-]+)\s+m2=([\d.E+-]+)\s+rms=\s*([\d.E+-]+)\s+lambda=([\d.E+-]+)'
)


def parse_log(path):
    rows = []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            m = LINE_RE.match(line)
            if m:
                f_, m2, rms, lam = (float(g) for g in m.groups())
                rows.append((len(rows), rms, lam, m2, f_))
    return np.array(rows)


def run_summary(data, label):
    n = len(data)
    if n == 0:
        return f"{label}: empty log"
    return (f"{label}:  {n:3d} iters,  "
            f"RMS {data[0,1]:.2f}→{data[-1,1]:.2f},  "
            f"λ_end={data[-1,2]:.0e},  m²_end={data[-1,3]:.3f}")


# ============================ PLOTS & TABLES (PER BLOCK) ============================
def generate_block_outputs(parsed, suffix, with_zooms=True):
    if not parsed:
        print(f"  [INFO] No data to process for block '{suffix}', skipping.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- FIGURA 1: RMS Convergence ----
    if with_zooms:
        fig_rms = plt.figure(figsize=(15, 6))
        gs = fig_rms.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.35, wspace=0.22)
        ax_full = fig_rms.add_subplot(gs[:, 0])
        ax_z_top = fig_rms.add_subplot(gs[0, 1])
        ax_z_bot = fig_rms.add_subplot(gs[1, 1])
    else:
        fig_rms, ax_full = plt.subplots(figsize=(9, 6))

    for r in parsed:
        d = r['data']
        if len(d) == 0: continue
        ax_full.plot(d[:, 0], d[:, 1], '-', marker=r['marker'], ms=3.5, lw=1.3,
                     color=r['color'], label=r['label'])
        ax_full.axhline(d[-1, 1], color=r['color'], ls=':', lw=0.7, alpha=0.6)
    
    ax_full.set_xlabel('Iteració')
    ax_full.set_ylabel('RMS')
    ax_full.grid(alpha=0.3)
    ax_full.legend(fontsize=8, loc='upper right')

    if with_zooms:
        ZOOMS = [
            (ax_z_top, 0, 140, 10, 18),
            (ax_z_bot, 0,  95,  0, 15),
        ]
        for ax, xmin, xmax, ymin, ymax in ZOOMS:
            for r in parsed:
                d = r['data']
                if len(d) == 0: continue
                mask = (d[:, 0] >= xmin) & (d[:, 0] <= xmax)
                if not mask.any(): continue
                ax.plot(d[mask, 0], d[mask, 1], '-', marker=r['marker'], ms=4, lw=1.4,
                        color=r['color'])
                ax.axhline(d[-1, 1], color=r['color'], ls=':', lw=0.7, alpha=0.6)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel('Iteració')
            ax.set_ylabel('RMS')
            ax.grid(alpha=0.3)

    fig_rms.suptitle('Convergència del RMS', fontsize=14, fontweight='bold')
    fig_rms.tight_layout(rect=[0, 0, 1, 0.96])
    path_rms = OUT_DIR / f'rms_comparison_{suffix}.png'
    fig_rms.savefig(path_rms, dpi=140, bbox_inches='tight')
    plt.close(fig_rms)
    print(f"Saved: {path_rms}")

    # ---- FIGURA 2: Lambda Decay ----
    fig_lam, ax = plt.subplots(figsize=(8, 6))
    for r in parsed:
        d = r['data']
        if len(d) == 0: continue
        ax.semilogy(d[:, 0], d[:, 2], '-', marker=r['marker'], ms=3.5, lw=1.3,
                    color=r['color'], label=r['label'])
    ax.axhline(1e-8, color='black', ls=':', lw=1, label='Límit de λ → sortida automàtica')
    ax.set_xlabel('Iteració')
    ax.set_ylabel(r'$\lambda$  (paràmetre d\'esmorteïment)')
    ax.set_title(r'Decaïment del paràmetre $\lambda$', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8.5, loc='lower left')
    fig_lam.tight_layout()
    path_lam = OUT_DIR / f'lambda_decay_{suffix}.png'
    fig_lam.savefig(path_lam, dpi=140, bbox_inches='tight')
    plt.close(fig_lam)
    print(f"Saved: {path_lam}")

    # ---- FIGURA 3: Model Norm m² ----
    fig_m2, ax = plt.subplots(figsize=(8, 6))
    for r in parsed:
        d = r['data']
        if len(d) == 0: continue
        ax.plot(d[:, 0], d[:, 3], '-', marker=r['marker'], ms=3.5, lw=1.3,
                color=r['color'], label=r['label'])
    ax.set_xlabel('Iteració')
    ax.set_ylabel(r'$m^2$  (norma del model)')
    ax.set_title(r'Norma del model $m^2$', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc='upper left')
    fig_m2.tight_layout()
    path_m2 = OUT_DIR / f'model_norm_{suffix}.png'
    fig_m2.savefig(path_m2, dpi=140, bbox_inches='tight')
    plt.close(fig_m2)
    print(f"Saved: {path_m2}")

    # ---- EXPORTACIÓN DE TABLA DE RESUMEN (CSV) ----
    csv_path = OUT_DIR / f'runs_summary_table_{suffix}.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Label,Total_Iterations,Initial_RMS,Final_RMS,Final_Lambda,Final_Model_Norm\n")
        for r in parsed:
            d = r['data']
            if len(d) == 0: continue
            f.write(f'"{r["label"]}",{len(d)},{d[0,1]:.4f},{d[-1,1]:.4f},{d[-1,2]:.2e},{d[-1,3]:.4f}\n')
    print(f"Saved summary table: {csv_path}")

    # ---- EXPORTACIÓN DE TABLA EN CÓDIGO LATEX ----
    def _tex_esc(s):
        return (s.replace('\\', r'\textbackslash{}').replace('&', r'\&')
                 .replace('%', r'\%').replace('_', r'\_').replace('#', r'\#'))

    def _sci(x):
        s = f"{x:.2e}"
        mant, exp = s.split('e')
        return rf"${mant}\times10^{{{int(exp)}}}$"

    tex_path = OUT_DIR / f'runs_summary_table_{suffix}.tex'
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(r"% Requiere \usepackage{booktabs} en el preámbulo" + "\n")
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"  \centering" + "\n")
        f.write(r"  \caption{Resum de les inversions ModEM NLCG (" + suffix.upper() + r"): nombre d'iteracions, "
                r"RMS inicial i final, funció objectiu final $f$, norma del model final "
                r"$m^2$ i paràmetre d'esmorteïment final $\lambda$.}" + "\n")
        f.write(r"  \label{tab:modem_runs_" + suffix + r"}" + "\n")
        f.write(r"  \begin{tabular}{l r r r r r r}" + "\n")
        f.write(r"    \toprule" + "\n")
        f.write(r"    Assaig & Iteracions & RMS inicial & RMS final & $f_\text{final}$ "
                r"& $m^2_\text{final}$ & $\lambda_\text{final}$ \\" + "\n")
        f.write(r"    \midrule" + "\n")
        for r in parsed:
            d = r['data']
            if len(d) == 0: continue
            f.write(f"    {_tex_esc(r['label'])} & {len(d)} & "
                    f"{d[0,1]:.2f} & {d[-1,1]:.2f} & {_sci(d[-1,4])} & "
                    f"{d[-1,3]:.3f} & {_sci(d[-1,2])} \\\\\n")
        f.write(r"    \bottomrule" + "\n")
        f.write(r"  \end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"Saved LaTeX table: {tex_path}\n")


# ============================ MAIN ============================
def process_block(runs_list, suffix, with_zooms=True):
    print(f'--- Parsing logs for block: {suffix.upper()} ---')
    parsed = []
    for r in runs_list:
        path = Path(r['log'])
        if '*' in str(path):
            matches = list(path.parent.glob(path.name))
            if not matches:
                print(f"  [WARN] no match for {r['log']}, skipping")
                continue
            path = matches[0]
        if not path.exists():
            print(f"  [WARN] {path} not found, skipping")
            continue
        d = parse_log(path)
        print('  ' + run_summary(d, r['label']))
        parsed.append({**r, 'data': d})
    
    print()
    generate_block_outputs(parsed, suffix, with_zooms=with_zooms)


def main():
    # Processa el bloc base (amb zooms al RMS)
    process_block(RUNS_BASE, 'base', with_zooms=True)

    # Processa el bloc de suavitzats (sense zooms al RMS, només gràfic general)
    process_block(RUNS_SMOOTH, 'smooth', with_zooms=False)


if __name__ == '__main__':
    main()