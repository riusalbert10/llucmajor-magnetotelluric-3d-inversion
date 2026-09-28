#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per cada estació, genera una figura 1x2 amb l'ajust del model (NOMÉS resistivitat
aparent, sense les fases):

    Columna esq.  : RESISTIVITAT APARENT de les components ANTIDIAGONALS (Zxy, Zyx)
    Columna dreta : RESISTIVITAT APARENT de les components DIAGONALS       (Zxx, Zyy)

En cada panell coexisteixen les dues components del grup:
  - DADES MESURADES (observades)      -> punts amb barres d'error
  - RESPOSTA DEL MODEL (predites)     -> corba (l'ajust)

rho_a = 0.2 * T * |Z|^2      (Z en [mV/km]/[nT], T en s)
"""
import numpy as np, math, collections, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ============================ CONFIG ============================
# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))

OBS  = str(TFG_DIR / "MT_Llucmajor_results/Model_bat_NO_otliers/Arxius_ejecució/Mall_mask_no_tip_1s_ef5_all")
PRED = str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_3/Depuracio_manual_6_NLCG_067.dat")
OUTDIR = str(TFG_DIR / "visualizations/Ajust_Dades")
# ===============================================================

ANTIDIAG = ("ZXY", "ZYX")
DIAG     = ("ZXX", "ZYY")
COLORS = {"ZXY": "tab:blue", "ZYX": "tab:red",
          "ZXX": "tab:green", "ZYY": "tab:purple"}
LABS   = {"ZXY": "xy", "ZYX": "yx", "ZXX": "xx", "ZYY": "yy"}
PHASE_OFFSET = {}   # cap desplaçament: fase en el rang natural [-180, 180]


def parse(fn):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))  # site->comp->{T:(Z,err)}
    for l in open(fn, encoding="utf-8", errors="replace"):
        s = l.strip()
        if not s or s.startswith("#") or s.startswith(">"):
            continue
        p = s.split()
        if len(p) < 11:
            continue
        d[p[1]][p[7]][float(p[0])] = (complex(float(p[8]), float(p[9])), float(p[10]))
    return d


def rhoa(Z, T):
    return 0.2 * T * abs(Z) ** 2


def plot_group(ax_rho, obs_s, pred_s, comps):
    """Plot the measured points and predicted curves (rho_a only) of a group."""
    for comp in comps:
        col = COLORS[comp]
        # --- mesurat: punts + barres d'error ---
        if comp in obs_s:
            Ts = sorted(obs_s[comp])
            x   = [math.log10(T) for T in Ts]
            yr, yre = [], []
            for T in Ts:
                Z, e = obs_s[comp][T]
                yr.append(math.log10(rhoa(Z, T)))
                yre.append(2 * (e / abs(Z)) / math.log(10))        # error en log10(rho_a)
            ax_rho.errorbar(x, yr, yerr=yre, fmt="o", ms=4, color=col,
                            capsize=2, lw=1, label=r"$\rho_a$ %s mesurat" % LABS[comp])
        # --- predit: corba (ajust del model) ---
        if comp in pred_s:
            Ts = sorted(pred_s[comp])
            xp = [math.log10(T) for T in Ts]
            ax_rho.plot(xp, [math.log10(rhoa(pred_s[comp][T][0], T)) for T in Ts],
                        "-", color=col, lw=1.8, label=r"$\rho_a$ %s predit" % LABS[comp])


def main():
    obs = parse(OBS)
    pred = parse(PRED)
    os.makedirs(OUTDIR, exist_ok=True)

    for code in sorted(obs):
        fig, (ax_ra, ax_rd) = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True)

        plot_group(ax_ra, obs[code], pred[code], ANTIDIAG)
        plot_group(ax_rd, obs[code], pred[code], DIAG)

        ax_ra.set_title("Antidiagonal  (Zxy, Zyx)", fontsize=11, fontweight="bold")
        ax_rd.set_title("Diagonal  (Zxx, Zyy)", fontsize=11, fontweight="bold")
        ax_ra.set_ylabel(r"log$_{10}$ $\rho_a$ ($\Omega\cdot$m)")
        for ax in (ax_ra, ax_rd):
            ax.set_xlabel(r"log$_{10}$ T (s)")
            ax.grid(True, ls=":", alpha=0.6)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + 0.25 * (ymax - ymin))  # marge superior per la llegenda
            ax.legend(fontsize=7.5, ncol=2, loc="upper left")

        fig.suptitle("DM 3 - Site %s" % code, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(OUTDIR, "%s.png" % code), dpi=120)
        plt.close(fig)

    print("generats %d grafics a %s" % (len(obs), OUTDIR))


if __name__ == "__main__":
    main()
