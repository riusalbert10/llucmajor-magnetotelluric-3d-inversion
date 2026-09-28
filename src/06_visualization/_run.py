#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per cada estacio, grafica log10(resistivitat aparent) vs log10(periode) de les
components off-diagonal Zxy i Zyx:
  - DADES MESURADES  -> punts (amb barres d'error)
  - RESPOSTA DEL MODEL (dades predites) -> corba (l'ajust)

rho_a = 0.2 * T * |Z|^2   (Z en [mV/km]/[nT], T en s)
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
PRED = str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_6/Depuracio_manual_6_NLCG_067.dat")
OUTDIR = str(TFG_DIR / "visualizations/ajust_rho")
# ===============================================================


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


obs = parse(OBS)
pred = parse(PRED)
os.makedirs(OUTDIR, exist_ok=True)
colors = {"ZXY": "tab:blue", "ZYX": "tab:red"}
labs = {"ZXY": "xy", "ZYX": "yx"}

for code in sorted(obs):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for comp in ("ZXY", "ZYX"):
        # mesurat: punts + barres d'error
        if comp in obs[code]:
            Ts = sorted(obs[code][comp])
            x = [math.log10(T) for T in Ts]
            y, ye = [], []
            for T in Ts:
                Z, e = obs[code][comp][T]
                y.append(math.log10(rhoa(Z, T)))
                ye.append(2 * (e / abs(Z)) / math.log(10))          # error en log10
            ax.errorbar(x, y, yerr=ye, fmt="o", ms=4, color=colors[comp],
                        capsize=2, lw=1, label=r"$\rho_a$ %s mesurat" % labs[comp])
        # predit: corba (ajust del model)
        if comp in pred[code]:
            Ts = sorted(pred[code][comp])
            xp = [math.log10(T) for T in Ts]
            yp = [math.log10(rhoa(pred[code][comp][T][0], T)) for T in Ts]
            ax.plot(xp, yp, "-", color=colors[comp], lw=1.8,
                    label=r"$\rho_a$ %s predit" % labs[comp])
    ax.set_xlabel(r"log$_{10}$ T (s)")
    ax.set_ylabel(r"log$_{10}$ $\rho_a$ ($\Omega\cdot$m)")
    ax.set_title("Site %s" % code)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "%s.png" % code), dpi=120)
    plt.close(fig)

print("generats %d grafics a %s" % (len(obs), OUTDIR))
