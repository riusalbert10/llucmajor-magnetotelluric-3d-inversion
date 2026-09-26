#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per cada estació, genera una figura 2x2 amb l'ajust del model, resistivitat
aparent I fase:

    Columna esq.  : components ANTIDIAGONALS (Zxy, Zyx)
    Columna dreta : components DIAGONALS       (Zxx, Zyy)

    Fila superior : RESISTIVITAT APARENT (rho_a)
    Fila inferior : FASE                            (just a sota de cada
                                                       component, com es demana)

En cada panell coexisteixen les dues components del grup:
  - DADES MESURADES (observades)      -> punts amb barres d'error
  - RESPOSTA DEL MODEL (predites)     -> corba (l'ajust)

rho_a = 0.2 * T * |Z|^2      (Z en [mV/km]/[nT], T en s)
fase  = atan2(Im Z, Re Z)    (graus, rang natural [-180, 180], sense desplaçament)
"""
import numpy as np, math, collections, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ CONFIG ============================
OBS  = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_bat_NO_otliers\Arxius_ejecució\Mall_mask_no_tip_1s_ef5_all_depD26_52_53"
PRED = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Depuracio_manual_4\Run_667\Depuracio_manual_7_NLCG_073.dat"
OUTDIR = r"C:\Users\alber\TFG\visualizations\Ajust_Dades_Fase"
# ===============================================================

ANTIDIAG = ("ZXY", "ZYX")
DIAG     = ("ZXX", "ZYY")
COLORS = {"ZXY": "tab:blue", "ZYX": "tab:red",
          "ZXX": "tab:green", "ZYY": "tab:purple"}
LABS   = {"ZXY": "xy", "ZYX": "yx", "ZXX": "xx", "ZYY": "yy"}


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


def phase_deg(Z):
    return math.degrees(math.atan2(Z.imag, Z.real))


def plot_group(ax_rho, ax_pha, obs_s, pred_s, comps):
    """Plot the measured points and predicted curves (rho_a i fase) d'un grup."""
    for comp in comps:
        col = COLORS[comp]
        # --- mesurat: punts + barres d'error ---
        if comp in obs_s:
            Ts = sorted(obs_s[comp])
            x = [math.log10(T) for T in Ts]

            yr, yre = [], []
            yp, ype = [], []
            for T in Ts:
                Z, e = obs_s[comp][T]
                yr.append(math.log10(rhoa(Z, T)))
                yre.append(2 * (e / abs(Z)) / math.log(10))        # error en log10(rho_a)
                yp.append(phase_deg(Z))
                ype.append(math.degrees(e / abs(Z)))                # error en fase (graus)

            ax_rho.errorbar(x, yr, yerr=yre, fmt="o", ms=4, color=col,
                             capsize=2, lw=1, label=r"$\rho_a$ %s mesurat" % LABS[comp])
            ax_pha.errorbar(x, yp, yerr=ype, fmt="o", ms=4, color=col,
                             capsize=2, lw=1, label=r"$\phi$ %s mesurat" % LABS[comp])

        # --- predit: corba (ajust del model) ---
        if comp in pred_s:
            Ts = sorted(pred_s[comp])
            xp = [math.log10(T) for T in Ts]
            ax_rho.plot(xp, [math.log10(rhoa(pred_s[comp][T][0], T)) for T in Ts],
                        "-", color=col, lw=1.8, label=r"$\rho_a$ %s predit" % LABS[comp])
            ax_pha.plot(xp, [phase_deg(pred_s[comp][T][0]) for T in Ts],
                        "-", color=col, lw=1.8, label=r"$\phi$ %s predit" % LABS[comp])


def main():
    obs = parse(OBS)
    pred = parse(PRED)
    os.makedirs(OUTDIR, exist_ok=True)

    for code in sorted(obs):
        fig, ((ax_ra, ax_rd), (ax_pa, ax_pd)) = plt.subplots(
            2, 2, figsize=(11.0, 7.6), sharex=True, height_ratios=[1.3, 1.0])

        plot_group(ax_ra, ax_pa, obs[code], pred[code], ANTIDIAG)
        plot_group(ax_rd, ax_pd, obs[code], pred[code], DIAG)

        ax_ra.set_title("Antidiagonal  (Zxy, Zyx)", fontsize=11, fontweight="bold")
        ax_rd.set_title("Diagonal  (Zxx, Zyy)", fontsize=11, fontweight="bold")

        ax_ra.set_ylabel(r"log$_{10}$ $\rho_a$ ($\Omega\cdot$m)")
        ax_pa.set_ylabel(r"Fase $\phi$ (graus)")

        for ax in (ax_ra, ax_rd, ax_pa, ax_pd):
            ax.grid(True, ls=":", alpha=0.6)
            ax.legend(fontsize=7.5, ncol=2)
        for ax in (ax_pa, ax_pd):
            ax.set_xlabel(r"log$_{10}$ T (s)")

        fig.suptitle("Ajust $\\rho_a$ i $\\phi$ - Site %s" % code, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(OUTDIR, "%sf.png" % code), dpi=120)
        plt.close(fig)

    print("generats %d grafics a %s" % (len(obs), OUTDIR))


if __name__ == "__main__":
    main()
