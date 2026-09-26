#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per cada estació, genera una figura amb les DADES ORIGINALS (observades) NOMÉS
de les components ANTIDIAGONALS (Zxy, Zyx), sense l'ajust del model:

    Fila superior : RESISTIVITAT APARENT (log10 rho_a)
    Fila inferior : FASE (graus)

Es generen DOS conjunts (dues carpetes): les dades NO depurades i les dades
DEPurades. Perquè la comparació sigui directa, els límits dels eixos es fixen a
partir de les dades NO depurades (REF) i es mantenen idèntics als dos conjunts,
de manera que la gràfica no es reescala en passar d'un a l'altre.

rho_a = 0.2 * T * |Z|^2      (Z en [mV/km]/[nT], T en s)
fase  = atan2(Im Z, Re Z)    (rang natural [-180, 180] graus, sense desplaçaments)
"""
import numpy as np, math, collections, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ CONFIG ============================
VIS = r"C:\Users\alber\TFG\visualizations"

# Referència d'escala: dades NO depurades
REF_OBS = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Costa_Suau1\Arxius_ejecucio\Mall_Z_sense_depurar"

# Conjunts a graficar: (fitxer OBS, carpeta de sortida, etiqueta del títol, sufix del nom)
DATASETS = [
    (r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Costa_Suau1\Arxius_ejecucio\Mall_Z_sense_depurar",
     os.path.join(VIS, "Dades_No_Depurades"), "no depurades", "_no_dep"),
    (r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_bat_NO_otliers\Arxius_ejecució\Malla_Good_10s_efOD5_D15",
     os.path.join(VIS, "Dades_Depurades"), "depurades", "_dep"),
]
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


def phase_deg(Z, comp):
    return math.degrees(math.atan2(Z.imag, Z.real)) + PHASE_OFFSET.get(comp, 0.0)


def compute_limits(obs_ref):
    """Per-station axis limits (x = log10 T, y = log10 rho_a per group), derived
    from the reference dataset so both plots share identical axes."""
    lims = {}
    for code, comps in obs_ref.items():
        allT = [T for c in comps for T in comps[c]]
        if not allT:
            continue
        xlo, xhi = math.log10(min(allT)), math.log10(max(allT))

        def rho_range(group):
            vals = []
            for c in group:
                for T, (Z, e) in comps.get(c, {}).items():
                    v = math.log10(rhoa(Z, T)); err = 2 * (e / abs(Z)) / math.log(10)
                    vals += [v - err, v + err]
            if not vals:
                return None
            lo, hi = min(vals), max(vals); m = 0.05 * ((hi - lo) or 1.0)
            return (lo - m, hi + m)

        lims[code] = dict(x=(xlo - 0.1, xhi + 0.1),
                          rho_anti=rho_range(ANTIDIAG),
                          rho_diag=rho_range(DIAG))
    return lims


def plot_group(ax_rho, ax_ph, obs_s, comps):
    for comp in comps:
        if comp not in obs_s:
            continue
        col = COLORS[comp]
        Ts = sorted(obs_s[comp])
        x   = [math.log10(T) for T in Ts]
        yr, yre, yp, ype = [], [], [], []
        for T in Ts:
            Z, e = obs_s[comp][T]
            yr.append(math.log10(rhoa(Z, T)))
            yre.append(2 * (e / abs(Z)) / math.log(10))
            yp.append(phase_deg(Z, comp))
            ype.append(math.degrees(e / abs(Z)))
        ax_rho.errorbar(x, yr, yerr=yre, fmt="o", ms=4, color=col,
                        capsize=2, lw=1, label=r"$\rho_a$ %s" % LABS[comp])
        ax_ph.errorbar(x, yp, yerr=ype, fmt="o", ms=4, color=col,
                       capsize=2, lw=1, label=r"$\phi$ %s" % LABS[comp])


def generate(obs_path, outdir, label, suffix, lims):
    obs = parse(obs_path)
    os.makedirs(outdir, exist_ok=True)
    for code in sorted(obs):
        fig, (ax_ra, ax_pa) = plt.subplots(2, 1, figsize=(6.5, 7.6), sharex=True)

        # NOMÉS components antidiagonals (Zxy, Zyx)
        plot_group(ax_ra, ax_pa, obs[code], ANTIDIAG)

        ax_ra.set_title("Antidiagonal  (Zxy, Zyx)", fontsize=11, fontweight="bold")
        ax_ra.set_ylabel(r"log$_{10}$ $\rho_a$ ($\Omega\cdot$m)")
        ax_pa.set_ylabel(r"Fase $\phi$ (graus)")
        ax_pa.set_xlabel(r"log$_{10}$ T (s)")
        for ax in (ax_ra, ax_pa):
            ax.grid(True, ls=":", alpha=0.6)
            ax.legend(fontsize=8, ncol=2)

        # --- límits d'eix FIXOS (referència = dades no depurades) ---
        lm = lims.get(code)
        if lm:
            for ax in (ax_ra, ax_pa):
                ax.set_xlim(*lm["x"])
            if lm["rho_anti"]:
                ax_ra.set_ylim(*lm["rho_anti"])
        # fase sempre en el rang natural [-180, 180]
        ax_pa.set_ylim(-180, 180)
        ax_pa.set_yticks(np.arange(-180, 181, 30))
        ax_pa.axhline(0, ls="--", color="gray", lw=1)

        fig.suptitle("Site %s  —  dades %s" % (code, label), fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(outdir, "%s%s.png" % (code, suffix)), dpi=120)
        plt.close(fig)
    print("generats %d grafics a %s" % (len(obs), outdir))


def main():
    lims = compute_limits(parse(REF_OBS))
    for obs_path, outdir, label, suffix in DATASETS:
        generate(obs_path, outdir, label, suffix, lims)


if __name__ == "__main__":
    main()
