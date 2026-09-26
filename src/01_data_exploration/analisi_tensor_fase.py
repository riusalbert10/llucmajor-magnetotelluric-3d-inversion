#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor de fase (skew beta) creuat amb el nRMS de les components diagonals (Zxx, Zyy).

Objectiu: decidir si el misfit alt de les diagonals d'una estacio ve de 3D REAL
o de SOROLL/DISTORSIO. El skew beta del tensor de fase (Caldwell, Bibby & Brown 2004)
es INMUNE a la distorsio galvanica:
   - |beta| petit i coherent  -> estructura 1D/2D -> diagonals grans = soroll/distorsio
   - |beta| gran i coherent   -> 3D real
   - beta molt dispers        -> soroll

Genera:
   - diagonals_vs_phasetensor.txt : taula per estacio (nRMS_Zxx, nRMS_Zyy, med|beta|, std, veredicte)
   - phasetensor_skew.png         : beta vs periode per a les pitjors i millors estacions

Us:  python analisi_tensor_fase.py
"""
import numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ============================ CONFIG ============================
OBS  = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_bat_NO_otliers\Arxius_ejecució\Mall_mask_no_tip_1s_ef5"                             # dades observades
PRED = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Comparable_Tesis_Arango\Run_632_results\mallorca_17km_seafixed_ef5_cov_inv_NLCG_067.dat"     # resposta del model
OUTDIR = r"C:\Users\alber\TFG\visualizations"
BETA_THR = 5.0    # llindar de skew (graus): |beta| < BETA_THR  => estructura 1D/2D
# ===============================================================


def parse(fn):
    """Retorna: Z[(site,periode)]={comp:complex}, i D[(site,comp,per_str)]=(Z, error)."""
    Z = collections.defaultdict(dict)
    D = {}
    for l in open(fn, encoding="utf-8", errors="replace"):
        s = l.strip()
        if not s or s.startswith("#") or s.startswith(">"):
            continue
        p = s.split()
        if len(p) < 11:
            continue
        z = complex(float(p[8]), float(p[9]))
        Z[(p[1], float(p[0]))][p[7]] = z
        D[(p[1], p[7], "%.6e" % float(p[0]))] = (z, float(p[10]))
    return Z, D


Zobs, obs = parse(OBS)
_, pred = parse(PRED)

# ---- skew beta del tensor de fase per (site, periode) ----
beta = collections.defaultdict(list)   # site -> [(T, beta_deg)]
for (site, T), z in Zobs.items():
    if not all(c in z for c in ("ZXX", "ZXY", "ZYX", "ZYY")):
        continue
    X = np.array([[z["ZXX"].real, z["ZXY"].real], [z["ZYX"].real, z["ZYY"].real]])
    Y = np.array([[z["ZXX"].imag, z["ZXY"].imag], [z["ZYX"].imag, z["ZYY"].imag]])
    try:
        Phi = np.linalg.inv(X) @ Y                      # tensor de fase = X^-1 Y
    except np.linalg.LinAlgError:
        continue
    b = 0.5 * np.degrees(np.arctan2(Phi[0, 1] - Phi[1, 0], Phi[0, 0] + Phi[1, 1]))
    beta[site].append((T, b))

# ---- nRMS de les diagonals (obs vs pred) ----
dg = collections.defaultdict(lambda: {"ZXX": [0., 0], "ZYY": [0., 0]})
for k, (Zo, e) in obs.items():
    if k[1] not in ("ZXX", "ZYY") or k not in pred:
        continue
    Zp = pred[k][0]
    rr = ((Zo.real - Zp.real) / e) ** 2 + ((Zo.imag - Zp.imag) / e) ** 2
    dg[k[0]][k[1]][0] += rr; dg[k[0]][k[1]][1] += 2
rms = lambda v: (v[0] / v[1]) ** 0.5 if v[1] else float("nan")


def verdict(mb, sb):
    if mb < BETA_THR:
        return "1D/2D -> diagonals espuries (down-weight)"
    if sb > mb:
        return "beta dispers -> soroll (down-weight)"
    return "3D coherent -> senyal real (conservar)"


# ---- taula ----
rows = []
for site in dg:
    bs = np.array([b for _, b in beta.get(site, [])])
    mb = np.median(np.abs(bs)) if len(bs) else float("nan")
    sb = np.std(bs) if len(bs) else float("nan")
    xx, yy = rms(dg[site]["ZXX"]), rms(dg[site]["ZYY"])
    rows.append((site, xx, yy, max(xx, yy), mb, sb))
rows.sort(key=lambda r: -r[3])

L = ["# Diagonals (Zxx,Zyy) vs tensor de fase (skew beta)  -  run_632",
     "# beta immune a distorsio galvanica. Llindar: |beta| < %.0f deg => 1D/2D" % BETA_THR,
     "# ordenat de MAJOR a menor nRMS diagonal",
     "# %-8s %8s %8s %10s %10s   %s" % ("site", "nRMS_Zxx", "nRMS_Zyy", "med|beta|", "std_beta", "veredicte")]
for site, xx, yy, mx, mb, sb in rows:
    L.append("  %-8s %8.2f %8.2f %8.1f %8.1f   %s" % (site, xx, yy, mb, sb, verdict(mb, sb)))
open(OUTDIR + "/diagonals_vs_phasetensor.txt", "w").write("\n".join(L) + "\n")

# ---- figura: beta vs periode (3 pitjors + 3 millors en diagonal) ----
worst = [r[0] for r in rows[:3]]
good = ["mall01"] + [r[0] for r in rows[-2:]]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for site in worst + good:
    pts = sorted(beta.get(site, []))
    if not pts:
        continue
    T = np.array([p for p, _ in pts]); b = np.array([q for _, q in pts])
    style = "-o" if site in worst else "--s"
    tag = " (diag alt)" if site in worst else " (diag baix)"
    ax.plot(np.log10(T), b, style, ms=4, label=site + tag)
ax.axhspan(-BETA_THR, BETA_THR, color="grey", alpha=0.15, label="|beta|<%g° (1D/2D)" % BETA_THR)
ax.axhline(0, color="k", lw=.6)
ax.set_xlabel(r"log$_{10}$ T (s)"); ax.set_ylabel(r"skew $\beta$ (graus)")
ax.set_title("Tensor de fase: skew per estació")
ax.grid(True, ls=":", alpha=.6); ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(OUTDIR + "/phasetensor_skew.png", dpi=120); plt.close(fig)

# ---- resum ----
print("Llindar |beta| < %.0f deg" % BETA_THR)
print("Top 6 pitjors en diagonals:")
for r in rows[:6]:
    print("  %-8s Zxx=%.1f Zyy=%.1f | med|beta|=%.1f std=%.1f -> %s"
          % (r[0], r[1], r[2], r[4], r[5], verdict(r[4], r[5])))
print("\nfitxers: diagonals_vs_phasetensor.txt , phasetensor_skew.png")
