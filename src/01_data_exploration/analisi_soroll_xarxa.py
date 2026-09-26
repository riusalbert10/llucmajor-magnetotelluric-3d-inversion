#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisi del soroll de xarxa electrica (50 Hz) en dades MT invertides amb ModEM.

Genera dues figures:
  1) nRMS per periode vs frequencia  -> mostra el pic a ~50 Hz (banda de xarxa).
  2) Mapa espacial del nRMS a la banda de xarxa per estacio -> comprova si el
     soroll s'agrupa a prop d'una font (torre / linia electrica).

Necessita:
  - fitxer de dades OBSERVADES (format ModEM, amb columnes X(m) Y(m) i el error
    ja amb l'error floor aplicat, p.ex. el _ef5).
  - fitxer de dades PREDITES pel model (el *_NLCG_###.dat que escriu ModEM).

Us:  python analisi_soroll_xarxa.py
"""
import numpy as np, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ CONFIG ============================
# Posa aqui les rutes als teus fitxers (mateixa parella obs/pred de la inversio)
OBS  = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_bat_NO_otliers\Arxius_ejecució\Mall_mask_no_tip_1s_ef5"                             # dades observades
PRED = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Comparable_Tesis_Arango\Run_632_results\mallorca_17km_seafixed_ef5_cov_inv_NLCG_067.dat"     # resposta del model
OUTDIR = r"C:\Users\alber\TFG\visualizations"                                                # on desar les figures
BAND = (40.0, 110.0)   # Hz: fonamental 50 Hz + ~100 Hz (1r harmonic)
# ===============================================================


def parse(fn, want_coords=False):
    """Llegeix un .dat de ModEM (Full_Impedance). Clau = (site, comp, periode)."""
    d, xy = {}, {}
    for line in open(fn, encoding="utf-8", errors="replace"):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(">"):
            continue
        p = s.split()
        if len(p) < 11:
            continue
        code, comp, per = p[1], p[7], float(p[0])
        d[(code, comp, "%.6e" % per)] = (complex(float(p[8]), float(p[9])),
                                         float(p[10]), per)
        if want_coords:
            xy[code] = (float(p[4]), float(p[5]))   # X = Nord, Y = Est
    return (d, xy) if want_coords else d


obs, xy = parse(OBS, True)
pred = parse(PRED)

# --- nRMS per periode (totes les estacions) i nRMS a la banda per estacio ---
per = collections.defaultdict(lambda: [0.0, 0])
st_band = collections.defaultdict(lambda: [0.0, 0])
for key, (Zo, err, T) in obs.items():
    if key not in pred:
        continue
    Zp = pred[key][0]
    rr = ((Zo.real - Zp.real) / err) ** 2 + ((Zo.imag - Zp.imag) / err) ** 2   # residu^2 (real+imag)
    per[T][0] += rr; per[T][1] += 2
    if BAND[0] <= 1.0 / T <= BAND[1]:
        st_band[key[0]][0] += rr; st_band[key[0]][1] += 2

rms = lambda v: (v[0] / v[1]) ** 0.5   # nRMS = sqrt( sum(residu^2) / N )

# ------------------ FIGURA 1: nRMS vs frequencia ------------------
Ts = sorted(per)
freqs = [1.0 / T for T in Ts]
vals = [rms(per[T]) for T in Ts]
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.semilogx(freqs, vals, "-o", ms=4)
ax.axvspan(BAND[0], BAND[1], color="red", alpha=0.15, label="banda xarxa 50/100 Hz")
ax.axvline(50, color="red", ls="--", lw=1)
ax.axvline(100, color="red", ls=":", lw=1)
ax.set_xlabel("Freqüència (Hz)")
ax.set_ylabel("nRMS per periode")
ax.set_title("Misfit vs freqüència ")
ax.grid(True, ls=":", alpha=0.6)
ax.legend()
fig.tight_layout()
fig.savefig(OUTDIR + "/powerline_nRMS_vs_freq.png", dpi=120)
plt.close(fig)

# ------------------ FIGURA 2: mapa espacial del misfit a la banda ------------------
codes = [c for c in st_band if st_band[c][1] > 0]
E = np.array([xy[c][1] for c in codes])   # Est
N = np.array([xy[c][0] for c in codes])   # Nord
R = np.array([rms(st_band[c]) for c in codes])
fig, ax = plt.subplots(figsize=(6.8, 6))
sc = ax.scatter(E, N, c=R, s=140, cmap="hot_r", edgecolor="k")
for c, e, n in zip(codes, E, N):
    ax.annotate(c.replace("mall", ""), (e, n), fontsize=6, ha="center", va="center")
plt.colorbar(sc, label="nRMS a la banda 50/100 Hz")
ax.set_xlabel("Est Y (m)")
ax.set_ylabel("Nord X (m)")
ax.set_title("Misfit a la banda de xarxa per estacio")
ax.set_aspect("equal")
ax.grid(True, ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(OUTDIR + "/powerline_map.png", dpi=120)
plt.close(fig)

# ------------------ resum per consola ------------------
print("Periodes dins la banda de xarxa (%g-%g Hz):" % BAND)
for T in Ts:
    if BAND[0] <= 1.0 / T <= BAND[1]:
        print("  T=%.5g s  f=%.1f Hz  nRMS=%.2f" % (T, 1.0 / T, rms(per[T])))
print("\nEstacions pitjors a la banda de xarxa:")
for c in sorted(codes, key=lambda c: -rms(st_band[c]))[:8]:
    print("  %-8s nRMS_banda=%.1f  (E=%.0f, N=%.0f)"
          % (c, rms(st_band[c]), xy[c][1], xy[c][0]))
print("\nFigures desades a:", OUTDIR)
