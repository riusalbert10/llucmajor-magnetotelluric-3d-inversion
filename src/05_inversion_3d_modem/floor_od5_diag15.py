#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A partir d'un fitxer MT natiu (format ModEM):
  1) elimina els periodes > TMAX s,
  2) floor del 5% a les OFF-diagonal (Zxy, Zyx) sobre sqrt(|Zxy*Zyx|):
        err = max(natiu, 0.05 * sqrt(|Zxy*Zyx|))
  3) floor RELATIU del 15% a les DIAGONALS (Zxx, Zyy) sobre el SEU PROPI modul:
        err = max(natiu, 0.15 * |Z_component|)
     (si l'error natiu ja dona err.rel > 15%, es conserva)

Us:  python floor_od5_diag15.py
"""
import collections, math
import numpy as np

# ============================ CONFIG ============================
SRC   = "Mall_mask_no_tip"                    # fitxer natiu (sense floor)
OUT   = "Mall_mask_no_tip_10s_od5_diag15"     # sortida (sobreescriu)
TMAX  = 10.0
FL_OFF  = 0.05    # 5%  sobre sqrt(|Zxy*Zyx|)  -> Zxy, Zyx
FL_DIAG = 0.15    # 15% sobre |Z| propi         -> Zxx, Zyy
OFF  = ("ZXY", "ZYX")
DIAG = ("ZXX", "ZYY")
# ===============================================================


def read(fn):
    hdr, rows = [], []
    for line in open(fn, encoding="utf-8", errors="replace"):
        s = line.rstrip("\n"); st = s.strip()
        if st.startswith("#") or st.startswith(">"):
            hdr.append(s)
        elif st and len(st.split()) >= 11:
            rows.append(st.split())
    return hdr, rows


hdr, rows = read(SRC)
rows = [p for p in rows if float(p[0]) <= TMAX]          # (1) treure periodes > TMAX

# referencia sqrt(|Zxy*Zyx|) per (estacio, periode), a partir de les off-diagonal
mag = collections.defaultdict(dict)
for p in rows:
    mag[(p[1], p[0])][p[7]] = math.hypot(float(p[8]), float(p[9]))
ref = {}
for key, z in mag.items():
    if "ZXY" in z and "ZYX" in z:
        ref[key] = math.sqrt(z["ZXY"] * z["ZYX"])
    elif "ZXY" in z:
        ref[key] = z["ZXY"]
    elif "ZYX" in z:
        ref[key] = z["ZYX"]

out_rows, nraised = [], 0
for p in rows:
    comp = p[7]
    absZ = math.hypot(float(p[8]), float(p[9]))   # |Z| propi
    en = float(p[10])                              # error natiu
    if comp in OFF:
        r = ref.get((p[1], p[0]))
        fl = FL_OFF * r if r is not None else FL_OFF * absZ     # (2) 5% de sqrt(|Zxy*Zyx|)
    elif comp in DIAG:
        fl = FL_DIAG * absZ                                     # (3) 15% del modul propi
    else:
        fl = 0.0
    e = max(en, fl)
    if e > en + 1e-12:
        nraised += 1
    out_rows.append(p[:10] + ["%.6E" % e])

periods = sorted(set(float(p[0]) for p in out_rows))
stations = sorted(set(p[1] for p in out_rows))

out = []
for h in hdr:
    t = h.split()
    if h.strip().startswith(">") and len(t) == 3 and t[1].isdigit() and t[2].isdigit():
        out.append("> %d %d" % (len(periods), len(stations)))
    else:
        out.append(h)
for p in out_rows:
    out.append("%s  %-8s %s %s %s %s %s  %-4s %s %s %s" %
               (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]))
open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")

# --- informe + verificacio ---
print("ESCRIT:", OUT)
print("  periodes:", len(periods), "(max %.4g s)" % max(periods), "| estacions:", len(stations),
      "| components:", len(out_rows), "| errors pujats:", nraised)
Z = collections.defaultdict(dict); E = collections.defaultdict(dict)
for p in out_rows:
    Z[(p[1], p[0])][p[7]] = math.hypot(float(p[8]), float(p[9]))
    E[(p[1], p[0])][p[7]] = float(p[10])
print("  OFF-diagonal  ->  err / sqrt(|Zxy*Zyx|)  (ha de ser >=5%):")
for c in OFF:
    v = [E[k][c] / math.sqrt(z["ZXY"] * z["ZYX"]) for k, z in Z.items()
         if "ZXY" in z and "ZYX" in z and c in E[k] and z["ZXY"] * z["ZYX"] > 0]
    v = np.array(v); print("    %s: minim=%.1f%%  mediana=%.1f%%" % (c, 100 * v.min(), 100 * np.median(v)))
print("  DIAGONALS     ->  err / |Z_propi|  (ha de ser >=15%):")
for c in DIAG:
    v = [E[k][c] / Z[k][c] for k in Z if c in Z[k] and Z[k][c] > 0]
    v = np.array(v); print("    %s: minim=%.1f%%  mediana=%.1f%%" % (c, 100 * v.min(), 100 * np.median(v)))
