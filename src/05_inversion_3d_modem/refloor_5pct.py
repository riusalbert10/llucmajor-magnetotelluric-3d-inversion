#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construeix el fitxer de 39 periodes (<=10 s) amb un error floor UNIFORME del 5%.

Parteix dels fitxers estables:
  - BASE   : seleccio <=1 s amb les diagonals ja depurades (maskD).
  - SOURCE : fitxer de banda completa, per als periodes 1 s < T <= TMAX.
  - NATIVE : errors NATIUS (sense floor), per re-aplicar el 5% net.

Floor:  err = max(error_natiu, 0.05 * sqrt(|Zxy*Zyx|))  a les 4 components.
(Es fa des dels natius per eliminar qualsevol floor previ, p.ex. el 7% de la fila Y.)

Us:  python refloor_5pct.py
"""
import numpy as np, collections

# ============================ CONFIG ============================
BASE   = "Mall_maskD_no_tip_1s_ef5"       # seleccio <=1 s (diagonals depurades)
SOURCE = "Mall_mask_no_tip_ef5"           # font per als periodes 1-10 s
NATIVE = "Mall_mask_no_tip"               # errors natius (sense floor)
OUT    = "Mall_maskD_no_tip_full_ef5_u5"  # sortida, floor uniforme, TOTS els periodes
TMAX   = 1e9
FLOOR  = 0.05
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


hbase, rbase = read(BASE)
_, rsrc = read(SOURCE)
_, rnat = read(NATIVE)

k = lambda p: (p[1], p[7], "%.6e" % float(p[0]))

# seleccio final: base (<=1 s, depurat) + font (1 s < T <= TMAX)
seen = set(k(p) for p in rbase)
rows = list(rbase)
for p in rsrc:
    if 1.0 < float(p[0]) <= TMAX and k(p) not in seen:
        rows.append(p); seen.add(k(p))

# errors natius i referencia sqrt(|Zxy*Zyx|)
nat = {k(p): p[10] for p in rnat}
tmp = collections.defaultdict(dict)
for p in rows:
    tmp[(p[1], p[0])][p[7]] = abs(complex(float(p[8]), float(p[9])))
ref = {}
for key, z in tmp.items():
    if "ZXY" in z and "ZYX" in z:
        ref[key] = np.sqrt(z["ZXY"] * z["ZYX"])
    elif "ZXY" in z:
        ref[key] = z["ZXY"]
    elif "ZYX" in z:
        ref[key] = z["ZYX"]

# aplicar floor 5% uniforme des dels natius
out_rows, raised, kept, nomiss = [], 0, 0, 0
for p in rows:
    en = nat.get(k(p))
    if en is None:
        en = p[10]; nomiss += 1
    e = float(en)
    r = ref.get((p[1], p[0]))
    if r is not None:
        fl = FLOOR * r
        if e < fl: e = fl; raised += 1
        else: kept += 1
    out_rows.append(p[:10] + ["%.6E" % e])

# ordenar i escriure
comp_ord = {"ZXX": 0, "ZXY": 1, "ZYX": 2, "ZYY": 3}
out_rows.sort(key=lambda p: (float(p[0]), p[1], comp_ord.get(p[7], 9)))
periods = sorted(set(float(p[0]) for p in out_rows))
stations = sorted(set(p[1] for p in out_rows))
out = []
for h in hbase:
    t = h.split()
    if h.strip().startswith(">") and len(t) == 3 and t[1].isdigit() and t[2].isdigit():
        out.append("> %d %d" % (len(periods), len(stations)))
    else:
        out.append(h)
for p in out_rows:
    out.append("%s  %-8s %s %s %s %s %s  %-4s %s %s %s" %
               (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]))
open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")

print("ESCRIT:", OUT)
print("  components:", len(out_rows), "| periodes:", len(periods), "| estacions:", len(stations))
print("  errors pujats al floor %.0f%%: %d | natius conservats (>floor): %d | sense natiu: %d" % (FLOOR*100, raised, kept, nomiss))

# verificacio: floor efectiu per component (minim ha de ser 5.0%)
Z = collections.defaultdict(dict); E = collections.defaultdict(dict)
for p in out_rows:
    Z[(p[1], p[0])][p[7]] = abs(complex(float(p[8]), float(p[9])))
    E[(p[1], p[0])][p[7]] = float(p[10])
print("  verificacio err / sqrt|ZxyZyx| (minim = floor):")
for c in ("ZXX", "ZYY", "ZXY", "ZYX"):
    v = [E[key][c] / np.sqrt(z["ZXY"] * z["ZYX"]) for key, z in Z.items()
         if "ZXY" in z and "ZYX" in z and c in E[key] and z["ZXY"] * z["ZYX"] > 0]
    v = np.array(v)
    print("    %s: minim=%.1f%%  mediana=%.1f%%" % (c, 100 * v.min(), 100 * np.median(v)))
