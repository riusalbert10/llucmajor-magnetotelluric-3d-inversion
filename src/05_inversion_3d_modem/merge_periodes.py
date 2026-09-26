#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Afegeix a un fitxer MT (base) els periodes que li falten, extrets d'un fitxer font
que en te mes. Mante la coherencia del format ModEM i verifica integritat.

Cas d'us: partir del fitxer <=1 s i afegir-hi els periodes 1 s < T <= TMAX s
extrets del fitxer de banda completa.

Us:  python merge_periodes.py
"""
import collections

# ============================ CONFIG ============================
BASE   = "Mall_maskD_no_tip_1s_ef5"    # fitxer base DEPURAT (diagonals netejades, <=1 s)
SOURCE = "Mall_mask_no_tip_ef5"        # fitxer font (te TOTS els periodes)
OUT    = "Mall_maskD_no_tip_10s_ef5"   # sortida
TMAX   = 10.0    # afegeix periodes amb 1 s < T <= TMAX  (posa 1e9 per a banda completa)
# ===============================================================


def read(fn):
    hdr, rows = [], []
    for line in open(fn, encoding="utf-8", errors="replace"):
        s = line.rstrip("\n")
        st = s.strip()
        if st.startswith("#") or st.startswith(">"):
            hdr.append(s)
        elif st:
            p = st.split()
            if len(p) >= 11:
                rows.append(p)
    return hdr, rows


hbase, rbase = read(BASE)
hsrc, rsrc = read(SOURCE)


def key(p):
    return (p[1], p[7], "%.6e" % float(p[0]))


# --- coherencia: els periodes afegits (1-10 s) no han de solapar amb el base (<=1 s) ---
basekeys = set(key(p) for p in rbase)
overlap = sum(1 for p in rsrc if 1.0 < float(p[0]) <= TMAX and key(p) in basekeys)
print("comprovacio: dades afegides que ja son al base (solapament) =", overlap, "(ha de ser 0)")
print("  -> el base <=1s (diagonals depurades) es mante intacte; nomes s'afegeixen 1-10 s")

# --- periodes base i periodes a afegir ---
per_base = sorted(set(float(p[0]) for p in rbase))
add = [p for p in rsrc if 1.0 < float(p[0]) <= TMAX]
per_add = sorted(set(float(p[0]) for p in add))
print("periodes base:", len(per_base), "| a afegir (1 s < T <= %g s):" % TMAX, len(per_add))
print("  periodes afegits:", [round(T, 3) for T in per_add])

# --- fusionar (base + afegits), evitant duplicats ---
seen = set(key(p) for p in rbase)
merged = list(rbase)
for p in add:
    if key(p) not in seen:
        merged.append(p); seen.add(key(p))

# ordenar per periode, estacio, component (nomes estetic; ModEM ho llegeix igual)
comp_ord = {"ZXX": 0, "ZXY": 1, "ZYX": 2, "ZYY": 3}
merged.sort(key=lambda p: (float(p[0]), p[1], comp_ord.get(p[7], 9)))

periods = sorted(set(float(p[0]) for p in merged))
stations = sorted(set(p[1] for p in merged))

# --- escriure amb la capcalera del base i el recompte actualitzat ---
out = []
for h in hbase:
    t = h.split()
    if h.strip().startswith(">") and len(t) == 3 and t[1].isdigit() and t[2].isdigit():
        out.append("> %d %d" % (len(periods), len(stations)))
    else:
        out.append(h)
for p in merged:
    out.append("%s  %-8s %s %s %s %s %s  %-4s %s %s %s" %
               (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10]))
open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")

# --- resum / validacio ---
print("\nESCRIT:", OUT)
print("  periodes: %d  (rang T: %.4g - %.4g s)" % (len(periods), min(periods), max(periods)))
print("  estacions: %d" % len(stations))
print("  components totals: %d  (base %d + afegits %d)" % (len(merged), len(rbase), len(merged) - len(rbase)))
print("  linia de recompte a la capcalera: > %d %d" % (len(periods), len(stations)))
cc = {}
for p in merged:
    cc[p[7]] = cc.get(p[7], 0) + 1
print("  per component:", cc)
