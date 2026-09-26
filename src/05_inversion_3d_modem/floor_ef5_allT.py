#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica un error floor del 5% a TOTES les components (ZXX, ZXY, ZYX, ZYY)
d'un fitxer de dades ModEM, SENSE retallar cap periode.
El floor es calcula, per a cada estacio i periode, sobre
    floor = 0.05 * sqrt(|Zxy * Zyx|)
i s'aplica com  error = max(error_original, floor).
Si a un periode/estacio falta Zxy o Zyx, s'usa el modul de la component
antidiagonal disponible; si falten totes dues, no es toca.

Us:  python floor_ef5_allT.py  ENTRADA  SORTIDA
"""
import math, sys

IN  = sys.argv[1]
OUT = sys.argv[2]
EF  = 0.05     # 5%

with open(IN) as f:
    lines = f.readlines()

header, data = [], []
for ln in lines:
    s = ln.strip()
    if s.startswith("#") or s.startswith(">"):
        header.append(ln)
    elif s:
        data.append(ln)

fields = lambda ln: ln.split()

# floor per (estacio, periode) a partir de |Zxy|, |Zyx|
zabs = {}
for ln in data:
    p = fields(ln)
    per, code, comp = p[0], p[1], p[7]
    if comp in ("ZXY", "ZYX"):
        zabs.setdefault((code, per), {})[comp] = math.hypot(float(p[8]), float(p[9]))

def floor_for(code, per):
    d = zabs.get((code, per), {})
    axy, ayx = d.get("ZXY"), d.get("ZYX")
    if axy is not None and ayx is not None: return EF*math.sqrt(axy*ayx)
    if axy is not None: return EF*axy
    if ayx is not None: return EF*ayx
    return None

out_data, n_floored = [], 0
for ln in data:
    p = fields(ln)
    fl = floor_for(p[1], p[0])
    err = float(p[10])
    if fl is not None and fl > err:
        err = fl; n_floored += 1
    out_data.append(
        "%-13s %-13s %13.6f %13.6f %11.3f %13.3f %13.3f     %-4s %16.6E %16.6E %16.6E\n"
        % (p[0], p[1], float(p[2]), float(p[3]), float(p[4]), float(p[5]),
           float(p[6]), p[7], float(p[8]), float(p[9]), err))

# capçalera: mateix nombre de periodes/estacions (no es retalla res)
periods = sorted({float(fields(ln)[0]) for ln in data})
codes   = {fields(ln)[1] for ln in data}
new_header = []
for h in header:
    t = h.split()
    if h.strip().startswith(">") and len(t) == 3 and t[1].isdigit() and t[2].isdigit():
        new_header.append("> %d %d\n" % (len(periods), len(codes)))
    else:
        new_header.append(h)

with open(OUT, "w") as f:
    f.writelines(new_header); f.writelines(out_data)

print("Sortida:      %s" % OUT)
print("Periodes:     %d  (rang %.6g - %.6g s)  [SENSE retall]" % (len(periods), periods[0], periods[-1]))
print("Estacions:    %d" % len(codes))
print("Linies dades: %d" % len(out_data))
print("Errors elevats al floor: %d de %d (%.0f%%)" % (n_floored, len(out_data), 100.0*n_floored/len(out_data)))
