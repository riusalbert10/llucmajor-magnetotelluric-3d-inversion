#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera un nou fitxer de dades ModEM a partir d'un altre:
  1) Retalla els periodes > 10 s.
  2) Aplica un error floor del 5% a TOTES les components (ZXX, ZXY, ZYX, ZYY).
     El floor es calcula, per a cada estacio i periode, sobre
        floor = 0.05 * sqrt(|Zxy * Zyx|)
     i s'aplica com  error = max(error_original, floor).
     Si a un periode/estacio falta Zxy o Zyx, s'usa el modul de la
     component antidiagonal disponible; si falten totes dues, no es toca.
"""
import math, sys

IN  = sys.argv[1]
OUT = sys.argv[2]
PMAX = 10.0     # s
EF   = 0.05     # 5%

with open(IN) as f:
    lines = f.readlines()

# separar capçalera (linies > i #) de les dades
header, data = [], []
for ln in lines:
    s = ln.strip()
    if s.startswith("#") or s.startswith(">"):
        header.append(ln)
    elif s:
        data.append(ln)

def fields(ln):
    return ln.split()

# 1) retall de periodes
kept = [ln for ln in data if float(fields(ln)[0]) <= PMAX]

# 2) calcul del floor per (estacio, periode) a partir de Zxy, Zyx
#    clau = (code, period_str)
zabs = {}   # clau -> {"ZXY":|Z|, "ZYX":|Z|}
for ln in kept:
    p = fields(ln)
    per, code, comp = p[0], p[1], p[7]
    re, im = float(p[8]), float(p[9])
    if comp in ("ZXY", "ZYX"):
        zabs.setdefault((code, per), {})[comp] = math.hypot(re, im)

def floor_for(code, per):
    d = zabs.get((code, per), {})
    axy, ayx = d.get("ZXY"), d.get("ZYX")
    if axy is not None and ayx is not None:
        return EF * math.sqrt(axy * ayx)
    if axy is not None:
        return EF * axy
    if ayx is not None:
        return EF * ayx
    return None

# 3) reescriure aplicant el floor, conservant l'amplada de columnes
out_data = []
n_floored = 0
for ln in kept:
    p = fields(ln)
    per, code = p[0], p[1]
    fl = floor_for(code, per)
    err = float(p[10])
    if fl is not None and fl > err:
        err = fl
        n_floored += 1
    # reconstruir amb format consistent (mateix estil que ModEM)
    out_data.append(
        "%-13s %-13s %13.6f %13.6f %11.3f %13.3f %13.3f     %-4s %16.6E %16.6E %16.6E\n"
        % (per, code, float(p[2]), float(p[3]), float(p[4]), float(p[5]),
           float(p[6]), p[7], float(p[8]), float(p[9]), err))

# 4) actualitzar el recompte de periodes a la capçalera ( > nPer nStat )
periods = sorted({float(fields(ln)[0]) for ln in kept})
codes   = {fields(ln)[1] for ln in kept}
nPer, nStat = len(periods), len(codes)

new_header = []
for h in header:
    t = h.split()
    if h.strip().startswith(">") and len(t) == 3 and t[1].isdigit() and t[2].isdigit():
        new_header.append("> %d %d\n" % (nPer, nStat))
    else:
        new_header.append(h)

with open(OUT, "w") as f:
    f.writelines(new_header)
    f.writelines(out_data)

print("Entrada:      %s" % IN)
print("Sortida:      %s" % OUT)
print("Periodes:     %d  (rang %.6g - %.6g s)" % (nPer, periods[0], periods[-1]))
print("Estacions:    %d" % nStat)
print("Linies dades: %d" % len(out_data))
print("Errors elevats al floor: %d de %d (%.0f%%)"
      % (n_floored, len(out_data), 100.0*n_floored/len(out_data)))
