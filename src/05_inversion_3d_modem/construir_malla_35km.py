#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construeix una malla ModEM nova a partir de Model_Bat:
  - Extensio horitzontal reduida (~35 km) mantenint el CENTRE  -> inclou el mar (costa a ~13.6 km).
  - Profunditat dimensionada per la banda <=10 s i les resistivitats de la zona
    (aquitard 5-20 ohm-m; zones resistives ~180 ohm-m) -> fons ~52 km.
  - Nucli fi sobre les estacions; padding geometric fins a la vora.
  - Resistivitats (mar 0.3, unitat 30 ohm-m, fons 100) remapejades per vei mes proper.
  - Covariancia sea-fixed (mask 9 al mar) compatible.

Us:  python construir_malla_35km.py
"""
import numpy as np, math

SRC        = "Model_Bat"
OUT_MODEL  = "Model_Bat_35km_recond"
OUT_COV    = "Model_Bat_35km_recond_cov_seafixed"

# --------- parametres de disseny ---------
EXT_H     = 35000.0     # extensio horitzontal total (m)
CORE_DX   = 150.0       # mida de cel.la al nucli (m)
N_CORE    = 50          # nombre de cel.les de nucli per eix  (50*150=7.5 km, +/-3.75 km)
PAD_F     = 1.5         # factor de creixement del padding horitzontal
DEPTH     = 52000.0     # fons de malla (m)  ~3x skin depth a 8.66 s i 180 ohm-m
Z0        = 12.0        # gruix de la 1a capa (m)
ZF_FINE   = 1.2         # creixement a la zona fina
Z_FINE_TO = 2500.0      # profunditat fins on manté resolucio fina (m)
ZF_PAD    = 1.55        # creixement del padding vertical
# -----------------------------------------

def nums(l): return [float(x) for x in l.split()]
def centers(d, o):
    e = o + np.concatenate([[0], np.cumsum(d)]); return 0.5*(e[:-1]+e[1:])
def geom(first, f, n): return [first*f**i for i in range(1, n+1)]
def pad_to(remain, first, f):
    c = []
    while sum(c) < remain: c.append((c[-1] if c else first)*f if c else first)
    return [x*remain/sum(c) for x in c]      # escala per encaixar exacte

# ---- llegir model original ----
F = open(SRC).read().split("\n")
h = F[1].split(); Nx, Ny, Nz = int(h[0]), int(h[1]), int(h[2])
loge = h[4] if len(h) > 4 else "LOGE"
dx, dy, dz = nums(F[2]), nums(F[3]), nums(F[4])
rows = [nums(F[i]) for i in range(5, len(F)) if len(F[i].split()) == Nx][:Nx*Nz]
A = np.array(rows).reshape(Nz, Nx, Ny)
tail = [l for l in F if l.strip() != ""]
ox, oy, oz = nums(tail[-2]); rot = float(tail[-1])
LN100, LN30, LN03 = math.log(100.), math.log(30.), math.log(0.3)

# centre original (per conservar-lo)
cxc = ox + sum(dx)/2.0
cyc = oy + sum(dy)/2.0

# ---- nou grid horitzontal (centrat), extensio EXT_H ----
core = [CORE_DX]*N_CORE
rem = (EXT_H - sum(core))/2.0
pad = pad_to(rem, CORE_DX*PAD_F, PAD_F)
newdx = pad[::-1] + core + pad
newdy = list(newdx)
nox = cxc - sum(newdx)/2.0          # origen que conserva el centre
noy = cyc - sum(newdy)/2.0
Nx2 = Ny2 = len(newdx)

# ---- nou grid vertical ----
zt = [Z0]
while sum(zt) < Z_FINE_TO: zt.append(zt[-1]*ZF_FINE)
zp = []
while sum(zt)+sum(zp) < DEPTH: zp.append((zp[-1] if zp else zt[-1])*ZF_PAD)
newdz = zt + zp
Nz2 = len(newdz)

# ---- remap resistivitats (vei mes proper en coordenades) ----
orc, occ, ozc = centers(dx, ox), centers(dy, oy), centers(dz, oz)
nrc, ncc, nzc = centers(newdx, nox), centers(newdy, noy), centers(newdz, oz)
ir = np.array([np.abs(orc-v).argmin() for v in nrc])
ic = np.array([np.abs(occ-v).argmin() for v in ncc])
iz = np.array([np.abs(ozc-v).argmin() for v in nzc])
B = A[np.ix_(iz, ir, ic)]

# ---- escriure model ----
def frow(v): return " ".join("%12.5E" % x for x in v)
with open(OUT_MODEL, "w") as o:
    o.write("# Malla 35km, fons ~52km, mar remapejat (des de Model_Bat)\n")
    o.write(" %d  %d  %d  0 %s\n" % (Nx2, Ny2, Nz2, loge))
    o.write(" ".join("%.3f" % v for v in newdx) + "\n")
    o.write(" ".join("%.3f" % v for v in newdy) + "\n")
    o.write(" ".join("%.3f" % v for v in newdz) + "\n")
    for k in range(Nz2):
        o.write("\n")
        for r in range(Nx2):
            o.write(frow(B[k, r, :]) + "\n")
    o.write("%.3f    %.3f    %.3f\n" % (nox, noy, oz))
    o.write("%.1f\n" % rot)

# ---- covariancia sea-fixed (mask 9 al mar) ----
sea = (np.abs(B - LN03) < 1e-3)
SMOOTH, NAPP = 0.2, 2
HED = ("+-----------------------------------------------------------------------------+\n"
"| This file defines model covariance for a recursive autoregression scheme.   |\n"
"| The model space may be divided into distinct areas using integer masks.     |\n"
"| Mask 0 is reserved for air; mask 9 is reserved for ocean. Smoothing between |\n"
"| air, ocean and the rest of the model is turned off automatically. You can   |\n"
"| also define exceptions to override smoothing between any two model areas.   |\n"
"| To turn off smoothing set it to zero. This header is 16 lines long.         |\n"
"| 1. Grid dimensions excluding air layers (Nx, Ny, NzEarth)                   |\n"
"| 2. Smoothing in the X direction (NzEarth real values)                       |\n"
"| 3. Smoothing in the Y direction (NzEarth real values)                       |\n"
"| 4. Vertical smoothing (1 real value)                                        |\n"
"| 5. Number of times the smoothing should be applied (1 integer >= 0)         |\n"
"| 6. Number of exceptions (1 integer >= 0)                                    |\n"
"| 7. Exceptions in the form e.g. 2 3 0. (to turn off smoothing between 2 & 3) |\n"
"| 8. Two integer layer indices and Nx x Ny block of masks, repeated as needed.|\n"
"+-----------------------------------------------------------------------------+\n")
with open(OUT_COV, "w") as o:
    o.write(HED)
    o.write("\n %d            %d            %d \n\n" % (Nx2, Ny2, Nz2))
    o.write(" " + "   ".join("%.1f" % SMOOTH for _ in range(Nz2)) + "  \n")
    o.write(" " + "   ".join("%.1f" % SMOOTH for _ in range(Nz2)) + "  \n")
    o.write(" %.1f \n\n" % SMOOTH)
    o.write(" %d \n\n" % NAPP)
    o.write(" 0 \n\n\n")
    for k in range(Nz2):
        o.write(" %d             %d \n" % (k+1, k+1))
        for r in range(Nx2):
            o.write(" " + "  ".join("9" if sea[k, r, c] else "1" for c in range(Ny2)) + " \n")

# ---- informe ----
print("ORIGINAL : %dx%dx%d  ext %.1f km  fons %.1f km" % (Nx,Ny,Nz,sum(dx)/1e3,sum(dz)/1e3))
print("NOVA     : %dx%dx%d  ext %.1f km  fons %.1f km" % (Nx2,Ny2,Nz2,sum(newdx)/1e3,sum(newdz)/1e3))
print("nucli %.0f m (%d cel.les)  1a capa %.0f m  capes top<=600m: %d  <=2km: %d"
      % (CORE_DX, N_CORE, newdz[0], int(np.sum(np.cumsum(newdz)<=600)), int(np.sum(np.cumsum(newdz)<=2000))))
print("centre conservat: X %.0f->%.0f  Y %.0f->%.0f" % (cxc, nox+sum(newdx)/2, cyc, noy+sum(newdy)/2))
u,c = np.unique(np.round(B,4), return_counts=True)
lab={round(LN100,4):"100 (fons)",round(LN30,4):"30",round(LN03,4):"0.3 (MAR)"}
print("cel.les remapejades:")
for v,n in zip(u,c): print("   %-12s: %d"%(lab.get(round(float(v),4),"%.1f ohm"%math.exp(v)),n))
print("MAR capturat: %d cel.les (era 0 en 17km!)" % int(sea.sum()))
print("escrits: %s , %s"%(OUT_MODEL,OUT_COV))
