#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Justificacio de la profunditat de malla per condicions de contorn en MT.

Genera:
  - contorno_skindepth.txt : resum de la derivacio i les taules numeriques.
  - contorno_skindepth.png : |E(z)|/E0 = exp(-z/delta) vs profunditat, per T=10,100,2278 s,
    amb el fons de malla (68 km) i els criteris de contorn (5%, 1%).

Us:  python contorno_skindepth.py
"""
import numpy as np, math
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --------- parametres ---------
MU0   = 4e-7*math.pi          # H/m
RHO   = 180.0                 # ohm-m (resistivitat maxima -> cas pitjor)
RHO2  = 100.0                 # ohm-m (fons)
H     = 68.0e3                # m, fons de malla actual
PERIODS = [10, 30, 100, 300, 1000, 2278]   # s
EPS   = [0.05, 0.01]          # residus tolerats al contorn
# ------------------------------

def skin(rho, T):             # skin depth (m)
    return math.sqrt(rho*T/(math.pi*MU0))     # = 503*sqrt(rho*T)

# ---------- TXT ----------
L = []
L.append("="*72)
L.append(" CONDICIONS DE CONTORN I PROFUNDITAT DE MALLA EN MAGNETOTELLURICA")
L.append("="*72)
L.append("")
L.append("1) EQUACIO DE DIFUSIO (quasi-estatic, sigma >> omega*eps):")
L.append("      d2E/dz2 = i*omega*mu0*sigma * E")
L.append("")
L.append("2) SOLUCIO ONA PLANA (decaiment amb la profunditat):")
L.append("      E(z) = E(0) * exp(-z/delta) * exp(-i z/delta)")
L.append("      |E(z)|/|E(0)| = exp(-z/delta)")
L.append("")
L.append("3) SKIN DEPTH:")
L.append("      delta = sqrt(2*rho/(omega*mu0)) = sqrt(rho*T/(pi*mu0)) ~= 503*sqrt(rho*T) [m]")
L.append("")
L.append("4) CRITERI DE CONTORN:")
L.append("      el camp ha de decaure a una fraccio eps al fons z=H:")
L.append("          exp(-H/delta) <= eps   =>   H >= delta * ln(1/eps)")
L.append("      eps=5%%  -> ln(20)=3.0  -> H >= 3.0*delta")
L.append("      eps=1%%  -> ln(100)=4.6 -> H >= 4.6*delta")
L.append("      (d'aqui la 'regla del 3-5x skin depth')")
L.append("")
L.append("5) RESISTIVITAT: delta ~ sqrt(rho) -> mes penetracio en el medi MES RESISTIU")
L.append("   -> s'usa rho_max (cas pitjor). Aqui rho_max = %.0f ohm-m." % RHO)
L.append("")
L.append("-"*72)
L.append("TAULA 1 - Skin depth delta(T)  [km]")
L.append("  %-8s %12s %12s" % ("T (s)", "delta@100", "delta@180"))
for T in PERIODS:
    L.append("  %-8.4g %10.1f km %10.1f km" % (T, skin(RHO2,T)/1e3, skin(RHO,T)/1e3))
L.append("")
L.append("TAULA 2 - Residu de camp al fons actual H=%.0f km:  exp(-H/delta)  (rho=%.0f)" % (H/1e3, RHO))
L.append("  %-8s %10s %14s %10s" % ("T (s)", "delta(km)", "H/delta", "residu"))
for T in PERIODS:
    d = skin(RHO,T); r = math.exp(-H/d)
    flag = "OK" if r <= 0.05 else ("marginal" if r <= 0.15 else "MALAMENT")
    L.append("  %-8.4g %10.1f %14.2f %9.1f%%  %s" % (T, d/1e3, H/d, 100*r, flag))
L.append("")
L.append("TAULA 3 - Profunditat requerida H_min = delta*ln(1/eps)  [km]  (rho=%.0f)" % RHO)
L.append("  %-8s %14s %14s" % ("T (s)", "H(eps=5%)", "H(eps=1%)"))
for T in PERIODS:
    d = skin(RHO,T)
    L.append("  %-8.4g %11.0f km %11.0f km" % (T, d*math.log(1/0.05)/1e3, d*math.log(1/0.01)/1e3))
L.append("")
L.append("-"*72)
L.append("CONCLUSIO: la malla de 68 km compleix el contorn nomes fins a ~10 s")
L.append("(residu ~4%%). A 100 s el residu es 36%% i a la banda completa 81%%: el camp")
L.append("no ha decaigut i la condicio de contorn contamina la solucio.")
L.append("")
L.append("FORMULA RESUM:  H >= ln(1/eps) * sqrt(rho_max*T_max/(pi*mu0))")
L.append("                  ~= ln(1/eps) * 503 * sqrt(rho_max*T_max)")
open("contorno_skindepth.txt","w").write("\n".join(L)+"\n")

# ---------- FIGURA ----------
z = np.linspace(0, 150e3, 500)     # profunditat 0-150 km
fig, ax = plt.subplots(figsize=(8,5))
cols = {10:"tab:green",100:"tab:orange",2278:"tab:red"}
for T in (10,100,2278):
    d = skin(RHO,T)
    ax.semilogy(z/1e3, np.exp(-z/d), color=cols[T], lw=2,
                label="T=%g s  (delta=%.0f km)"%(T,d/1e3))
    r = math.exp(-H/d)
    ax.plot(H/1e3, r, "o", color=cols[T], ms=7)
    ax.annotate("%.0f%%"%(100*r), (H/1e3, r), textcoords="offset points",
                xytext=(8,0), color=cols[T], fontweight="bold", va="center")
ax.axvline(H/1e3, color="k", ls="--", lw=1.5, label="fons malla = 68 km")
for e,lab in [(0.05,"5%"),(0.01,"1%")]:
    ax.axhline(e, color="grey", ls=":", lw=1)
    ax.text(2, e*1.15, "criteri "+lab, color="grey", fontsize=8)
ax.set_xlabel("profunditat z (km)")
ax.set_ylabel(r"$|E(z)|/|E_0| = e^{-z/\delta}$")
ax.set_title(r"Decaiment del camp i condicio de contorn ($\rho$=180 $\Omega\cdot$m)")
ax.set_ylim(1e-3, 1.2); ax.set_xlim(0,150)
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig("contorno_skindepth.png", dpi=130); plt.close(fig)

print("escrits: contorno_skindepth.txt , contorno_skindepth.png")
for T in (10,100,2278):
    print("  T=%-6g s: delta=%.0f km, residu al fons de 68 km = %.0f%%"%(T,skin(RHO,T)/1e3,100*math.exp(-H/skin(RHO,T))))
