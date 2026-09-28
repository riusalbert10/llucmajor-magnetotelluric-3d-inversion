#!/usr/bin/env python3
"""
Script para analizar la dimensionalidad de datos magnetoteluricos

Determina si las estructuras geologicas son:
- 1D: Capas horizontales (variacion solo con profundidad)
- 2D: Estructuras alargadas (ej: falla, dique) con una direccion preferencial
- 3D: Estructuras complejas (variacion en todas direcciones)

Metodos utilizados:
1. Skew (sesgo) de Swift - Indica desviacion de modelo 2D
2. Elipticidad de la fase
3. Analisis del determinante del tensor
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mtpy.core.mt import MT
from pathlib import Path

# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))
EDI_DIR = TFG_DIR / "data" / "edi"   # one EDI file per station (see data/README.md)

edi_file = str(EDI_DIR / "mall67.edi")
output_dir = str(TFG_DIR)

print("="*80)
print("ANALISIS DE DIMENSIONALIDAD - Estacion mall67")
print("="*80)

# Leer archivo EDI con MTpy
print("\n1. Leyendo archivo EDI con MTpy...")
mt_obj = MT()
mt_obj.read(edi_file)

print(f"   Estacion: {mt_obj.station}")
print(f"   Coordenadas: {mt_obj.latitude:.6f}N, {mt_obj.longitude:.6f}E")
print(f"   Numero de frecuencias: {len(mt_obj.frequency)}")
print(f"   Rango de frecuencias: {mt_obj.frequency.min():.4f} - {mt_obj.frequency.max():.1f} Hz")

# Obtener tensor de impedancia
freq = mt_obj.frequency
periodo = 1.0 / freq
Z = mt_obj.Z

print("\n" + "="*80)
print("METODO 1: SKEW DE SWIFT")
print("="*80)
print("\nEl skew mide la asimetria del tensor de impedancias.")
print("- Skew < 0.2: Estructura 1D o 2D")
print("- 0.2 < Skew < 0.4: Estructura 2D con efectos 3D")
print("- Skew > 0.4: Estructura 3D")

# Calcular skew de Swift manualmente
# Skew = |Zxx + Zyy| / |Zxy - Zyx|
Zxx = Z.z[:, 0, 0]
Zxy = Z.z[:, 0, 1]
Zyx = Z.z[:, 1, 0]
Zyy = Z.z[:, 1, 1]

numerator = np.abs(Zxx + Zyy)
denominator = np.abs(Zxy - Zyx)
skew = numerator / denominator

print(f"\nEstadisticas del Skew:")
print(f"  Minimo: {np.nanmin(skew):.3f}")
print(f"  Maximo: {np.nanmax(skew):.3f}")
print(f"  Media: {np.nanmean(skew):.3f}")
print(f"  Mediana: {np.nanmedian(skew):.3f}")

# Clasificar segun skew promedio
skew_mean = np.nanmean(skew)
if skew_mean < 0.2:
    dimension_skew = "1D/2D"
    interp_skew = "Estructura simple de capas horizontales o estructura 2D bien definida"
elif skew_mean < 0.4:
    dimension_skew = "2D con efectos 3D"
    interp_skew = "Estructura principalmente 2D con heterogeneidades laterales"
else:
    dimension_skew = "3D"
    interp_skew = "Estructura geologica compleja tridimensional"

print(f"\nCLASIFICACION (Skew): {dimension_skew}")
print(f"Interpretacion: {interp_skew}")

print("\n" + "="*80)
print("METODO 2: ANALISIS DE DIAGONAL vs OFF-DIAGONAL")
print("="*80)
print("\nEn una estructura 1D/2D ideal, los elementos diagonales (Zxx, Zyy)")
print("deberian ser mucho menores que los off-diagonal (Zxy, Zyx)")

# Las componentes ya fueron obtenidas arriba (Zxx, Zxy, Zyx, Zyy)

# Calcular magnitudes
mag_diag = np.sqrt(np.abs(Zxx)**2 + np.abs(Zyy)**2)
mag_offdiag = np.sqrt(np.abs(Zxy)**2 + np.abs(Zyx)**2)

# Ratio diagonal/off-diagonal (menor es mejor para 1D/2D)
ratio_diag = mag_diag / mag_offdiag

print(f"\nRatio (Diagonal / Off-diagonal):")
print(f"  Minimo: {np.nanmin(ratio_diag):.3f}")
print(f"  Maximo: {np.nanmax(ratio_diag):.3f}")
print(f"  Media: {np.nanmean(ratio_diag):.3f}")
print(f"  Mediana: {np.nanmedian(ratio_diag):.3f}")

ratio_mean = np.nanmean(ratio_diag)
if ratio_mean < 0.2:
    dimension_ratio = "1D/2D"
    interp_ratio = "Estructura simple 1D o 2D bien orientada"
elif ratio_mean < 0.5:
    dimension_ratio = "2D"
    interp_ratio = "Estructura 2D con posible rotacion necesaria"
else:
    dimension_ratio = "3D"
    interp_ratio = "Estructura 3D o 2D mal orientada"

print(f"\nCLASIFICACION (Ratio): {dimension_ratio}")
print(f"Interpretacion: {interp_ratio}")

print("\n" + "="*80)
print("METODO 3: ELIPTICIDAD DE LA FASE")
print("="*80)
print("\nLa elipticidad mide la diferencia entre las fases de xy y yx")
print("- Elipticidad baja: Estructura 1D")
print("- Elipticidad moderada: Estructura 2D")
print("- Elipticidad alta: Estructura 3D")

# Calcular fases
phase = Z.phase

phase_xy = phase[:, 0, 1]
phase_yx = phase[:, 1, 0]

# Diferencia de fase (convertir a rango -180 a 180)
phase_diff = phase_xy - phase_yx
phase_diff = np.where(phase_diff > 180, phase_diff - 360, phase_diff)
phase_diff = np.where(phase_diff < -180, phase_diff + 360, phase_diff)

# Elipticidad (beta en literatura MT)
# beta = (phase_xy - phase_yx) / 2
beta = phase_diff / 2.0

print(f"\nElipticidad (beta):")
print(f"  Minima: {np.nanmin(beta):.2f} grados")
print(f"  Maxima: {np.nanmax(beta):.2f} grados")
print(f"  Media: {np.nanmean(beta):.2f} grados")
print(f"  Mediana: {np.nanmedian(beta):.2f} grados")

beta_mean = np.abs(np.nanmean(beta))
if beta_mean < 5:
    dimension_beta = "1D"
    interp_beta = "Estructura de capas horizontales"
elif beta_mean < 15:
    dimension_beta = "2D"
    interp_beta = "Estructura elongada con direccion preferencial"
else:
    dimension_beta = "3D"
    interp_beta = "Estructura compleja tridimensional"

print(f"\nCLASIFICACION (Elipticidad): {dimension_beta}")
print(f"Interpretacion: {interp_beta}")

print("\n" + "="*80)
print("METODO 4: STRIKE ANGLE (Direccion de estructura 2D)")
print("="*80)
print("\nEl strike angle indica la direccion de estructuras elongadas")
print("Un strike constante sugiere estructura 2D en esa direccion")

# Calcular strike angle manualmente usando el metodo de Swift
# Strike = 0.5 * arctan(2 * Re(Zxy + Zyx) / (|Zxx|^2 - |Zyy|^2))
try:
    # Calcular numerador y denominador para el strike
    numerator = 2 * np.real(Zxy + Zyx)
    denominator = np.abs(Zxx)**2 - np.abs(Zyy)**2

    # Calcular strike en radianes y convertir a grados
    # Agregar pequeño offset para evitar division por cero
    pt_strike = 0.5 * np.arctan2(numerator, denominator + 1e-10) * 180 / np.pi

    # Normalizar al rango [0, 90] grados
    pt_strike = np.where(pt_strike < 0, pt_strike + 90, pt_strike)
    pt_strike = np.where(pt_strike > 90, pt_strike - 90, pt_strike)

    print(f"\nPhase Tensor Strike:")
    print(f"  Minimo: {np.nanmin(pt_strike):.1f} grados")
    print(f"  Maximo: {np.nanmax(pt_strike):.1f} grados")
    print(f"  Media: {np.nanmean(pt_strike):.1f} grados")
    print(f"  Desviacion std: {np.nanstd(pt_strike):.1f} grados")

    pt_std = np.nanstd(pt_strike)
    if pt_std < 10:
        strike_interp = f"Strike bien definido ~{np.nanmean(pt_strike):.1f}° (estructura 2D)"
    elif pt_std < 30:
        strike_interp = f"Strike variable (estructura 2D con complejidad)"
    else:
        strike_interp = "Strike muy variable (estructura 3D)"

    print(f"\nInterpretacion: {strike_interp}")
    has_strike = True
except Exception as e:
    print(f"\nNo se pudo calcular phase tensor strike: {str(e)}")
    has_strike = False

print("\n" + "="*80)
print("RESUMEN DE DIMENSIONALIDAD")
print("="*80)

print(f"\nMetodo 1 (Skew):              {dimension_skew}")
print(f"Metodo 2 (Ratio Diag/Off):    {dimension_ratio}")
print(f"Metodo 3 (Elipticidad):       {dimension_beta}")

# Votar por dimensionalidad dominante
dimensiones = [dimension_skew, dimension_ratio, dimension_beta]
if dimensiones.count("1D") >= 2 or dimensiones.count("1D/2D") >= 2:
    dim_final = "1D/2D"
    color_final = "verde (simple)"
elif dimensiones.count("2D") >= 2 or dimensiones.count("2D con efectos 3D") >= 1:
    dim_final = "2D"
    color_final = "amarillo (moderado)"
else:
    dim_final = "3D"
    color_final = "rojo (complejo)"

print(f"\nDIMENSIONALIDAD DOMINANTE: {dim_final}")

if dim_final == "1D/2D":
    print("\nCONCLUSION:")
    print("Los datos sugieren una estructura geologica principalmente 1D o 2D simple.")
    print("Recomendacion: Modelado 1D o 2D es apropiado.")
elif dim_final == "2D":
    print("\nCONCLUSION:")
    print("Los datos sugieren una estructura geologica 2D.")
    if has_strike:
        print(f"Direccion preferencial (strike): ~{np.nanmean(pt_strike):.1f} grados")
    print("Recomendacion: Modelado 2D con rotacion a strike principal.")
else:
    print("\nCONCLUSION:")
    print("Los datos sugieren una estructura geologica compleja 3D.")
    print("Recomendacion: Modelado 3D completo o inversion con varias estaciones.")

# Generar graficos
print("\n" + "="*80)
print("GENERANDO GRAFICOS DE DIMENSIONALIDAD")
print("="*80)

fig = plt.figure(figsize=(15, 8))

# Grafico 1: Skew vs Period
ax1 = plt.subplot(2, 3, 1)
ax1.semilogx(periodo, skew, 'o-', color='darkblue', markersize=5, linewidth=1.5)
ax1.axhline(y=0.2, color='green', linestyle='--', linewidth=2, label='1D/2D limit')
ax1.axhline(y=0.4, color='orange', linestyle='--', linewidth=2, label='2D/3D limit')
ax1.set_xlabel('Period (s)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Swift Skew', fontsize=11, fontweight='bold')
ax1.set_title('Skew vs Period', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# Grafico 2: Ratio Diagonal/Off-diagonal vs Period
ax2 = plt.subplot(2, 3, 2)
ax2.semilogx(periodo, ratio_diag, 'o-', color='darkred', markersize=5, linewidth=1.5)
ax2.axhline(y=0.2, color='green', linestyle='--', linewidth=2, label='1D/2D limit')
ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='2D/3D limit')
ax2.set_xlabel('Period (s)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Ratio (Diagonal/Off-diagonal)', fontsize=11, fontweight='bold')
ax2.set_title('Diagonal Ratio vs Period', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# Grafico 3: Ellipticity vs Period
ax3 = plt.subplot(2, 3, 3)
ax3.semilogx(periodo, np.abs(beta), 'o-', color='darkgreen', markersize=5, linewidth=1.5)
ax3.axhline(y=5, color='green', linestyle='--', linewidth=2, label='1D limit')
ax3.axhline(y=15, color='orange', linestyle='--', linewidth=2, label='2D limit')
ax3.set_xlabel('Period (s)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Ellipticity |beta| (degrees)', fontsize=11, fontweight='bold')
ax3.set_title('Ellipticity vs Period', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# Grafico 4: Phase Tensor Strike vs Period
ax4 = plt.subplot(2, 3, 4)
if has_strike:
    ax4.semilogx(periodo, pt_strike, 'o-', color='purple', markersize=5, linewidth=1.5)
    ax4.set_ylabel('Strike Angle (degrees)', fontsize=11, fontweight='bold')
    ax4.set_title('Phase Tensor Strike vs Period', fontsize=12, fontweight='bold')
    ax4.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax4.axhline(y=90, color='gray', linestyle=':', linewidth=1, alpha=0.5)
else:
    ax4.text(0.5, 0.5, 'Strike not available', ha='center', va='center',
             transform=ax4.transAxes, fontsize=12)
ax4.set_xlabel('Period (s)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Grafico 5: Apparent Resistivities (reference)
ax5 = plt.subplot(2, 3, 5)
rho_xy = Z.resistivity[:, 0, 1]
rho_yx = Z.resistivity[:, 1, 0]
ax5.loglog(periodo, rho_xy, 'o-', color='blue', label='rho_xy (TE)',
           markersize=5, linewidth=1.5)
ax5.loglog(periodo, rho_yx, 's-', color='red', label='rho_yx (TM)',
           markersize=5, linewidth=1.5)
ax5.set_xlabel('Period (s)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Apparent Resistivity (Ohm*m)', fontsize=11, fontweight='bold')
ax5.set_title('Apparent Resistivities (Reference)', fontsize=12, fontweight='bold')
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, alpha=0.3)

plt.suptitle(f'Dimensionality Analysis - Station {mt_obj.station}',
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Guardar figura
output_file = os.path.join(output_dir, 'analisis_dimensionalidad.png')
fig.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nGrafico guardado en: {output_file}")

print("\n" + "="*80)
print("ANALISIS COMPLETADO")
print("="*80)

if __name__ == '__main__':
    # No mostrar graficos interactivos (solo guardar)
    # plt.show()
    pass
