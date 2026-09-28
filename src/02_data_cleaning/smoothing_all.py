"""
smoothing_mall63.py  (v2)
-------------------------
Detección y atenuación de outliers en componentes off-diagonal (Zxy, Zyx)
de un EDI MT mediante filtro de Hampel + LOWESS sobre log10(rho_a) y fase.

Cambios respecto a v1:
    - Umbrales mínimos absolutos (MIN_LOG_RHO_RESID, MIN_PHASE_RESID) para
      evitar falsos positivos en zonas suaves donde MAD es minúsculo.
    - Lógica AND dentro de componente: un punto se flaggea solo si Hampel
      y LOWESS COINCIDEN en rho_a, o si COINCIDEN en phase. Mucho más robusto
      que la versión OR-cualquiera-de-los-4.
    - Detección de violación de cuadrante como WARNING (no flaggea).
      Para Zxy se espera φ ∈ (0°, 90°); para Zyx, φ ∈ (-180°, -90°).
    - Bandas verdes sombreadas en el plot que muestran el cuadrante esperado.

Albert / TFG MT Mallorca — versión batch (procesa todos los EDI de una carpeta).
"""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# =========================================================================
# CONFIG
# =========================================================================
# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))
EDI_DIR = TFG_DIR / "data" / "edi"   # one EDI file per station (see data/README.md)

EDI_DIR        = Path(str(EDI_DIR))  # carpeta con .edi
GLOB_PATTERN   = "mall*.edi"                                      # patrón de archivos
OUT_DIR        = Path(str(TFG_DIR / "smoothing_out"))        # destino EDIs depurados
PLOTS_DIR      = Path(str(TFG_DIR / "smoothing_plots"))      # destino QC plots
MAKE_PLOTS     = True    # True si quieres además generar el QC plot por estación

# --- Hampel (rolling median + MAD) ---
HAMPEL_HALFWIN = 3      # ventana = 2*N+1 puntos
HAMPEL_K       = 3.5    # umbral en MADs (3.0 = agresivo, 3.5 = razonable, 4.0 = conservador)

# --- LOWESS (residuos vs ajuste suave) ---
LOWESS_FRAC    = 0.30   # fracción de puntos en regresión local
LOWESS_K       = 3.5    # umbral en MADs sobre residuos

# --- Umbrales MÍNIMOS ABSOLUTOS (evitan falsos positivos en zonas suaves) ---
MIN_LOG_RHO_RESID = 0.15   # ~41% en lineal — no flaggear desviaciones menores
MIN_PHASE_RESID   = 8.0    # grados — no flaggear desviaciones menores

# --- Lógica de combinación ---
#  'and_within_component' (recomendado):
#     outlier si (Hampel & LOWESS coinciden en rho_a) O (lo mismo en phase).
#     Exige que el punto sea inconsistente tanto local como globalmente.
#  'or_any' (versión antigua):
#     outlier si cualquiera de los 4 criterios se dispara. Más agresivo.
COMBINATION_MODE = "and_within_component"

# --- Tratamiento ---
ERROR_INFLATION = 10.0  # factor multiplicador de Z*.VAR en outliers

OUT_DIR.mkdir(parents=True, exist_ok=True)
if MAKE_PLOTS:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================================
# EDI PARSER
# =========================================================================
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")

def _extract_block(text: str, key: str):
    """Extrae el bloque numérico tras '>KEY ... //  N'."""
    pat = rf">\s*{re.escape(key)}\b[^\n]*//\s*(\d+)\s*\n([\s\S]*?)(?=\n\s*>|\Z)"
    m = re.search(pat, text)
    if not m:
        raise ValueError(f"Bloque '{key}' no encontrado en EDI.")
    raw = m.group(2)
    vals = np.array([float(x) for x in NUM_RE.findall(raw)], dtype=float)
    return raw, vals, m.span(2)

def read_edi(path: Path) -> dict:
    text = path.read_text()
    _, freq, _ = _extract_block(text, "FREQ")
    Z = {}
    for c in ["XX", "XY", "YX", "YY"]:
        _, ZR, _ = _extract_block(text, f"Z{c}R")
        _, ZI, _ = _extract_block(text, f"Z{c}I")
        _, ZV, var_span = _extract_block(text, f"Z{c}.VAR")
        Z[c] = {"R": ZR, "I": ZI, "VAR": ZV,
                "Z": ZR + 1j * ZI, "var_span": var_span}
    return {"freq": freq, "Z": Z, "text": text}

def write_var_block(values: np.ndarray, per_line: int = 5) -> str:
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i:i+per_line]
        lines.append("    " + "   ".join(f"{v:.5E}" for v in chunk))
    return "\n".join(lines) + "\n"

# =========================================================================
# FÍSICA MT
# =========================================================================
def rho_phase(Z: np.ndarray, freq: np.ndarray):
    """rho_a [Ohm·m] y phase [°] desde Z [mV/km/nT] y f [Hz]."""
    T = 1.0 / freq
    rho = 0.2 * T * np.abs(Z) ** 2
    phase = np.degrees(np.arctan2(Z.imag, Z.real))
    return rho, phase

# =========================================================================
# DETECCIÓN DE OUTLIERS
# =========================================================================
def hampel(y: np.ndarray, halfwin: int = 3, k: float = 3.0,
           min_residual: float = 0.0) -> np.ndarray:
    """Hampel con suelo absoluto: flaggea si |y - mediana_local| > max(k·sigma, min_residual)."""
    y = np.asarray(y, float)
    n = len(y)
    flag = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - halfwin), min(n, i + halfwin + 1)
        local = y[lo:hi]
        med = np.nanmedian(local)
        mad = np.nanmedian(np.abs(local - med))
        sigma = 1.4826 * mad
        threshold = max(k * sigma, min_residual)
        if abs(y[i] - med) > threshold:
            flag[i] = True
    return flag

def lowess_outliers(x: np.ndarray, y: np.ndarray,
                    frac: float = 0.3, k: float = 3.0,
                    min_residual: float = 0.0):
    """LOWESS robusto + flag por residuos. Devuelve (flag, y_smooth)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    smooth = np.full_like(y, np.nan)
    flag = np.zeros_like(y, dtype=bool)
    if valid.sum() < 5:
        return flag, smooth
    sm_out = sm.nonparametric.lowess(
        y[valid], x[valid], frac=frac, it=3, return_sorted=False
    )
    smooth[valid] = sm_out
    resid = y - smooth
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    sigma = 1.4826 * mad
    threshold = max(k * sigma, min_residual)
    flag = (np.abs(resid) > threshold) & valid
    return flag, smooth

def check_quadrant(phase: np.ndarray, comp_label: str) -> np.ndarray:
    """Marca puntos fuera del cuadrante 1D-canónico esperado.
    NO flaggea como outlier — solo warning para inspección visual.
       Zxy: φ ∈ (0°, 90°)
       Zyx: φ ∈ (-180°, -90°)  [convención de los datos mall*]
    """
    if comp_label == "ZXY":
        return (phase < 0) | (phase > 90)
    elif comp_label == "ZYX":
        return (phase > -90) | (phase < -180)
    return np.zeros_like(phase, dtype=bool)

# =========================================================================
# PIPELINE PRINCIPAL
# =========================================================================
def process_component(Zc, var, freq, label):
    rho, phase = rho_phase(Zc, freq)
    log_rho = np.log10(rho)
    log_T   = np.log10(1.0 / freq)

    # Hampel con suelo absoluto
    f_h_rho   = hampel(log_rho, HAMPEL_HALFWIN, HAMPEL_K, MIN_LOG_RHO_RESID)
    f_h_phase = hampel(phase,   HAMPEL_HALFWIN, HAMPEL_K, MIN_PHASE_RESID)

    # LOWESS con suelo absoluto
    f_l_rho,   sm_rho   = lowess_outliers(log_T, log_rho,
                                          LOWESS_FRAC, LOWESS_K, MIN_LOG_RHO_RESID)
    f_l_phase, sm_phase = lowess_outliers(log_T, phase,
                                          LOWESS_FRAC, LOWESS_K, MIN_PHASE_RESID)

    # Combinación
    if COMBINATION_MODE == "and_within_component":
        flag_rho   = f_h_rho   & f_l_rho      # ambos métodos en rho_a
        flag_phase = f_h_phase & f_l_phase    # ambos métodos en phase
        flag = flag_rho | flag_phase
    else:
        flag = f_h_rho | f_h_phase | f_l_rho | f_l_phase

    # Warning de cuadrante (NO flaggea)
    quad_violation = check_quadrant(phase, label)

    var_new = var.copy()
    var_new[flag] *= ERROR_INFLATION

    return {
        "label": label,
        "rho": rho, "phase": phase,
        "log_rho": log_rho, "log_T": log_T,
        "smooth_log_rho": sm_rho, "smooth_phase": sm_phase,
        "flag": flag,
        "flag_h_rho": f_h_rho, "flag_h_phase": f_h_phase,
        "flag_l_rho": f_l_rho, "flag_l_phase": f_l_phase,
        "quad_violation": quad_violation,
        "var_old": var, "var_new": var_new,
    }

# =========================================================================
# QC PLOT
# =========================================================================
def qc_plot(freq, Z, results, station_id, out_path):
    T = 1.0 / freq
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    components = ["XY", "YX"]
    color = {"XY": "tab:blue", "YX": "tab:red"}
    quad_band = {"XY": (0, 90), "YX": (-180, -90)}

    for j, comp in enumerate(components):
        r = results[comp]
        sigma_Z = np.sqrt(np.maximum(Z[comp]["VAR"], 0))
        absZ = np.abs(Z[comp]["Z"])
        rho_err   = 0.4 * T * absZ * sigma_Z
        phase_err = np.degrees(sigma_Z / np.where(absZ > 0, absZ, np.nan))
        order = np.argsort(r["log_T"])

        # --- panel rho_a ---
        ax = axes[0, j]
        ax.errorbar(T, r["rho"], yerr=rho_err, fmt="o", ms=5,
                    color="gray", ecolor="lightgray", alpha=0.7, label="data")
        ax.plot(T[order], 10 ** r["smooth_log_rho"][order],
                "-", lw=2, color=color[comp], label="LOWESS")
        # warning de cuadrante también en rho_a (cuadrado naranja hueco)
        if r["quad_violation"].any():
            ax.plot(T[r["quad_violation"]], r["rho"][r["quad_violation"]],
                    "s", ms=11, mew=2, mfc="none", color="darkorange",
                    label="off-quadrant warn")
        if r["flag"].any():
            ax.plot(T[r["flag"]], r["rho"][r["flag"]], "x",
                    ms=14, mew=2.5, color="red", label="outlier")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(r"$\rho_a$ (Ω·m)")
        ax.set_title(f"Z{comp}  —  apparent resistivity")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9, loc="best")

        # --- panel fase ---
        ax = axes[1, j]
        # banda del cuadrante esperado
        lo, hi = quad_band[comp]
        ax.axhspan(lo, hi, color="green", alpha=0.06, zorder=0,
                   label=f"1D quadrant ({lo}°,{hi}°)")
        ax.errorbar(T, r["phase"], yerr=phase_err, fmt="o", ms=5,
                    color="gray", ecolor="lightgray", alpha=0.7)
        ax.plot(T[order], r["smooth_phase"][order], "-", lw=2, color=color[comp])
        if r["quad_violation"].any():
            ax.plot(T[r["quad_violation"]], r["phase"][r["quad_violation"]],
                    "s", ms=11, mew=2, mfc="none", color="darkorange",
                    label="off-quadrant warn")
        if r["flag"].any():
            ax.plot(T[r["flag"]], r["phase"][r["flag"]], "x",
                    ms=14, mew=2.5, color="red", label="outlier")
        ax.set_xscale("log")
        ax.set_xlabel("Period T (s)")
        ax.set_ylabel(r"$\varphi$ (°)")
        ax.set_title(f"Z{comp}  —  phase")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"{station_id}  —  off-diagonal outlier detection  "
        f"[mode: {COMBINATION_MODE}]\n"
        f"Hampel(halfwin={HAMPEL_HALFWIN}, k={HAMPEL_K}) | "
        f"LOWESS(frac={LOWESS_FRAC}, k={LOWESS_K}) | "
        f"min_resid: log_ρ={MIN_LOG_RHO_RESID}, φ={MIN_PHASE_RESID}° | "
        f"VAR ×{ERROR_INFLATION:.0f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)    # libera memoria; no abre ventana
    print(f"[QC plot] -> {out_path}")

# =========================================================================
# ESCRITURA EDI DEPURADO
# =========================================================================
def write_depurated_edi(edi_text, Z, results, out_path):
    new_text = edi_text
    edits = []
    for comp in ["XY", "YX"]:
        span = Z[comp]["var_span"]
        new_block = write_var_block(results[comp]["var_new"])
        edits.append((span, new_block))
    edits.sort(key=lambda e: e[0][0], reverse=True)
    for (a, b), block in edits:
        new_text = new_text[:a] + block + new_text[b:]
    out_path.write_text(new_text)
    print(f"[EDI depurado] -> {out_path}")

# =========================================================================
# MAIN
# =========================================================================
def process_one_edi(edi_path: Path, out_dir: Path, make_plot: bool = False) -> dict:
    """Procesa un único EDI. Devuelve dict con stats + ruta del EDI depurado."""
    station_id = edi_path.stem
    edi = read_edi(edi_path)
    freq = edi["freq"]; Z = edi["Z"]

    results = {
        "XY": process_component(Z["XY"]["Z"], Z["XY"]["VAR"], freq, "ZXY"),
        "YX": process_component(Z["YX"]["Z"], Z["YX"]["VAR"], freq, "ZYX"),
    }

    if make_plot:
        qc_plot(freq, Z, results, station_id,
                PLOTS_DIR / f"{station_id}_QC_offdiag.png")

    out_edi = out_dir / f"{station_id}_smoothed.edi"
    write_depurated_edi(edi["text"], Z, results, out_edi)

    return {
        "station": station_id,
        "n_freq": len(freq),
        "n_xy_outliers": int(results["XY"]["flag"].sum()),
        "n_yx_outliers": int(results["YX"]["flag"].sum()),
        "n_xy_quad":     int(results["XY"]["quad_violation"].sum()),
        "n_yx_quad":     int(results["YX"]["quad_violation"].sum()),
        "out_path": out_edi,
    }


def main():
    edi_files = sorted(EDI_DIR.glob(GLOB_PATTERN))
    print(f"== Batch smoothing off-diagonal | "
          f"modo: {COMBINATION_MODE} | k={HAMPEL_K} ==")
    print(f"  Origen:   {EDI_DIR}")
    print(f"  Destino:  {OUT_DIR}")
    print(f"  Patrón:   {GLOB_PATTERN}  ->  {len(edi_files)} archivos")
    print(f"  Plots:    {'sí' if MAKE_PLOTS else 'no'}")
    print()

    if not edi_files:
        print(f"⚠ No se encontró ningún archivo {GLOB_PATTERN} en {EDI_DIR}")
        return

    print(f"{'estación':<12} {'nf':>4} | "
          f"{'XY out':>6} {'XY %':>6} {'XY quad':>7} | "
          f"{'YX out':>6} {'YX %':>6} {'YX quad':>7}")
    print("-" * 72)

    summary, failed = [], []
    for edi_path in edi_files:
        try:
            s = process_one_edi(edi_path, OUT_DIR, MAKE_PLOTS)
            n = s["n_freq"]
            print(f"{s['station']:<12} {n:>4} | "
                  f"{s['n_xy_outliers']:>6} "
                  f"{100*s['n_xy_outliers']/n:>5.1f}% "
                  f"{s['n_xy_quad']:>7} | "
                  f"{s['n_yx_outliers']:>6} "
                  f"{100*s['n_yx_outliers']/n:>5.1f}% "
                  f"{s['n_yx_quad']:>7}")
            summary.append(s)
        except Exception as e:
            print(f"{edi_path.stem:<12}  ✗ ERROR: {e}")
            failed.append((edi_path.name, str(e)))

    print("-" * 72)
    if summary:
        total_pts = sum(s["n_freq"] for s in summary)
        total_xy  = sum(s["n_xy_outliers"] for s in summary)
        total_yx  = sum(s["n_yx_outliers"] for s in summary)
        print(f"TOTAL: {len(summary)} estaciones procesadas, "
              f"{total_pts} puntos por componente")
        print(f"       ZXY outliers: {total_xy:4d} ({100*total_xy/total_pts:.1f}%)")
        print(f"       ZYX outliers: {total_yx:4d} ({100*total_yx/total_pts:.1f}%)")
    if failed:
        print(f"\n⚠ Estaciones falladas: {len(failed)}")
        for name, err in failed:
            print(f"   {name}: {err}")

    # CSV resumen
    csv_path = OUT_DIR / "smoothing_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("station,n_freq,xy_outliers,xy_quad,yx_outliers,yx_quad,output_edi\n")
        for s in summary:
            f.write(f"{s['station']},{s['n_freq']},"
                    f"{s['n_xy_outliers']},{s['n_xy_quad']},"
                    f"{s['n_yx_outliers']},{s['n_yx_quad']},"
                    f"{s['out_path'].name}\n")
    print(f"\n[CSV resumen] -> {csv_path}")
    print("== Done ==")


if __name__ == "__main__":
    main()
