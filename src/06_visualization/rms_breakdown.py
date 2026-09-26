"""
rms_breakdown.py
----------------
A partir d'un fitxer de dades observades (OBS) i la resposta del model (PRED),
tots dos en format ModEM (Full_Impedance), calcula el RMS normalitzat desglossat:

  * Taula 1 (LaTeX): RMS per ESTACIÓ i per COMPONENT, amb el total per estació
    (totes les components), el total per component (totes les estacions) i el
    total global.
  * Taula 2 (LaTeX): RMS per PERÍODE i per COMPONENT, amb els mateixos totals.
  * Figura: mapa de les estacions acolorides pel seu RMS (estil powerline_map.png).

Definició del RMS (misfit normalitzat de ModEM). Per a cada dada complexa es
compten per separat la part real i la imaginària:
    r = (d_obs - d_pred) / sigma
    RMS_grup = sqrt( ( sum r_real^2 + sum r_imag^2 ) / N_grup )
on N_grup és el nombre de valors reals del grup (2 per cada component observada).
sigma (l'error) es pren SEMPRE del fitxer OBS; el PRED de ModEM porta un error
fictici (2e15) que s'ignora. Els "totals" són RMS agregats (agrupant tots els
residus del grup), no una suma aritmètica de RMS, que no tindria sentit físic.
"""
import os
import re
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ============================ CONFIG ============================
OBS  = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_bat_NO_otliers\Arxius_ejecució\Mall_mask_no_tip_1s_ef5"
PRED = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Comparable_Tesis_Arango\Run_632_results\mallorca_17km_seafixed_ef5_cov_inv_NLCG_067.dat"

OUT_DIR   = r"C:\Users\alber\TFG\visualizations"
TEX_STA   = os.path.join(OUT_DIR, "rms_per_station.tex")
TEX_PER   = os.path.join(OUT_DIR, "rms_per_period.tex")
FIG_MAP   = os.path.join(OUT_DIR, "rms_station_map.png")
FIG_MAP_PDF = os.path.join(OUT_DIR, "rms_station_map.pdf")

COMPONENTS = ["ZXX", "ZXY", "ZYX", "ZYY"]
ERR_MAX    = 1e10          # ignore data whose error is a masking placeholder
CMAP       = "hot_r"       # white(low) -> yellow -> red -> black(high), as in the reference


# ============================ PARSER ============================
def parse_modem_dat(path):
    """Return dict keyed by (code, period, component) -> (real, imag, err),
    plus a dict code -> (x_m, y_m, lat, lon)."""
    data, coords = {}, {}
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(">"):
                continue
            p = s.split()
            if len(p) < 11:
                continue
            try:
                period = float(p[0])
                code   = p[1]
                lat, lon = float(p[2]), float(p[3])
                x_m, y_m = float(p[4]), float(p[5])
                comp   = p[7].upper()
                re_, im_, err = float(p[8]), float(p[9]), float(p[10])
            except ValueError:
                continue
            data[(code, round(period, 12), comp)] = (re_, im_, err)
            coords.setdefault(code, (x_m, y_m, lat, lon))
    return data, coords


def station_num(code):
    m = re.search(r"(\d+)", code)
    return int(m.group(1)) if m else -1


# ============================ RMS ACCUMULATION ============================
def accumulate(obs, pred):
    """Return sums of squared normalized residuals and counts, keyed for the
    station and period breakdowns, plus the sets of stations/periods."""
    sta = {}   # (code, comp) -> [sum_sq, n]
    per = {}   # (period, comp) -> [sum_sq, n]
    stations, periods = set(), set()
    for key, (o_r, o_i, err) in obs.items():
        if err <= 0 or err >= ERR_MAX:
            continue
        if key not in pred:
            continue
        p_r, p_i, _ = pred[key]
        code, period, comp = key
        sq = ((o_r - p_r) / err) ** 2 + ((o_i - p_i) / err) ** 2
        sta.setdefault((code, comp), [0.0, 0]); sta[(code, comp)][0] += sq; sta[(code, comp)][1] += 2
        per.setdefault((period, comp), [0.0, 0]); per[(period, comp)][0] += sq; per[(period, comp)][1] += 2
        stations.add(code); periods.add(period)
    return sta, per, sorted(stations, key=station_num), sorted(periods)


def rms(sum_sq, n):
    return np.sqrt(sum_sq / n) if n > 0 else np.nan


# ============================ LATEX TABLES ============================
def _fmt(v):
    return "--" if (v is None or np.isnan(v)) else f"{v:.2f}"


def write_latex_table(path, row_keys, row_label_fn, cell, row_header, caption, label):
    """Generic RMS table with a Total column and a Total row (pooled RMS)."""
    ncol = len(COMPONENTS)
    with open(path, "w", encoding="utf-8") as f:
        f.write(r"% Requereix \usepackage{booktabs}" + "\n")
        f.write(r"\begin{table}[htbp]" + "\n  \\centering" + "\n")
        f.write(f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n")
        f.write("  \\begin{tabular}{l" + "r" * (ncol + 1) + "}\n    \\toprule\n")
        f.write(f"    {row_header} & " + " & ".join(COMPONENTS) + r" & \textbf{Total} \\" + "\n")
        f.write("    \\midrule\n")
        col_sq = {c: 0.0 for c in COMPONENTS}; col_n = {c: 0 for c in COMPONENTS}
        grand_sq, grand_n = 0.0, 0
        for rk in row_keys:
            row_sq, row_n = 0.0, 0
            cells = []
            for c in COMPONENTS:
                ssq, nn = cell.get((rk, c), (0.0, 0))
                cells.append(_fmt(rms(ssq, nn)))
                row_sq += ssq; row_n += nn
                col_sq[c] += ssq; col_n[c] += nn
            grand_sq += row_sq; grand_n += row_n
            f.write(f"    {row_label_fn(rk)} & " + " & ".join(cells) +
                    f" & {_fmt(rms(row_sq, row_n))} \\\\\n")
        f.write("    \\midrule\n")
        tot_cells = [_fmt(rms(col_sq[c], col_n[c])) for c in COMPONENTS]
        f.write(r"    \textbf{Total} & " + " & ".join(tot_cells) +
                f" & {_fmt(rms(grand_sq, grand_n))} \\\\\n")
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    print(f"Saved LaTeX table: {path}")


# ============================ STATION MAP ============================
def station_map(sta, coords, stations):
    xs, ys, rr, labels = [], [], [], []
    for code in stations:
        ssq, nn = 0.0, 0
        for c in COMPONENTS:
            s, n = sta.get((code, c), (0.0, 0)); ssq += s; nn += n
        if nn == 0 or code not in coords:
            continue
        x_m, y_m, lat, lon = coords[code]
        xs.append(y_m); ys.append(x_m); rr.append(rms(ssq, nn))
        labels.append(f"{station_num(code):02d}")
    xs, ys, rr = np.array(xs), np.array(ys), np.array(rr)

    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    vmin, vmax = float(np.nanmin(rr)), float(np.nanmax(rr))
    sc = ax.scatter(xs, ys, c=rr, cmap=CMAP, vmin=vmin, vmax=vmax, s=430,
                    edgecolor="k", linewidth=1.0, zorder=3)
    # label colour: white on dark markers for readability
    span = (vmax - vmin) or 1.0
    for x, y, lab, v in zip(xs, ys, labels, rr):
        tc = "white" if (v - vmin) / span > 0.6 else "black"
        ax.text(x, y, lab, ha="center", va="center", fontsize=7.5, color=tc, zorder=4)
    ax.set_xlabel("Est Y (m)")
    ax.set_ylabel("Nord X (m)")
    ax.set_title("RMS normalitzat per estació")
    ax.grid(alpha=0.3, ls="--")
    ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("nRMS per estació")
    fig.tight_layout()
    fig.savefig(FIG_MAP, dpi=200, bbox_inches="tight")
    fig.savefig(FIG_MAP_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {FIG_MAP}")


# ============================ MAIN ============================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Parsing OBS ...");  obs, coords = parse_modem_dat(OBS)
    print("Parsing PRED ..."); pred, _ = parse_modem_dat(PRED)
    sta, per, stations, periods = accumulate(obs, pred)

    tot_sq = sum(v[0] for v in sta.values()); tot_n = sum(v[1] for v in sta.values())
    print(f"  Stations: {len(stations)}  Periods: {len(periods)}  "
          f"Matched complex data: {tot_n // 2}  ->  RMS global = {rms(tot_sq, tot_n):.3f}")

    write_latex_table(
        TEX_STA, stations, lambda c: f"mall{station_num(c):02d}", sta,
        "Estació",
        "RMS normalitzat per estació i per component del tensor d'impedància, "
        "amb el total agregat per estació (totes les components) i per component "
        "(totes les estacions).",
        "tab:rms_station")

    write_latex_table(
        TEX_PER, periods, lambda p: f"{p:.4e}", per,
        "Període (s)",
        "RMS normalitzat per període i per component del tensor d'impedància, "
        "amb el total agregat per període i per component.",
        "tab:rms_period")

    station_map(sta, coords, stations)


if __name__ == "__main__":
    main()
