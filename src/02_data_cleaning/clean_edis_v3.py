"""
Clean MT EDI files by removing problematic Zxy / Zyx points.

Criteria for flagging a point as bad (applied independently per component):
  1) Phase outside the physical quadrant [0, 90] (with tolerance):
     - Zxy phase should sit in [0, 90].
     - Zyx phase (raw atan2 in third quadrant) should sit in [-180, -90];
       after +180 shift it should be in [0, 90].
  2) Isolated spike: |log10(rho_a[i]) - predicted[i]| > SPIKE_DEV_THRESH,
     where predicted[i] is the log-T linear interpolation of rho_a between
     the nearest non-flagged neighbors at i-1 and i+1.
     - Distinguishes spikes (point off the line) from transitions
       (point on the line, even if the line is steep).
     - Iterated to handle consecutive spikes.

Note: there is NO cross-component discrepancy criterion. In a 2D / anisotropic
medium with no galvanic distortion (as stated in the paper for this dataset),
Zxy and Zyx differing by 1+ orders is real physics, not noise. We therefore
check each off-diagonal independently against its own period-curve.

Flagged points are written into the cleaned EDI by setting the
corresponding Z component (real, imag) AND its variance to 1.0E32.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR  = Path(r"C:\Users\alber\TFG\LLUCMAJOR_DADES_edi")
EDI_OUT    = Path(r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Smooth3_Bat_opt\Edis_clean_v3")
FIG_OUT    = Path(r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Smooth3_Bat_opt\figures_cleaned_v3")
EDI_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

FLAG_VAR      = 1.0e15   # large variance → effectively "ignored" by ModEM, but
                         # keeps Z values intact so 3DGrid / MTpy / etc still plot
                         # the apparent resistivity and phase normally.
PHASE_TOL     = 10.0     # degrees outside [0, 90] still tolerated
TREND_K       = 7        # window size (nearest neighbors in log-period)
TREND_DEG     = 2        # local polynomial degree (quadratic captures curvature)
TREND_THR_ABS = 0.7      # absolute floor for the residual threshold (factor ~5)
TREND_THR_MAD = 4.0      # multiplier on the local MAD of residuals
TREND_ITER    = 3        # iterations to refine outlier mask


# -----------------------------
# EDI parser / writer
# -----------------------------
BLOCK_RE = re.compile(
    r'(?P<header>^\s*>(?P<name>[A-Z][A-Z0-9.]*)[^\n]*?//\s*\d+[^\n]*\r?\n)'
    r'(?P<body>.*?)'
    r'(?=^\s*>)',
    re.DOTALL | re.MULTILINE,
)
NUM_RE = re.compile(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?')


def read_block(content: str, name: str):
    """Return (values, body_span) for a numerical >NAME block. body_span is
    (start, end) inside `content` for the body text only (no header)."""
    for m in BLOCK_RE.finditer(content):
        if m.group('name') == name:
            body = m.group('body')
            vals = np.array([float(v) for v in NUM_RE.findall(body)], dtype=float)
            return vals, (m.start('body'), m.end('body'))
    return None, None


def format_block_body(values: np.ndarray, per_line: int = 5) -> str:
    out = []
    for i in range(0, len(values), per_line):
        row = "  " + "  ".join(f"{v:14.5E}" for v in values[i:i+per_line])
        out.append(row)
    return "\n".join(out) + "\n"


def replace_block(content: str, name: str, new_values: np.ndarray) -> str:
    vals, span = read_block(content, name)
    if vals is None:
        return content
    new_body = format_block_body(new_values)
    return content[:span[0]] + new_body + content[span[1]:]


# -----------------------------
# Physics helpers
# -----------------------------
def rho_phase(zr: np.ndarray, zi: np.ndarray, freq: np.ndarray):
    """Apparent resistivity (Ω·m) and phase (deg) for Z in [mV/km]/[nT]."""
    period = 1.0 / freq
    z2     = zr**2 + zi**2
    rho    = 0.2 * period * z2
    phase  = np.degrees(np.arctan2(zi, zr))
    return rho, phase


def phase_in_quadrant_xy(phase_xy):
    return (phase_xy >= -PHASE_TOL) & (phase_xy <= 90.0 + PHASE_TOL)


def phase_in_quadrant_yx(phase_yx):
    """Zyx phase from atan2 should naturally fall in [-180, -90]. Bring it
    into [0, 90] by +180 shift, then check tolerance."""
    shifted = phase_yx + 180.0
    return (shifted >= -PHASE_TOL) & (shifted <= 90.0 + PHASE_TOL)


# -----------------------------
# Outlier detection  (v2)
# -----------------------------
def _trend_outliers_one_component(period, rho, k=TREND_K, deg=TREND_DEG,
                                  thr_abs=TREND_THR_ABS, thr_mad=TREND_THR_MAD,
                                  n_iter=TREND_ITER):
    """Flag points whose log10(rho) deviates from a local polynomial fit
    on `k` nearest neighbors in log-period.

    The threshold is *adaptive*: a point is flagged only when its residual
    exceeds BOTH an absolute floor (`thr_abs`) AND `thr_mad` × MAD of the
    residuals of its neighbors. This way:

      - In smooth regions (low MAD), only the absolute floor matters.
      - In genuinely scattered regions (e.g. dead-band, transition zones),
        the per-point threshold rises automatically and natural scatter is
        not mistaken for outliers.

    The fit uses degree `deg` (quadratic by default) to track real curve
    curvature (e.g. conductive-layer "bowls"), so points sitting on a real
    curve do not appear as outliers.

    The mask is refined iteratively: at each pass the regression base
    excludes already-flagged points so a single bad point cannot pollute
    its neighbours' fits.
    """
    n      = len(period)
    log_T  = np.log10(period)
    log_R  = np.log10(np.where(rho > 0, rho, 1e-30))
    bad    = np.zeros(n, dtype=bool)

    for _ in range(n_iter):
        new_bad = np.zeros(n, dtype=bool)
        for i in range(n):
            mask        = np.ones(n, dtype=bool)
            mask[i]     = False
            mask       &= ~bad
            if mask.sum() < deg + 2:
                continue
            dist            = np.abs(log_T - log_T[i])
            dist[~mask]     = np.inf
            nbr_idx         = np.argsort(dist)[:k]
            x, y            = log_T[nbr_idx], log_R[nbr_idx]
            local_deg       = deg if (np.ptp(x) > 0 and len(x) > deg) else min(1, len(x) - 1)
            if local_deg < 1:
                predicted   = np.median(y)
                residuals   = y - predicted
            else:
                coef        = np.polyfit(x, y, local_deg)
                predicted   = np.polyval(coef, log_T[i])
                residuals   = y - np.polyval(coef, x)
            # Local MAD of neighbour residuals → adaptive noise estimate
            mad             = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
            adaptive_thr    = max(thr_abs, thr_mad * mad)
            if abs(log_R[i] - predicted) > adaptive_thr:
                new_bad[i]  = True
        if np.array_equal(new_bad, bad):
            break
        bad = new_bad
    return bad


def detect_outliers(rho_xy, phase_xy, rho_yx, phase_yx, freq):
    period = 1.0 / freq
    n      = len(period)
    bad_xy = np.zeros(n, dtype=bool)
    bad_yx = np.zeros(n, dtype=bool)

    # 1) Phase outside physical quadrant (unchanged)
    bad_xy |= ~phase_in_quadrant_xy(phase_xy)
    bad_yx |= ~phase_in_quadrant_yx(phase_yx)

    # 2) Local trend deviation in log10(rho) — independent per component.
    #    This catches both "jumps" and "off-trend but in-range" outliers,
    #    while respecting real gradients (the linear fit follows them).
    bad_xy |= _trend_outliers_one_component(period, rho_xy)
    bad_yx |= _trend_outliers_one_component(period, rho_yx)

    # NOTE: the v1 reciprocal-discrepancy criterion has been removed,
    # since it falsely flagged real 2D/3D structure (TE ≠ TM splits).
    return bad_xy, bad_yx


# -----------------------------
# Plot
# -----------------------------
def plot_station(station, freq, rho_xy, phase_xy, rho_yx, phase_yx,
                 bad_xy, bad_yx, n_orig, removed_xy, removed_yx, outpath):
    period = 1.0 / freq
    log_T  = np.log10(period)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # ---- Apparent resistivity ----
    ax1.scatter(log_T[~bad_xy], np.log10(rho_xy[~bad_xy]),
                c='tab:red',  s=40, label='Zxy kept', marker='o')
    ax1.scatter(log_T[~bad_yx], np.log10(rho_yx[~bad_yx]),
                c='tab:blue', s=40, label='Zyx kept', marker='s')
    if bad_xy.any():
        ax1.scatter(log_T[bad_xy], np.log10(rho_xy[bad_xy]),
                    c='tab:red', s=110, marker='x', linewidths=2.2,
                    label='Zxy removed')
    if bad_yx.any():
        ax1.scatter(log_T[bad_yx], np.log10(rho_yx[bad_yx]),
                    c='tab:blue', s=110, marker='x', linewidths=2.2,
                    label='Zyx removed')
    ax1.set_ylabel(r'$\log_{10}\,\rho_a$  [Ω·m]')
    ax1.set_title(f'{station}    ({n_orig} periods, removed: '
                  f'{removed_xy} XY, {removed_yx} YX)')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc='best')

    # ---- Phase ----  (Zyx shifted +180 to plot in same quadrant)
    phase_yx_plot = phase_yx + 180.0
    ax2.scatter(log_T[~bad_xy], phase_xy[~bad_xy],
                c='tab:red',  s=40, marker='o')
    ax2.scatter(log_T[~bad_yx], phase_yx_plot[~bad_yx],
                c='tab:blue', s=40, marker='s')
    if bad_xy.any():
        ax2.scatter(log_T[bad_xy], phase_xy[bad_xy],
                    c='tab:red', s=110, marker='x', linewidths=2.2)
    if bad_yx.any():
        ax2.scatter(log_T[bad_yx], phase_yx_plot[bad_yx],
                    c='tab:blue', s=110, marker='x', linewidths=2.2)
    ax2.axhline(0,  color='k', lw=0.5, ls='--')
    ax2.axhline(90, color='k', lw=0.5, ls='--')
    ax2.set_xlabel(r'$\log_{10}\,T$  [s]')
    ax2.set_ylabel(r'Phase  [deg]   (Zyx + 180°)')
    ax2.set_ylim(-30, 120)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)


# -----------------------------
# Main loop
# -----------------------------
def process_one(edi_path: Path):
    content = edi_path.read_text(encoding='latin-1')
    station = edi_path.stem

    freq, _   = read_block(content, 'FREQ')
    zxyr, _   = read_block(content, 'ZXYR')
    zxyi, _   = read_block(content, 'ZXYI')
    zxy_var,_ = read_block(content, 'ZXY.VAR')
    zyxr, _   = read_block(content, 'ZYXR')
    zyxi, _   = read_block(content, 'ZYXI')
    zyx_var,_ = read_block(content, 'ZYX.VAR')

    if any(v is None for v in (freq, zxyr, zxyi, zxy_var, zyxr, zyxi, zyx_var)):
        print(f"  [skip] {station}: missing block")
        return None

    rho_xy, phase_xy = rho_phase(zxyr, zxyi, freq)
    rho_yx, phase_yx = rho_phase(zyxr, zyxi, freq)

    bad_xy, bad_yx = detect_outliers(rho_xy, phase_xy, rho_yx, phase_yx, freq)

    # Build flagged arrays: KEEP original Z values, ONLY inflate VAR.
    # This way ρₐ and φ can still be plotted normally by 3DGrid / MTpy / etc.,
    # while ModEM sees a huge error on the bad points and effectively ignores them.
    new_zxyvar = np.where(bad_xy, FLAG_VAR, zxy_var)
    new_zyxvar = np.where(bad_yx, FLAG_VAR, zyx_var)

    new_content = content
    new_content = replace_block(new_content, 'ZXY.VAR', new_zxyvar)
    new_content = replace_block(new_content, 'ZYX.VAR', new_zyxvar)

    out_edi = EDI_OUT / edi_path.name
    out_edi.write_text(new_content, encoding='latin-1')

    out_fig = FIG_OUT / f"{station}_clean.png"
    plot_station(station, freq, rho_xy, phase_xy, rho_yx, phase_yx,
                 bad_xy, bad_yx, len(freq),
                 int(bad_xy.sum()), int(bad_yx.sum()), out_fig)

    return station, len(freq), int(bad_xy.sum()), int(bad_yx.sum())


def main():
    summary = []
    edi_files = sorted(INPUT_DIR.glob('*.edi'))
    print(f"Processing {len(edi_files)} EDIs ...")
    for p in edi_files:
        res = process_one(p)
        if res is not None:
            summary.append(res)
            station, n, bx, by = res
            print(f"  {station}: n={n:3d}, removed XY={bx:2d}, YX={by:2d}")

    # Summary
    print("\n=== SUMMARY ===")
    total_pts = sum(s[1] for s in summary)
    total_xy  = sum(s[2] for s in summary)
    total_yx  = sum(s[3] for s in summary)
    print(f"Stations processed : {len(summary)}")
    print(f"Total period points: {total_pts}")
    print(f"Total Zxy removed  : {total_xy}  ({100*total_xy/total_pts:.1f}%)")
    print(f"Total Zyx removed  : {total_yx}  ({100*total_yx/total_pts:.1f}%)")

    # Stations with most removals
    print("\nTop-5 stations with most removed points (XY+YX):")
    summary.sort(key=lambda s: -(s[2] + s[3]))
    for s in summary[:5]:
        print(f"  {s[0]}: {s[2] + s[3]} removed (XY={s[2]}, YX={s[3]}, of {s[1]} periods)")

    # Save CSV summary
    csv_out = EDI_OUT.parent / 'cleaning_summary.csv'
    with csv_out.open('w') as f:
        f.write('station,n_periods,removed_xy,removed_yx\n')
        for s in sorted(summary):
            f.write(f"{s[0]},{s[1]},{s[2]},{s[3]}\n")
    print(f"\nSummary CSV: {csv_out}")


if __name__ == "__main__":
    main()
