"""
Static publication-quality vertical cross-sections along the MT profiles
========================================================================

Adapted from `plot_cross_sections_profiles_interactive.py` — same data
pipeline (ModEM/WS `.rho` parsing, `.dat` survey centre, station coords from
the `.edi` files, 6 km padding crop, fixed 90° CW alignment, PCA-fitted
profile lines, bilinear sampling of the 3-D model along each oblique vertical
plane, log10(rho) with jet-reversed colour scale, iso-contours per decade),
rendered as a single static scientific figure (no HTML, no heat-map smoothing).

One panel per profile: vertical resistivity section, horizontal axis =
distance along the profile (km), vertical axis = depth (km, 0..MAX_DEPTH_KM).

Outputs (PNG @ 400 dpi + vector PDF):
    C:\\Users\\alber\\TFG\\visualizations\\cross_sections_profiles_figure.png
    C:\\Users\\alber\\TFG\\visualizations\\cross_sections_profiles_figure.pdf
"""
import os
import re
import glob
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration  (kept identical to the interactive viewer where relevant)
# ---------------------------------------------------------------------------
# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))
EDI_DIR = TFG_DIR / "data" / "edi"   # one EDI file per station (see data/README.md)

RHO_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_4/Run_667/Depuracio_manual_7_NLCG_073.rho")
DAT_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_4/Run_667/Depuracio_manual_7_NLCG_073.dat")
EDI_DIR  = str(EDI_DIR)
OUT_PERP_PNG = str(TFG_DIR / "visualizations/cross_sections_perpendicular_figure.png")
OUT_PERP_PDF = str(TFG_DIR / "visualizations/cross_sections_perpendicular_figure.pdf")
OUT_PAR_PNG  = str(TFG_DIR / "visualizations/cross_sections_parallel_figure.png")
OUT_PAR_PDF  = str(TFG_DIR / "visualizations/cross_sections_parallel_figure.pdf")

PAD_THRESHOLD_M = 6_000.0
MAX_DEPTH_KM    = 1.0
COLOR_VMIN, COLOR_VMAX = 1.0, 1e4
CONTOUR_DECADES = [1, 2, 3]          # iso-contours at 10, 100, 1000 ohm.m
ROTATE_CLOCKWISE_TURNS = 1
N_SAMPLES = 160
EXTEND_KM = 0.5

PROFILES = [
    ('perfil1_perp', 'perp', [10, 4, 2, 3, 6, 5, 7, 8, 9, 24]),
    ('perfil2_perp', 'perp', [65, 1, 51, 67, 23]),
    ('perfil3_perp', 'perp', [64, 52, 66, 22]),
    ('perfil4_perp', 'perp', [63, 25]),
    ('perfil5_perp', 'perp', [62, 61, 53, 26]),
    ('perfil1_par',  'par',  [30, 32, 34, 33, 27, 28, 29, 26, 21, 24]),
    ('perfil2_par',  'par',  [42, 66, 67, 9]),
    ('perfil3_par',  'par',  [61, 52, 51, 7]),
]


# ---------------------------------------------------------------------------
# Parsers  (identical behaviour to the interactive viewer)
# ---------------------------------------------------------------------------
def parse_rho(path):
    with open(path) as f:
        lines = f.read().splitlines()
    idx = 0
    while idx < len(lines) and (lines[idx].strip().startswith('#') or lines[idx].strip() == ''):
        idx += 1
    parts = lines[idx].split()
    nx, ny, nz = int(parts[0]), int(parts[1]), int(parts[2])
    log_mode = parts[4].upper() if len(parts) >= 5 else 'LINEAR'
    idx += 1

    def read_floats(start, n):
        vals, i = [], start
        while len(vals) < n and i < len(lines):
            s = lines[i].strip()
            if s:
                vals.extend(float(v) for v in s.split())
            i += 1
        return np.array(vals[:n], dtype=float), i

    dx, idx = read_floats(idx, nx)
    dy, idx = read_floats(idx, ny)
    dz, idx = read_floats(idx, nz)
    res = np.empty((nx, ny, nz), dtype=float)
    for k in range(nz):
        layer, idx = read_floats(idx, nx * ny)
        res[:, :, k] = layer.reshape(nx, ny)
    if log_mode == 'LOGE':
        res = np.exp(res)
    leftover = []
    while idx < len(lines):
        s = lines[idx].strip()
        if s:
            leftover.extend(s.split())
        idx += 1
    origin = (0.0, 0.0, 0.0)
    if len(leftover) >= 3:
        try:
            origin = (float(leftover[0]), float(leftover[1]), float(leftover[2]))
        except ValueError:
            pass
    return dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, res=res, origin=origin, log_mode=log_mode)


def parse_dat_centre(path):
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('>'):
                p = s[1:].split()
                if len(p) == 2:
                    try:
                        lat, lon = float(p[0]), float(p[1])
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            return lat, lon
                    except ValueError:
                        pass
    return None, None


def _dms(s):
    s = s.strip().replace('"', '')
    sign = -1 if s.startswith('-') else 1
    s = s.lstrip('-+')
    p = s.split(':')
    return sign * (float(p[0]) + float(p[1]) / 60 + float(p[2]) / 3600)


def parse_edi_coords(edi_dir):
    out = {}
    for f in glob.glob(os.path.join(edi_dir, '*.edi')):
        base = os.path.basename(f)
        m = re.search(r'(\d+)', base)
        if not m:
            continue
        num = int(m.group(1))
        lat = lon = None
        with open(f, errors='ignore') as fh:
            for line in fh:
                u = line.upper()
                if 'REFLAT' in u and lat is None:
                    lat = _dms(line.split('=')[1])
                elif 'REFLONG' in u and lon is None:
                    lon = _dms(line.split('=')[1])
                if lat is not None and lon is not None:
                    break
        if lat is not None and lon is not None:
            out[num] = (lat, lon)
    return out


# ---------------------------------------------------------------------------
# 1) Read model + coordinates
# ---------------------------------------------------------------------------
rho = parse_rho(RHO_FILE)
centre_lat, centre_lon = parse_dat_centre(DAT_FILE)
edi = parse_edi_coords(EDI_DIR)
print(f"Grid {rho['nx']}x{rho['ny']}x{rho['nz']} ({rho['log_mode']}) | "
      f"centre ({centre_lat:.5f}, {centre_lon:.5f}) | {len(edi)} EDI stations")


# ---------------------------------------------------------------------------
# 2) Build aligned model grid (same prep as the depth-slice viewer)
# ---------------------------------------------------------------------------
ox, oy, oz = rho['origin']
north_nodes = ox + np.concatenate(([0.0], np.cumsum(rho['dx'])))
east_nodes  = oy + np.concatenate(([0.0], np.cumsum(rho['dy'])))
z_nodes     = oz + np.concatenate(([0.0], np.cumsum(rho['dz'])))
north_c = (north_nodes[:-1] + north_nodes[1:]) / 2
east_c  = (east_nodes[:-1]  + east_nodes[1:])  / 2
z_c     = (z_nodes[:-1]     + z_nodes[1:])     / 2

n_core = np.where(np.abs(north_c) < PAD_THRESHOLD_M)[0]
e_core = np.where(np.abs(east_c)  < PAD_THRESHOLD_M)[0]
n0, n1 = n_core[0], n_core[-1] + 1
e0, e1 = e_core[0], e_core[-1] + 1
north_km = north_c[n0:n1] / 1000.0
east_km  = east_c[e0:e1]  / 1000.0
z_km_all = z_c / 1000.0
res_core = rho['res'][n0:n1, e0:e1, :]

if ROTATE_CLOCKWISE_TURNS:
    nn, ne = res_core.shape[:2]
    if nn != ne:
        keep = min(nn, ne)
        if nn > keep:
            lo = (nn - keep) // 2
            res_core = res_core[lo:lo + keep, :, :]; north_km = north_km[lo:lo + keep]
        if ne > keep:
            lo = (ne - keep) // 2
            res_core = res_core[:, lo:lo + keep, :]; east_km = east_km[lo:lo + keep]
    res_core = np.rot90(res_core, k=ROTATE_CLOCKWISE_TURNS, axes=(0, 1))

zsel = np.where(z_km_all <= MAX_DEPTH_KM)[0]
zlast = (zsel[-1] + 2) if len(zsel) else 4
zlast = min(zlast, len(z_km_all))
z_km = z_km_all[:zlast]
log_res = np.log10(np.clip(res_core[:, :, :zlast], COLOR_VMIN, COLOR_VMAX))
nz_sel = log_res.shape[2]


# ---------------------------------------------------------------------------
# 3) Geographic helpers + station positions (survey-centre frame)
# ---------------------------------------------------------------------------
DEG_LAT = 1.0 / 111_000.0
DEG_LON = 1.0 / (111_000.0 * np.cos(np.radians(centre_lat)))

def station_en(num):
    if num not in edi:
        return None
    lat, lon = edi[num]
    e = (lon - centre_lon) / (DEG_LON * 1000.0)
    n = (lat - centre_lat) / (DEG_LAT * 1000.0)
    return e, n


# ---------------------------------------------------------------------------
# 4) Bilinear sampler along a line
# ---------------------------------------------------------------------------
def sample_section(line_e, line_n):
    out = np.full((nz_sel, len(line_e)), np.nan)
    for s in range(len(line_e)):
        e, n = line_e[s], line_n[s]
        if e < east_km[0] or e > east_km[-1] or n < north_km[0] or n > north_km[-1]:
            continue
        j = min(np.searchsorted(east_km, e) - 1, len(east_km) - 2); j = max(j, 0)
        i = min(np.searchsorted(north_km, n) - 1, len(north_km) - 2); i = max(i, 0)
        tx = (e - east_km[j]) / (east_km[j + 1] - east_km[j])
        ty = (n - north_km[i]) / (north_km[i + 1] - north_km[i])
        a = log_res[i, j, :]     * (1 - tx) + log_res[i, j + 1, :]     * tx
        b = log_res[i + 1, j, :] * (1 - tx) + log_res[i + 1, j + 1, :] * tx
        out[:, s] = a * (1 - ty) + b * ty
    return out


# ---------------------------------------------------------------------------
# 5) Build every profile
# ---------------------------------------------------------------------------
profiles = []
for name, kind, nums in PROFILES:
    present = [num for num in nums if station_en(num) is not None]
    pts = np.array([station_en(num) for num in present])
    c = pts.mean(axis=0)
    d = pts - c
    _, _, vt = np.linalg.svd(d)
    axis = vt[0]
    if np.degrees(np.arctan2(axis[0], axis[1])) % 360.0 >= 180.0:
        axis = -axis
    proj = d @ axis
    pmin = float(proj.min()) - EXTEND_KM
    pmax = float(proj.max()) + EXTEND_KM
    length = pmax - pmin
    az = np.degrees(np.arctan2(axis[0], axis[1])) % 360.0
    start = c + axis * pmin
    dist = np.linspace(0.0, length, N_SAMPLES)
    line_e = start[0] + dist * axis[0]
    line_n = start[1] + dist * axis[1]
    sec = sample_section(line_e, line_n)
    st_list = sorted(
        [{'num': int(num), 'dist': float((np.array(station_en(num)) - c) @ axis - pmin)}
         for num in present], key=lambda r: r['dist'])
    profiles.append(dict(name=name, kind=kind, az=az, length=length,
                         dist=dist, z=sec, stations=st_list))
    print(f"  {name:13s} ({kind}) n={len(present):2d} len={length:4.2f}km az={az:5.1f}deg")


# ---------------------------------------------------------------------------
# 6) Figure
# ---------------------------------------------------------------------------
def edges(centres):
    c = np.asarray(centres, float)
    e = np.empty(c.size + 1)
    e[1:-1] = (c[:-1] + c[1:]) / 2
    e[0] = c[0] - (c[1] - c[0]) / 2
    e[-1] = c[-1] + (c[-1] - c[-2]) / 2
    return e
y_edges = edges(z_km)

def pretty(name):
    m = re.match(r'perfil(\d+)_(perp|par)', name)
    sym = r'$\perp$' if m.group(2) == 'perp' else r'$\parallel$'
    return f"Perfil {m.group(1)} {sym}"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.linewidth": 0.8, "mathtext.default": "regular",
})
norm = Normalize(vmin=np.log10(COLOR_VMIN), vmax=np.log10(COLOR_VMAX))
cmap = plt.get_cmap("jet_r")


def render_figure(subset, title, row_pattern, figsize, out_png, out_pdf):
    """row_pattern: nombre de panells per fila, p.ex. [2, 2, 1] centra l'ultim
    panell solitari de la darrera fila respecte als de les files anteriors
    (en lloc de deixar-lo arraconat amb un buit al costat)."""
    nrows = len(row_pattern)
    max_cols = max(row_pattern)
    fine_ncols = max_cols * 2
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, fine_ncols, left=0.07, right=0.87, bottom=0.06, top=0.86,
                          wspace=0.16, hspace=0.55)
    flat = []
    for r, count in enumerate(row_pattern):
        seg = fine_ncols // count
        start = (fine_ncols - seg * count) // 2   # centra el bloc de la fila
        for k in range(count):
            c0 = start + k * seg
            flat.append(fig.add_subplot(gs[r, c0:c0 + seg]))

    for ax, p, panel in zip(flat, subset, "abcdefgh"):
        x_edges = edges(p['dist'])
        ax.pcolormesh(x_edges, y_edges, p['z'], cmap=cmap, norm=norm,
                      shading="flat", rasterized=True)
        cs = ax.contour(p['dist'], z_km, p['z'], levels=CONTOUR_DECADES,
                        colors="k", linewidths=0.7, alpha=0.55)
        fmt = {lv: f"{10**lv:g} " + r"$\Omega\cdot$m" for lv in CONTOUR_DECADES}
        ax.clabel(cs, fmt=fmt, fontsize=7, inline=True, inline_spacing=3)

        sx = [s['dist'] for s in p['stations']]
        ax.scatter(sx, [0] * len(sx), marker="v", s=34, facecolor="white",
                   edgecolor="k", linewidth=0.9, clip_on=False, zorder=6)
        for s in p['stations']:
            ax.text(s['dist'], -0.05 * MAX_DEPTH_KM, str(s['num']),
                    ha="center", va="bottom", fontsize=6.5, color="#0f172a")

        ax.set_xlim(0, p['length'])
        ax.set_ylim(MAX_DEPTH_KM, -0.03 * MAX_DEPTH_KM)
        ax.tick_params(direction="out", length=3, labelsize=8)
        ax.set_ylabel("Profunditat (km)", fontsize=9)
        ax.set_xlabel("Distància al llarg del perfil (km)", fontsize=9)
        ax.set_title(f"({panel})  {pretty(p['name'])}  ·  az {p['az']:.0f}°  ·  {p['length']:.1f} km",
                     fontsize=10.5, fontweight="bold", pad=16)

    # Hide any unused panels
    for ax in flat[len(subset):]:
        ax.set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.90, 0.14, 0.017, 0.70])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 1, 2, 3, 4])
    cb.set_ticklabels(["1", "10", "100", "1 000", "10 000"])
    cb.set_label(r"Resistivitat  $\rho$  ($\Omega\cdot$m)", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=13.5, fontweight="bold", x=0.47, y=0.95)

    for path, dpi in ((out_png, 400), (out_pdf, None)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print("Saved:", path)
    plt.close(fig)


perp = [p for p in profiles if p['kind'] == 'perp']
par  = [p for p in profiles if p['kind'] == 'par']

render_figure(perp,
              "Seccions verticals de resistivitat — perfils perpendiculars a la direcció de l'strike · Llucmajor (Mallorca)",
              row_pattern=[1, 2, 2], figsize=(13.5, 11.0),
              out_png=OUT_PERP_PNG, out_pdf=OUT_PERP_PDF)

render_figure(par,
              "Seccions verticals de resistivitat — perfils paral·lels a la direcció de l'strike · Llucmajor (Mallorca)",
              row_pattern=[1, 1, 1], figsize=(8.5, 11.0),
              out_png=OUT_PAR_PNG, out_pdf=OUT_PAR_PDF)
