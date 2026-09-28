"""
Static publication-quality depth-slice figure of the 3D resistivity model
=========================================================================

Adapted from `plot_depth_slices_interactive.py` — same data pipeline
(ModEM/WS `.rho` parsing, `.dat` header for coordinates, 5 km padding crop,
fixed 90° CW alignment, nearest real layer per target depth, log10(rho) with
jet-reversed colour scale, one iso-contour per decade), rendered as a single
static scientific figure (no HTML, no vertical interpolation, no heat-map
smoothing).

Seven panels: horizontal slices at 0, 100, 200, 300, 400, 500 and 600 m depth,
arranged in a wide, horizontal layout: 4+3 (second row centred under the
first), read left->right, top->bottom by increasing depth.

Outputs (PNG @ 400 dpi + vector PDF):
    C:\\Users\\alber\\TFG\\visualizations\\depth_slices_figure.png
    C:\\Users\\alber\\TFG\\visualizations\\depth_slices_figure.pdf
"""
import os
import re
import numpy as np
import shapely
from shapely.geometry import MultiPoint
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator, FuncFormatter
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration  (kept identical to the interactive viewer where relevant)
# ---------------------------------------------------------------------------
# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))

RHO_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_4/Run_667/Depuracio_manual_7_NLCG_073.rho")
DAT_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_Depuracio_manual_4/Run_667/Depuracio_manual_7_NLCG_073.dat")
OUT_PNG  = str(TFG_DIR / "visualizations/depth_slices_figure.png")
OUT_PDF  = str(TFG_DIR / "visualizations/depth_slices_figure.pdf")

PAD_THRESHOLD_M = 5_000.0            # crop padding cells beyond this distance (m) from centre
COLOR_VMIN, COLOR_VMAX = 1.0, 1e4    # resistivity colour-scale range (ohm·m, log)
CONTOUR_DECADES = [1, 2, 3]          # iso-contours at 10, 100, 1000 ohm·m (log10 = 1,2,3)
ROTATE_CLOCKWISE_TURNS = 1           # validated data-orientation correction (do not change)
DEPTH_TARGETS_M = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
STATION_MARGIN_KM = 0.3              # margin around the outer-station polygon (km) = 300 m


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
    origin, rotation = (0.0, 0.0, 0.0), 0.0
    if len(leftover) >= 3:
        try:
            origin = (float(leftover[0]), float(leftover[1]), float(leftover[2]))
        except ValueError:
            pass
    if len(leftover) >= 4:
        try:
            rotation = float(leftover[3])
        except ValueError:
            pass
    return dict(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                res=res, origin=origin, rotation=rotation, log_mode=log_mode)


def parse_dat_header(path):
    centre_lat = centre_lon = None
    stations = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('>'):
                parts = s[1:].split()
                if len(parts) == 2:
                    try:
                        lat, lon = float(parts[0]), float(parts[1])
                        if -90 <= lat <= 90 and -180 <= lon <= 180 and centre_lat is None:
                            centre_lat, centre_lon = lat, lon
                    except ValueError:
                        pass
                continue
            if s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) >= 11:
                sid = parts[1].split('_')[0]
                if sid in stations:
                    continue
                try:
                    stations[sid] = dict(name=sid, lat=float(parts[2]), lon=float(parts[3]),
                                         x_m=float(parts[4]), y_m=float(parts[5]))
                except ValueError:
                    pass
    return centre_lat, centre_lon, list(stations.values())


# ---------------------------------------------------------------------------
# 1) Read + build grid (metres relative to survey centre), crop, align
# ---------------------------------------------------------------------------
rho = parse_rho(RHO_FILE)
centre_lat, centre_lon, stations = parse_dat_header(DAT_FILE)
print(f"Grid {rho['nx']}x{rho['ny']}x{rho['nz']} ({rho['log_mode']}) | "
      f"centre ({centre_lat:.5f}, {centre_lon:.5f}) | {len(stations)} stations")

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

# Nearest real model layer per target depth (no vertical interpolation)
sel = [int(np.argmin(np.abs(z_c - t))) for t in DEPTH_TARGETS_M]
log_res = np.log10(np.clip(res_core, COLOR_VMIN, COLOR_VMAX))

# Cell-edge coordinates for crisp pcolormesh (no smoothing)
def edges(centres):
    c = np.asarray(centres, float)
    e = np.empty(c.size + 1)
    e[1:-1] = (c[:-1] + c[1:]) / 2
    e[0] = c[0] - (c[1] - c[0]) / 2
    e[-1] = c[-1] + (c[-1] - c[-2]) / 2
    return e
x_edges, y_edges = edges(east_km), edges(north_km)

# Sobre-mostreig "nearest-neighbour" NOMES per calcular els isocontorns.
# ax.contour() interpola linealment entre centres de cel·la i dona corbes
# suaus, mentre que pcolormesh(shading="flat") pinta blocs rectangulars secs;
# aixo feia que la linia de contorn no seguis la vora real del bloc de color
# (poca correlacio amb la barra de color). Repetint cada cel·la N cops es
# construeix un camp esglaonat identic al de pcolormesh, i el contorn calculat
# sobre aquest camp SI que ressegueix exactament les vores dels blocs.
CONTOUR_UPSAMPLE = 6
def upsample_nn_centres(centres, factor):
    e = edges(centres)
    return np.concatenate([np.linspace(e[i], e[i + 1], factor, endpoint=False)
                            + (e[i + 1] - e[i]) / (2 * factor)
                            for i in range(len(centres))])
east_km_hi  = upsample_nn_centres(east_km, CONTOUR_UPSAMPLE)
north_km_hi = upsample_nn_centres(north_km, CONTOUR_UPSAMPLE)
def repeat_nn(Z, factor):
    return np.repeat(np.repeat(Z, factor, axis=0), factor, axis=1)

# Geographic tick helpers
DEG_LAT = 1.0 / 111_000.0
DEG_LON = 1.0 / (111_000.0 * np.cos(np.radians(centre_lat)))
km_to_lat = lambda km: centre_lat + km * 1000.0 * DEG_LAT
km_to_lon = lambda km: centre_lon + km * 1000.0 * DEG_LON

st_x = np.array([s['y_m'] / 1000.0 for s in stations])  # east km
st_y = np.array([s['x_m'] / 1000.0 for s in stations])  # north km

# Poligon que delimita el mapa d'estacions: envolupant convexa de les
# estacions, eixamplada 100 m cap enfora (vertexs rectes, junta "mitre").
station_hull = MultiPoint(np.column_stack([st_x, st_y])).convex_hull
boundary_poly = station_hull.buffer(STATION_MARGIN_KM, join_style="mitre")
poly_x, poly_y = boundary_poly.exterior.xy
poly_x, poly_y = np.asarray(poly_x), np.asarray(poly_y)
poly_xlim = (poly_x.min(), poly_x.max())
poly_ylim = (poly_y.min(), poly_y.max())

# Retallem les dades: tot el que quedi fora del poligon es marca com a NaN,
# aixi el pcolormesh (i els contorns) no es dibuixen fora del perimetre.
Xc, Yc = np.meshgrid(east_km, north_km)             # shape (north, east), com log_res[:,:,k]
inside_poly = shapely.contains_xy(boundary_poly, Xc, Yc)
log_res = np.where(inside_poly[:, :, None], log_res, np.nan)

m = re.search(r'_(\d+)\.rho$', os.path.basename(RHO_FILE))
iter_no = (m.group(1).lstrip('0') or '0') if m else '?'
model_label = os.path.basename(os.path.dirname(os.path.dirname(RHO_FILE)))


# ---------------------------------------------------------------------------
# 2) Figure — 2x2x2x1 grid (last, odd-one-out panel centred under the rest),
#    reading order left->right, top->bottom by increasing depth.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.linewidth": 0.8, "mathtext.default": "regular",
})
norm = Normalize(vmin=np.log10(COLOR_VMIN), vmax=np.log10(COLOR_VMAX))
cmap = plt.get_cmap("jet_r")

def station_num(name):
    """'mall02' -> '02': nomes la part numerica del codi d'estacio."""
    m_num = re.search(r'\d+', name)
    return m_num.group(0) if m_num else name

# Etiquetem nomes un subconjunt d'estacions, distribuit espacialment (una per
# cel·la ocupada d'una graella grossa NxN sobre l'area), no totes les 40.
N_LABEL_BINS = 4
bx_edges = np.linspace(poly_xlim[0], poly_xlim[1], N_LABEL_BINS + 1)
by_edges = np.linspace(poly_ylim[0], poly_ylim[1], N_LABEL_BINS + 1)
bin_ix = np.clip(np.digitize(st_x, bx_edges) - 1, 0, N_LABEL_BINS - 1)
bin_iy = np.clip(np.digitize(st_y, by_edges) - 1, 0, N_LABEL_BINS - 1)
seen = {}
for i, key in enumerate(zip(bin_ix, bin_iy)):
    seen.setdefault(key, i)
label_idx = sorted(seen.values())

lon_fmt = FuncFormatter(lambda x, pos: f"{km_to_lon(x):.3f}°E")
lat_fmt = FuncFormatter(lambda y, pos: f"{km_to_lat(y):.3f}°N")

n_targets = len(DEPTH_TARGETS_M)
row_pattern = [4, 3]                            # disposicio horitzontal en dues files
assert sum(row_pattern) == n_targets
nrows = len(row_pattern)
max_cols = max(row_pattern)
fine_ncols = max_cols * 2

fig = plt.figure(figsize=(17.0, 7.6))
gs = fig.add_gridspec(nrows, fine_ncols, left=0.05, right=0.91, bottom=0.09, top=0.87,
                      wspace=0.9, hspace=0.6)
ordered = []
for r, count in enumerate(row_pattern):
    seg = fine_ncols // count
    start = (fine_ncols - seg * count) // 2     # centra el bloc de la fila
    for c in range(count):
        c0 = start + c * seg
        ordered.append(fig.add_subplot(gs[r, c0:c0 + seg]))

xlim, ylim = poly_xlim, poly_ylim               # el poligon d'estacions + marge

for ax, t, panel in zip(ordered, DEPTH_TARGETS_M, "abcdefg"):
    k = int(np.argmin(np.abs(z_c - t)))
    Z = log_res[:, :, k]
    ax.pcolormesh(x_edges, y_edges, Z, cmap=cmap, norm=norm, shading="flat",
                  rasterized=True)
    # Iso-contours per decade, labelled in ohm·m — calculats sobre el camp
    # esglaonat (nearest-neighbour) perque segueixin la vora real dels blocs
    # de color, no una corba suavitzada per interpolacio lineal.
    Z_hi = repeat_nn(Z, CONTOUR_UPSAMPLE)
    cs = ax.contour(east_km_hi, north_km_hi, Z_hi, levels=CONTOUR_DECADES,
                    colors="k", linewidths=0.7, alpha=0.55)
    fmt = {lv: f"{10**lv:g} " + r"$\Omega\cdot$m" for lv in CONTOUR_DECADES}
    ax.clabel(cs, fmt=fmt, fontsize=7, inline=True, inline_spacing=3)
    # MT stations
    ax.scatter(st_x, st_y, marker="v", s=26, facecolor="white",
               edgecolor="k", linewidth=0.8, zorder=5)
    # Numero d'estacio (nomes el numero, en negreta) en un subconjunt
    # distribuit (evita saturar el mapa)
    for i in label_idx:
        ax.annotate(station_num(stations[i]['name']), (st_x[i], st_y[i]),
                    textcoords="offset points", xytext=(3.5, 3.5),
                    fontsize=6.5, fontweight="bold", color="#1f2937", zorder=6)
    # Perimetre del poligon de retall (envolupant convexa + marge)
    ax.plot(poly_x, poly_y, color="k", lw=0.9, ls="--", alpha=0.7, zorder=4)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(lon_fmt)
    ax.yaxis.set_major_formatter(lat_fmt)
    ax.tick_params(direction="out", length=3, labelsize=7.5)
    ax.tick_params(axis="x", labelrotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    # Sense requadre: nomes els eixos esquerre i inferior (origen a baix a
    # l'esquerra), marcant les coordenades horitzontal i vertical.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Longitud", fontsize=8.5)
    ax.set_ylabel("Latitud", fontsize=8.5)
    ax.set_title(f"({panel})  z = {t:.0f} m",
                 fontsize=11, fontweight="bold", pad=6)
    # North arrow (top-right)
    ax.annotate("N", xy=(0.94, 0.97), xytext=(0.94, 0.80),
                xycoords="axes fraction", ha="center", va="center",
                fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1))

# Shared colorbar (log resistivity)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cax = fig.add_axes([0.94, 0.20, 0.013, 0.60])
cb = fig.colorbar(sm, cax=cax)
cb.set_ticks([0, 1, 2, 3, 4])
cb.set_ticklabels(["1", "10", "100", "1 000", "10 000"])
cb.set_label(r"Resistivitat  $\rho$  ($\Omega\cdot$m)", fontsize=10)
cb.ax.tick_params(labelsize=8)

fig.suptitle("Model de resistivitat magnetotel·lúrica 3-D de la zona de Llucmajor (Mallorca) — talls horitzontals de profunditat",
             fontsize=13.5, fontweight="bold", x=0.47, y=0.975)

for ext, path in ((".png", OUT_PNG), (".pdf", OUT_PDF)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=400 if ext == ".png" else None, bbox_inches="tight")
    print("Saved:", path)
