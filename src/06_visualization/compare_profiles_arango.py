"""
Comparison figure: perpendicular MT cross-sections vs. reference study (Arango)
===============================================================================

Places the reference resistivity profiles (grayscale image) side by side with
the perpendicular cross-sections of the FINAL chosen model (Depuració Manual 2,
Run 632, comparable with the reference study). Same data pipeline as
`plot_cross_sections_profiles_figure.py`, but cropped to 700 m depth and limited
to the three perpendicular profiles requested, stacked bottom-to-top as 1, 2, 5
to mirror the reference layout (Profile 1 / 2 / 3 from bottom to top).

Outputs (PNG @ 400 dpi + vector PDF):
    C:\\Users\\alber\\TFG\\visualizations\\compare_profiles_arango_figure.png
    C:\\Users\\alber\\TFG\\visualizations\\compare_profiles_arango_figure.pdf
"""
import os
import re
import glob
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RHO_FILE = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Depuracio_manual_4\Run_667\Depuracio_manual_7_NLCG_073.rho"
DAT_FILE = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Depuracio_manual_4\Run_667\Depuracio_manual_7_NLCG_073.dat"
EDI_DIR  = r"C:\Users\alber\TFG\LLUCMAJOR_DADES_edi"
REF_IMG  = r"C:\Users\alber\TFG\visualizations\arango_reference.png"
OUT_PNG  = r"C:\Users\alber\TFG\visualizations\compare_profiles_arango_figure.png"
OUT_PDF  = r"C:\Users\alber\TFG\visualizations\compare_profiles_arango_figure.pdf"

PAD_THRESHOLD_M = 6_000.0
MAX_DEPTH_KM    = 0.7                 # cropped to match the reference (0-600/700 m)
COLOR_VMIN, COLOR_VMAX = 1.0, 1e4
CONTOUR_DECADES = [1, 2, 3]
ROTATE_CLOCKWISE_TURNS = 1
N_SAMPLES = 160
EXTEND_KM = 0.5

# Perpendicular profiles, listed TOP -> BOTTOM in the right column
# (so bottom-to-top reads 1, 2, 5, mirroring reference Profile 1/2/3).
PROFILES = [
    ('perfil5_perp', 'perp', [62, 61, 53, 26]),   # top    (~ ref Profile 3)
    ('perfil2_perp', 'perp', [65, 1, 51, 67, 23]), # middle (~ ref Profile 2)
    ('perfil1_perp', 'perp', [10, 4, 2, 3, 6, 5, 7, 8, 9, 24]),  # bottom (~ ref Profile 1)
]


# ---------------------------------------------------------------------------
# Parsers
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
        m = re.search(r'(\d+)', os.path.basename(f))
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
# Read model + coordinates + build aligned grid
# ---------------------------------------------------------------------------
rho = parse_rho(RHO_FILE)
centre_lat, centre_lon = parse_dat_centre(DAT_FILE)
edi = parse_edi_coords(EDI_DIR)
print(f"Grid {rho['nx']}x{rho['ny']}x{rho['nz']} ({rho['log_mode']}) | "
      f"centre ({centre_lat:.5f}, {centre_lon:.5f}) | {len(edi)} EDI stations")

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

DEG_LAT = 1.0 / 111_000.0
DEG_LON = 1.0 / (111_000.0 * np.cos(np.radians(centre_lat)))

def station_en(num):
    if num not in edi:
        return None
    lat, lon = edi[num]
    return ((lon - centre_lon) / (DEG_LON * 1000.0),
            (lat - centre_lat) / (DEG_LAT * 1000.0))


def sample_section(line_e, line_n):
    out = np.full((nz_sel, len(line_e)), np.nan)
    for s in range(len(line_e)):
        e, n = line_e[s], line_n[s]
        if e < east_km[0] or e > east_km[-1] or n < north_km[0] or n > north_km[-1]:
            continue
        j = max(min(np.searchsorted(east_km, e) - 1, len(east_km) - 2), 0)
        i = max(min(np.searchsorted(north_km, n) - 1, len(north_km) - 2), 0)
        tx = (e - east_km[j]) / (east_km[j + 1] - east_km[j])
        ty = (n - north_km[i]) / (north_km[i + 1] - north_km[i])
        a = log_res[i, j, :]     * (1 - tx) + log_res[i, j + 1, :]     * tx
        b = log_res[i + 1, j, :] * (1 - tx) + log_res[i + 1, j + 1, :] * tx
        out[:, s] = a * (1 - ty) + b * ty
    return out


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
    pmin, pmax = float(proj.min()) - EXTEND_KM, float(proj.max()) + EXTEND_KM
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
    print(f"  {name:13s} n={len(present):2d} len={length:4.2f}km az={az:5.1f}deg")


# ---------------------------------------------------------------------------
# Figure: reference image (left) + 3 perpendicular sections (right)
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
    n = re.match(r'perfil(\d+)_', name).group(1)
    return f"Perfil {n} " + r"$\perp$"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.linewidth": 0.8, "mathtext.default": "regular"})
norm = Normalize(vmin=np.log10(COLOR_VMIN), vmax=np.log10(COLOR_VMAX))
cmap = plt.get_cmap("jet_r")

fig = plt.figure(figsize=(15, 8.2))
gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.05],
                      left=0.045, right=0.9, bottom=0.07, top=0.9,
                      hspace=0.42, wspace=0.13)

# Left: reference image spanning the three rows
ax_ref = fig.add_subplot(gs[:, 0])
ax_ref.imshow(mpimg.imread(REF_IMG))
ax_ref.set_axis_off()
ax_ref.set_title("(Arango Galván 2005)", fontsize=11, fontweight="bold", pad=6)

# Right: three perpendicular sections, top->bottom = 5,2,1
for row, p in enumerate(profiles):
    ax = fig.add_subplot(gs[row, 1])
    x_edges = edges(p['dist'])
    ax.pcolormesh(x_edges, y_edges * 1000.0, p['z'], cmap=cmap, norm=norm,
                  shading="flat", rasterized=True)
    cs = ax.contour(p['dist'], z_km * 1000.0, p['z'], levels=CONTOUR_DECADES,
                    colors="k", linewidths=0.7, alpha=0.55)
    fmt = {lv: f"{10**lv:g} " + r"$\Omega\cdot$m" for lv in CONTOUR_DECADES}
    ax.clabel(cs, fmt=fmt, fontsize=7, inline=True, inline_spacing=3)

    sx = [s['dist'] for s in p['stations']]
    ax.scatter(sx, [0] * len(sx), marker="v", s=30, facecolor="white",
               edgecolor="k", linewidth=0.9, clip_on=False, zorder=6)
    for s in p['stations']:
        ax.text(s['dist'], -35, str(s['num']), ha="center", va="bottom",
                fontsize=6.5, color="#0f172a")

    ax.set_xlim(0, p['length'])
    ax.set_ylim(MAX_DEPTH_KM * 1000.0, -0.03 * MAX_DEPTH_KM * 1000.0)
    ax.tick_params(direction="out", length=3, labelsize=8)
    ax.set_ylabel("Profunditat (m)", fontsize=9)
    ax.set_title(f"{pretty(p['name'])}  ·  az {p['az']:.0f}°", fontsize=10.5,
                 fontweight="bold", pad=12)
    ax.text(0.99, 0.06, "S 40 E", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, fontstyle="italic", color="#333")
    if row == len(profiles) - 1:
        ax.set_xlabel("Distància al llarg del perfil (km)", fontsize=9)

# Shared colorbar
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cax = fig.add_axes([0.915, 0.12, 0.015, 0.7])
cb = fig.colorbar(sm, cax=cax)
cb.set_ticks([0, 1, 2, 3, 4])
cb.set_ticklabels(["1", "10", "100", "1 000", "10 000"])
cb.set_label(r"Resistivitat  $\rho$  ($\Omega\cdot$m)", fontsize=10)
cb.ax.tick_params(labelsize=8)

fig.suptitle("Comparació dels perfils perpendiculars amb l'estudi de referència",
             fontsize=13.5, fontweight="bold", x=0.47, y=0.965)

for path, dpi in ((OUT_PNG, 400), (OUT_PDF, None)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print("Saved:", path)
