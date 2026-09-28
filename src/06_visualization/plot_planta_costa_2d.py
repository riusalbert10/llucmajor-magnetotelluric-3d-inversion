"""
Planta horitzontal (2D, vista zenital) del model inicial: mapa TERRA / MAR sobre
tota la superfície de la malla (~45 x 45 km), amb el perfil de costa, les
estacions MT i la ciutat de Llucmajor com a referència.

  - Terra (semiespai 100 ohm.m) en marró, Mar (~0.3 ohm.m) en blau,
    a partir de la capa superior del model (llindar log10 rho < 1 = mar).
  - Costa : línia del KML del perfil de costa (fins als marges del plot).
  - Estacions: punts del KML de sites.  Llucmajor: estrella de referència.

Sortida:
    C:\\Users\\alber\\TFG\\visualizations\\model_inicial_planta_2d.png
"""
import re
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
from pathlib import Path

# ============================ CONFIG ============================
# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))

MODEL = str(TFG_DIR / "MT_Llucmajor_results/Model_bat_NO_otliers/Arxius_ejecució/Model_Bat_recond")
COAST = str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat/Generar_Costa_Batimetria/Perfil_costa_Llucmajor.kml")
SITES = str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat/Generar_Costa_Batimetria/Coordenades_sites.KML")
OUT   = str(TFG_DIR / "visualizations/model_inicial_planta_2d.png")

LAT0, LON0 = 39.476669, 2.895278        # centre geogràfic (origen de la malla)
LLUC_LAT, LLUC_LON = 39.491667, 2.891667  # ciutat de Llucmajor
ROT = 1                                  # alineació 90 CW validada
SEA_LOG10_THRESHOLD = 1.0                # log10(rho) < 1 -> mar
LAND, SEA = "#9c6b3f", "#3d7bd6"         # colors terra / mar
# ===============================================================


def parse_rho_top(path):
    lines = open(path).read().splitlines()
    i = 0
    while lines[i].strip().startswith('#') or lines[i].strip() == '':
        i += 1
    p = lines[i].split(); nx, ny, nz = int(p[0]), int(p[1]), int(p[2]); mode = p[4]; i += 1

    def rf(n, i):
        v = []
        while len(v) < n:
            t = lines[i].strip()
            if t:
                v += [float(x) for x in t.split()]
            i += 1
        return np.array(v[:n]), i

    dx, i = rf(nx, i); dy, i = rf(ny, i); dz, i = rf(nz, i)
    layer, i = rf(nx * ny, i)
    top = layer.reshape(nx, ny)
    if mode.upper() == 'LOGE':
        top = np.exp(top)
    return dx, dy, top


dx, dy, top = parse_rho_top(MODEL)
xn = np.concatenate(([0], np.cumsum(dx))); xn -= xn.mean(); xn /= 1000.0   # nord nodes
yn = np.concatenate(([0], np.cumsum(dy))); yn -= yn.mean(); yn /= 1000.0   # est nodes
sea = (np.log10(top) < SEA_LOG10_THRESHOLD).astype(int)                    # 1=mar,0=terra
if ROT:
    n = min(sea.shape)
    sea = np.rot90(sea[:n, :n], ROT); xn = xn[:n + 1]; yn = yn[:n + 1]
# nodes en km -> lon/lat (est -> lon, nord -> lat)
KX = 111.0 * np.cos(np.radians(LAT0))
lon_edges = LON0 + yn / KX
lat_edges = LAT0 + xn / 111.0


# ---- KML: coordenades lon/lat directes ----
def coords_block(txt):
    out = []
    for tok in txt.replace("\n", " ").split():
        pp = tok.split(",")
        if len(pp) >= 2:
            try:
                out.append((float(pp[0]), float(pp[1])))   # (lon, lat)
            except ValueError:
                pass
    return np.array(out)

coast = coords_block(re.search(r"<coordinates>(.*?)</coordinates>",
                               open(COAST).read(), re.S).group(1))
sites = np.array([[float(v) for v in m.group(2).split(",")[:2]]
                  for m in re.finditer(
                      r"<name>(mall\d+)</name>.*?<coordinates>([^<]+)</coordinates>",
                      open(SITES).read(), re.S)])

# ---- figura 2D (lon/lat) ----
from matplotlib.ticker import FuncFormatter
cmap = ListedColormap([LAND, SEA])
fig, ax = plt.subplots(figsize=(10.5, 9.2))
ax.pcolormesh(lon_edges, lat_edges, sea, cmap=cmap, vmin=0, vmax=1, shading="flat")
ax.plot(coast[:, 0], coast[:, 1], color="k", lw=2.6)
ax.scatter(sites[:, 0], sites[:, 1], marker="v", s=44, c="white", edgecolor="k",
           linewidth=0.9, zorder=5)
ax.scatter([LLUC_LON], [LLUC_LAT], marker="*", s=360, c="gold", edgecolor="k",
           linewidth=1.1, zorder=7)
ax.annotate("Llucmajor", (LLUC_LON, LLUC_LAT),
            xytext=(LLUC_LON + 0.012, LLUC_LAT + 0.012),
            fontsize=12, fontweight="bold", color="k", zorder=8)

ax.set_xlim(lon_edges[0], lon_edges[-1]); ax.set_ylim(lat_edges[0], lat_edges[-1])
ax.set_aspect(1.0 / np.cos(np.radians(LAT0)))
_fmt = FuncFormatter(lambda v, _: f"{v:.2f}°")
ax.xaxis.set_major_formatter(_fmt); ax.yaxis.set_major_formatter(_fmt)
ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
ax.set_title("Model inicial i localització de les estacions",
             fontsize=14, fontweight="bold")

handles = [mpatches.Patch(color=LAND, label="Terra"),
           mpatches.Patch(color=SEA, label="Mar"),
           Line2D([0], [0], color="k", lw=2.6, label="Perfil de costa"),
           Line2D([0], [0], marker="v", color="none", markerfacecolor="white",
                  markeredgecolor="k", markersize=10, label="Estacions MT"),
           Line2D([0], [0], marker="*", color="none", markerfacecolor="gold",
                  markeredgecolor="k", markersize=16, label="Llucmajor")]
ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.92)

fig.tight_layout()
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print("Saved:", OUT)
