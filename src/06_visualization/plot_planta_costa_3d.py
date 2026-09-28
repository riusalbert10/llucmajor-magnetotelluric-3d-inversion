"""
Plot 3D de la PLANTA HORITZONTAL del model inicial de resistivitats, amb el
perfil de costa (línia) i les estacions MT sobreposats.

  - Planta: capa superior del model (log10 rho) com un pla horitzontal en 3D,
    acolorida amb jet invertit (mar ~0.3 ohm.m en vermell, semiespai 100 ohm.m).
  - Costa : línia del fitxer KML del perfil de costa.
  - Estacions: punts del KML de coordenades dels sites.

Entrades:
  MODEL : fitxer .rho del model inicial (format ModEM/WS LOGE)
  COAST : KML amb el LineString del perfil de costa (lon,lat,alt)
  SITES : KML amb Placemarks de les estacions (Point lon,lat,alt)

Sortida:
    C:\\Users\\alber\\TFG\\visualizations\\model_inicial_planta_3d.png
"""
import re
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from pathlib import Path

# ============================ CONFIG ============================
# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))

MODEL = str(TFG_DIR / "MT_Llucmajor_results/Model_bat_NO_otliers/Arxius_ejecució/Model_Bat_recond")
COAST = str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat/Generar_Costa_Batimetria/Perfil_costa_Llucmajor.kml")
SITES = str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat/Generar_Costa_Batimetria/Coordenades_sites.KML")
OUT   = str(TFG_DIR / "visualizations/model_inicial_planta_3d.png")

LAT0, LON0 = 39.476669, 2.895278     # centre geogràfic del model (origen de la malla)
FOV = 15.0                           # semi-camp de visió (km)
ROT = 1                              # alineació 90 CW validada de la malla
VMIN, VMAX = 1.0, 1e4                # rang de la barra de color (ohm.m)
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
    layer, i = rf(nx * ny, i)                     # NOMÉS la capa superior (z=0)
    top = layer.reshape(nx, ny)
    if mode.upper() == 'LOGE':
        top = np.exp(top)
    return dx, dy, top


dx, dy, top = parse_rho_top(MODEL)
xn = np.concatenate(([0], np.cumsum(dx))); xn -= xn.mean(); xn /= 1000.0   # nord (km)
yn = np.concatenate(([0], np.cumsum(dy))); yn -= yn.mean(); yn /= 1000.0   # est (km)
nc = (xn[:-1] + xn[1:]) / 2; ec = (yn[:-1] + yn[1:]) / 2
top = np.log10(np.clip(top, VMIN, VMAX))          # (nord, est)
if ROT:
    n = min(top.shape)
    top = np.rot90(top[:n, :n], ROT); nc = nc[:n]; ec = ec[:n]
ni = np.where(np.abs(nc) <= FOV)[0]; ei = np.where(np.abs(ec) <= FOV)[0]
top = top[ni[0]:ni[-1] + 1, ei[0]:ei[-1] + 1]; nc = nc[ni]; ec = ec[ei]


def edges(c):
    c = np.asarray(c, float); e = np.empty(c.size + 1)
    e[1:-1] = (c[:-1] + c[1:]) / 2
    e[0] = c[0] - (c[1] - c[0]) / 2; e[-1] = c[-1] + (c[-1] - c[-2]) / 2
    return e
xe, ye = edges(ec), edges(nc)                     # est nodes, nord nodes


# ---- KML -> coordenades locals (km) ----
KX = 111.0 * np.cos(np.radians(LAT0))
ll2en = lambda lo, la: ((lo - LON0) * KX, (la - LAT0) * 111.0)

def coords_block(txt):
    out = []
    for tok in txt.replace("\n", " ").split():
        pp = tok.split(",")
        if len(pp) >= 2:
            try:
                out.append(ll2en(float(pp[0]), float(pp[1])))
            except ValueError:
                pass
    return np.array(out)

coast = coords_block(re.search(r"<coordinates>(.*?)</coordinates>",
                               open(COAST).read(), re.S).group(1))
sites = np.array([ll2en(*[float(v) for v in m.group(2).split(",")[:2]])
                  for m in re.finditer(
                      r"<name>(mall\d+)</name>.*?<coordinates>([^<]+)</coordinates>",
                      open(SITES).read(), re.S)])

# ---- figura 3D ----
cmap = plt.get_cmap("jet_r"); norm = Normalize(np.log10(VMIN), np.log10(VMAX))
X, Y = np.meshgrid(xe, ye)                          # est, nord (nodes)
Z = np.zeros_like(X)

fig = plt.figure(figsize=(12.5, 9.5))
ax = fig.add_subplot(111, projection='3d')
ax.set_proj_type('ortho')
try:
    ax.computed_zorder = False
except Exception:
    pass

ax.plot_surface(X, Y, Z, facecolors=cmap(norm(top)), rstride=1, cstride=1,
                shade=False, antialiased=False, linewidth=0, zorder=1)
m = (np.abs(coast[:, 0]) <= FOV + 1) & (np.abs(coast[:, 1]) <= FOV + 1)
ax.plot(coast[m, 0], coast[m, 1], 0.2, color="k", lw=2.8,
        label="Perfil de costa", zorder=5)
ax.scatter(sites[:, 0], sites[:, 1], 0.3, marker="v", s=42, c="white",
           edgecolor="k", linewidth=0.9, depthshade=False,
           label="Estacions MT", zorder=6)

ax.set_xlim(-FOV, FOV); ax.set_ylim(-FOV, FOV); ax.set_zlim(-1, 1.5)
ax.set_xlabel("Est (km)", labelpad=10); ax.set_ylabel("Nord (km)", labelpad=10)
ax.set_zticks([]); ax.set_box_aspect((1, 1, 0.35))
ax.view_init(elev=42, azim=-60)
ax.set_title("Model Inicial de resistivitats — planta horitzontal",
             fontsize=15, fontweight="bold", pad=20)

sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
cb.set_ticks([0, 1, 2, 3, 4]); cb.set_ticklabels(["1", "10", "100", "1 000", "10 000"])
cb.set_label(r"Resistivitat  $\rho$  ($\Omega\cdot$m)")
ax.legend(loc="upper left", fontsize=10)

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("Saved:", OUT)
