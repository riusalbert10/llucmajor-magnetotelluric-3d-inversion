"""
Figura 3D del model inicial de resistivitats (malla ModEM/WS, format LOGE).

Renderitza un bloc 3D amb un tall (cutaway) que revela l'interior, de manera que
s'aprecien les dimensions de la malla i les zones de resistivitat. En lloc d'una
rampa de color continua, cada cel·la es classifica per la seva litologia mes
propera (en log10) i es pinta amb un color solid i propi, molt mes clar de
llegir i que permet resaltar la zona d'estructura:
  - mar                ~0.3  ohm.m   -> blau
  - estructura (roca sedimentaria) ~30 ohm.m -> vermell/taronja (resaltat)
  - semiespai (terra)  ~100  ohm.m   -> ocre/marro

Requereix: pyvista  (pip install pyvista).  En Linux headless cal un display
virtual: executa amb  `xvfb-run -a python plot_model_inicial_3d.py`.  En Windows
amb pantalla no cal xvfb.

Sortida:
    C:\\Users\\alber\\TFG\\visualizations\\model_inicial_3d.png
"""
import numpy as np
import pyvista as pv

# ============================ CONFIG ============================
MODEL = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_bat_NO_otliers\Arxius_ejecució\Model_Bat_recond"
OUT   = r"C:\Users\alber\TFG\visualizations\model_3D.png"

# Litologies de referencia (rho en ohm.m) i el color solid que se'ls assigna.
# L'estructura es la mes rellevant (roca sedimentaria ~500 m per sota de
# l'ultima capa marina) i per aixo porta el color mes viu/contrastat.
LITOLOGIES = [
    dict(rho=0.3,  nom="Agua marina",       color=(0.10, 0.35, 0.85)),  # blau
    dict(rho=30.0, nom="Roca sedimentaria", color=(0.92, 0.05, 0.05)),  # vermell viu
    dict(rho=100.0, nom="Tierra",           color=(0.80, 0.62, 0.30)),  # ocre
]
# ===============================================================


def parse_rho(path):
    lines = open(path).read().splitlines()
    i = 0
    while lines[i].strip().startswith('#') or lines[i].strip() == '':
        i += 1
    p = lines[i].split()
    nx, ny, nz = int(p[0]), int(p[1]), int(p[2])
    mode = p[4] if len(p) >= 5 else 'LINEAR'
    i += 1

    def rf(n, i):
        v = []
        while len(v) < n:
            t = lines[i].strip()
            if t:
                v += [float(x) for x in t.split()]
            i += 1
        return np.array(v[:n]), i

    dx, i = rf(nx, i); dy, i = rf(ny, i); dz, i = rf(nz, i)
    res = np.empty((nx, ny, nz))
    for k in range(nz):
        lay, i = rf(nx * ny, i)
        res[:, :, k] = lay.reshape(nx, ny)
    if mode.upper() == 'LOGE':
        res = np.exp(res)
    return nx, ny, nz, dx, dy, dz, res


nx, ny, nz, dx, dy, dz, res = parse_rho(MODEL)
logr = np.log10(res)                                   # (north, east, depth)
print(f"Malla {nx}x{ny}x{nz} | extensio (km): "
      f"{dx.sum()/1e3:.1f} x {dy.sum()/1e3:.1f} x {dz.sum()/1e3:.1f}")

# Coordenades dels nodes (km): X,Y centrats; Z = fondaria (avall = negatiu)
xn = np.concatenate(([0], np.cumsum(dx))) / 1000.0; xn -= xn.mean()   # nord
yn = np.concatenate(([0], np.cumsum(dy))) / 1000.0; yn -= yn.mean()   # est
zn = -np.concatenate(([0], np.cumsum(dz))) / 1000.0                   # fondaria

grid = pv.RectilinearGrid(yn, xn, zn)                  # x=est, y=nord, z=fondaria
flat_logr = np.transpose(logr, (1, 0, 2)).flatten(order="F")

# Classificacio per litologia: cada cel·la agafa el color de la litologia de
# referencia mes propera en log10(rho). Aixo dona colors SOLIDS (no una
# rampa continua) i deixa l'estructura ben resaltada amb el seu color viu.
MAR_IDX, STRUCT_IDX, TERRA_IDX = 0, 1, 2   # index dins LITOLOGIES
targets = np.log10([lit["rho"] for lit in LITOLOGIES])
cat = np.argmin(np.abs(flat_logr[:, None] - targets[None, :]), axis=1)
colors_lut = np.array([lit["color"] for lit in LITOLOGIES])
grid.cell_data["rgb"] = (colors_lut[cat] * 255).astype(np.uint8)

pl = pv.Plotter(off_screen=True, window_size=[2200, 1500])
pl.set_background("white")
# bloc TRANSLUCID complet (mar+estructura+terra), amb la malla integrada,
# per apreciar les dimensions reals i deixar-hi veure a traves
pl.add_mesh(grid, scalars="rgb", rgb=True, opacity=0.75,
            show_edges=True, edge_color="black", line_width=1.2,
            show_scalar_bar=False)

# el mar es prim i gairebe superficial: es repinta opac (sense exagerar) per
# no perdre'l dins la translucidesa, encara que quedi tapat on l'estructura
# (mes extensa) hi ha just a sota
mar_grid = grid.extract_cells(np.where(cat == MAR_IDX)[0])
pl.add_mesh(mar_grid, scalars="rgb", rgb=True, opacity=1.0, lighting=False,
            show_edges=True, edge_color="black", line_width=1.0,
            show_scalar_bar=False)

# --- Resaltat de l'estructura ---
# Es dibuixa una COPIA de l'estructura, opaca i sense ombrejat (color pla,
# vermell viu en lloc del magenta apagat que dona la il·luminacio per
# defecte), amb un contorn groc a la silueta exterior. STRUCT_EXAGGER es un
# petit exagerament del seu gruix vertical (al voltant del propi centre de
# fondaria) NOMES a efectes visuals; es manté baix (i retallat al bloc) per
# no envair l'espai del mar, que ha de continuar apreciant-se per sobre.
STRUCT_EXAGGER = 1.0
struct_grid = grid.extract_cells(np.where(cat == STRUCT_IDX)[0]).copy()
z = struct_grid.points[:, 2]
z_mid = 0.5 * (z.min() + z.max())
struct_grid.points[:, 2] = np.clip(z_mid + (z - z_mid) * STRUCT_EXAGGER,
                                    zn.min(), zn.max())
struct_outline = struct_grid.extract_surface(algorithm="dataset_surface").extract_feature_edges(
    boundary_edges=True, non_manifold_edges=False, manifold_edges=False)

pl.add_mesh(struct_grid, scalars="rgb", rgb=True, opacity=1.0, lighting=False,
            show_edges=True, edge_color="black", line_width=1.0,
            show_scalar_bar=False)
pl.add_mesh(struct_outline, color="yellow", line_width=4, lighting=False)
# llegenda discreta amb els 3 colors/litologies, en lloc de la barra continua.
# Ancorada a la cantonada inferior esquerra perque el text creixi cap a dins
# del canvas i no es talli pel marge dret de la imatge.
def _rho_txt(lit):
    txt = f'{lit["rho"]:g} ohm.m'
    if lit is LITOLOGIES[STRUCT_IDX] and STRUCT_EXAGGER != 1.0:
        txt += f" (x{STRUCT_EXAGGER:g})"
    return txt

# Estil "sec" i professional: valor de resistivitat en el color de la
# litologia (fa de "quadradet" sense dependre de glifs Unicode que VTK no
# sap renderitzar) + nom SEMPRE en negre. pv.add_legend() en aquesta build
# de VTK no respecta ni el color de text ni la vora del requadre, aixi que
# la llegenda es construeix a ma amb add_text (que si que renderitza en
# negre de manera fiable), amb un fons blanc darrere cada linia.
LEGEND_X, LEGEND_Y0, LEGEND_DY = 0.03, 0.15, 0.05
for i, lit in enumerate(LITOLOGIES):
    y = LEGEND_Y0 + (len(LITOLOGIES) - 1 - i) * LEGEND_DY
    rho = pl.add_text(_rho_txt(lit), position=(LEGEND_X, y), color=lit["color"],
                       font_size=18, viewport=True)
    rho.GetTextProperty().SetBackgroundColor(1, 1, 1)
    rho.GetTextProperty().SetBackgroundOpacity(1.0)
    lbl = pl.add_text(lit["nom"], position=(LEGEND_X + 0.105, y),
                       color="black", font_size=18, viewport=True)
    lbl.GetTextProperty().SetBackgroundColor(1, 1, 1)
    lbl.GetTextProperty().SetBackgroundOpacity(1.0)
pl.show_bounds(location="outer", ticks="outside",
               n_xlabels=5, n_ylabels=5, n_zlabels=5,
               xtitle="Est (km)", ytitle="Nord (km)", ztitle="Fondaria (km)",
               font_size=18, color="black")
pl.add_text("Model Inicial de resistivitats", position="upper_edge",
            font_size=26, color="black")
pl.camera_position = [(78, -78, 52), (0, 0, -8), (0, 0, 1)]
pl.camera.zoom(0.95)
pl.screenshot(OUT)
print("Saved:", OUT)
