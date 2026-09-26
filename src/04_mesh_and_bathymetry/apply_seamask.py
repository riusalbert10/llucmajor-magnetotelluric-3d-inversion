"""
Apply land/sea mask to a homogeneous ModEM .rho model using a KML coastline polygon.

Reads:
  - Mallorca_Homo2  (ModEM .rho, LOGE format, 76x76x30, homogeneous 100 ohm.m)
  - Perfil_costa_homo2.kml  (closed polygon delimiting LAND area)

Produces:
  - Mallorca_Homo2_seamask.rho  (same grid, sea cells set to 0.3 ohm.m down to SEA_DEPTH_M)
  - mask_check.png              (visual check: cell centers colored by land/sea + polygon + sites)
"""
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from shapely.geometry import Polygon, Point

# ---------------------------------------------------------------- USER PARAMETERS
RHO_IN     = r"C:\Users\alber\TFG\Generar_Costa_Bat_new\Mallorca_Homo2"
KML_POLY   = r"C:\Users\alber\TFG\Generar_Costa_Bat_new\Perfil_costa_Llucmajor.kml"
KML_SITES  = r"C:\Users\alber\TFG\Generar_Costa_Bat_new\Coordenades_sites.KML"

RHO_OUT    = r"C:\Users\alber\TFG\Generar_Costa_Bat_new\Mallorca_Homo2_seamask.rho"
PNG_OUT    = r"C:\Users\alber\TFG\Generar_Costa_Bat_new\mask_check.png"

# Geographic origin (matches .dat file). Local (x=0, y=0) corresponds to this lat/lon.
LAT_ORIGIN = 39.476669
LON_ORIGIN = 2.895278

# Resistivity values (ohm.m, NOT log)
RHO_LAND = 500.0
RHO_SEA  = 0.3

# How deep should the sea extend in the model? Cells with depth_top < SEA_DEPTH_M
# inside sea columns are set to RHO_SEA. Set to None to fill the entire column with sea.
SEA_DEPTH_M = 100.0   # metres
# ---------------------------------------------------------------------------------


def read_rho(path):
    """Read a ModEM .rho file (LOGE format). Returns header info and 3D array."""
    with open(path) as f:
        text = f.read()

    # Tokenize, preserving structure
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # First non-empty line should be the header (sometimes there's a comment line first)
    # Skip leading comment lines that start with '#'
    i = 0
    while lines[i].lstrip().startswith('#'):
        i += 1
    header = lines[i].split()
    mnx, mny, mnz = int(header[0]), int(header[1]), int(header[2])
    rotation_in_header = int(header[3]) if len(header) > 3 else 0
    fmt = header[4] if len(header) > 4 else 'LOGE'
    i += 1

    # Read dx (mnx values), dy (mny values), dz (mnz values) - may span multiple lines
    def read_n(lines, start, n):
        vals = []
        idx = start
        while len(vals) < n:
            vals.extend(float(x) for x in lines[idx].split())
            idx += 1
        return np.array(vals[:n]), idx

    dx, i = read_n(lines, i, mnx)
    dy, i = read_n(lines, i, mny)
    dz, i = read_n(lines, i, mnz)

    # Read the resistivity values: mnz blocks of mnx*mny values
    nvals = mnx * mny * mnz
    flat = []
    while len(flat) < nvals and i < len(lines):
        toks = lines[i].split()
        # Stop if we hit the origin line (3 numbers, all parseable as floats but not enough)
        if len(toks) <= 4 and len(flat) >= nvals - 4:
            break
        flat.extend(float(x) for x in toks)
        i += 1
    flat = np.array(flat[:nvals])

    # Grid_3D convention: outermost = iz (top->bottom), then iy (0 to mny-1, normal),
    # then ix (mnx-1 down to 0, INVERTED). So in each block, rows iterate iy and
    # columns iterate ix inverted.
    # Reshape: vals_file[k, r, c] where r = iy (0..mny-1), c maps to ix = mnx-1-c
    vals_file = flat.reshape((mnz, mny, mnx))

    # Convert to a "natural" array indexed by [k, ix, iy] where ix=0 is south, iy=0 is west
    # natural[k, ix, iy] = vals_file[k, iy, mnx-1-ix]
    vals = vals_file.swapaxes(1, 2)[:, ::-1, :].copy()

    # Read remaining lines for origin and rotation
    rest = []
    while i < len(lines):
        rest.append(lines[i].strip())
        i += 1
    # First line of rest = origin (x0, y0, z0); next line = rotation
    origin = [float(x) for x in rest[0].split()]
    rotation = float(rest[1].split()[0]) if len(rest) > 1 else 0.0

    return {
        'mnx': mnx, 'mny': mny, 'mnz': mnz,
        'dx': dx, 'dy': dy, 'dz': dz,
        'vals': vals,            # natural indexing [iz, ix, iy]
        'origin': origin,        # [x0, y0, z0] in metres (SW-bottom corner)
        'rotation': rotation,
        'fmt': fmt,
        'rotation_in_header': rotation_in_header,
    }


def write_rho(path, m, vals_natural):
    """Write .rho in Grid_3D format (LOGE).
    Convention: rows iterate iy normal (0..mny-1), columns iterate ix inverted (mnx-1..0).
    """
    mnx, mny, mnz = m['mnx'], m['mny'], m['mnz']
    # Convert natural [iz, ix, iy] back to file layout [iz, r=iy, c=mnx-1-ix]
    # Steps: flip ix axis, then swap axes 1 and 2 -> shape (mnz, mny, mnx)
    vals_file = vals_natural[:, ::-1, :].swapaxes(1, 2)

    with open(path, 'w') as f:
        f.write('# 3D MT model - sea mask applied via KML polygon\n')
        f.write(f' {mnx}  {mny}  {mnz}  {m["rotation_in_header"]} {m["fmt"]}\n')
        # dx, dy, dz
        f.write(' '.join(f'{v:.3f}' for v in m['dx']) + ' \n')
        f.write(' '.join(f'{v:.3f}' for v in m['dy']) + ' \n')
        f.write(' '.join(f'{v:.3f}' for v in m['dz']) + ' \n')
        f.write('\n')
        # values: mnz blocks, each block has mny rows of mnx values
        for k in range(mnz):
            for r in range(mny):
                row = vals_file[k, r, :]
                f.write(' '.join(f'{v:.5E}' for v in row) + ' \n')
            f.write('\n')
        # origin and rotation
        f.write(f'{m["origin"][0]:.3f}    {m["origin"][1]:.3f}    {m["origin"][2]:.3f}\n')
        f.write(f'{m["rotation"]:.1f}\n')


def parse_kml_polygon(path):
    """Parse the first <Polygon> outerBoundary coordinates from a KML file.
    Returns a list of (lon, lat) tuples."""
    with open(path) as f:
        text = f.read()
    m = re.search(r'<outerBoundaryIs>.*?<coordinates>(.*?)</coordinates>',
                  text, re.DOTALL)
    if not m:
        raise ValueError("No outerBoundary polygon found in KML")
    coords_text = m.group(1).strip()
    pts = []
    for tok in coords_text.split():
        parts = tok.split(',')
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            pts.append((lon, lat))
    return pts


def parse_kml_sites(path):
    """Parse all <Placemark><Point> from a KML, returns list of (name, lon, lat)."""
    with open(path) as f:
        text = f.read()
    sites = []
    for m in re.finditer(
        r'<Placemark>.*?<name>(.*?)</name>.*?<Point>.*?<coordinates>(.*?)</coordinates>',
        text, re.DOTALL):
        name = m.group(1).strip()
        coord = m.group(2).strip().split(',')
        lon, lat = float(coord[0]), float(coord[1])
        sites.append((name, lon, lat))
    return sites


def cell_centers_latlon(m):
    """Return arrays (lat, lon) of shape (mnx, mny) for the cell centres."""
    mnx, mny = m['mnx'], m['mny']
    # x is North (positive North), y is East (positive East). Origin is SW corner.
    x0, y0 = m['origin'][0], m['origin'][1]
    # Cell edge positions
    x_edges = np.concatenate([[x0], x0 + np.cumsum(m['dx'])])
    y_edges = np.concatenate([[y0], y0 + np.cumsum(m['dy'])])
    # Centres
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])     # length mnx
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])     # length mny

    # Convert local metres to lat/lon (flat-earth around origin)
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(LAT_ORIGIN)))

    lat = LAT_ORIGIN + xc * deg_per_m_lat       # length mnx
    lon = LON_ORIGIN + yc * deg_per_m_lon       # length mny
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')   # shape (mnx, mny)
    return LAT, LON, xc, yc, x_edges, y_edges


def main():
    print("Reading model...")
    m = read_rho(RHO_IN)
    print(f"  Grid: {m['mnx']}x{m['mny']}x{m['mnz']}, format {m['fmt']}")
    print(f"  Origin (m): {m['origin']}, rotation {m['rotation']}")
    print(f"  Total extent: dx={m['dx'].sum():.0f} m, dy={m['dy'].sum():.0f} m, "
          f"dz={m['dz'].sum():.0f} m")
    rho_unique = np.unique(np.round(np.exp(m['vals']), 3))
    print(f"  Unique resistivity values (ohm.m): {rho_unique}")

    print("Reading polygon and sites...")
    poly_pts = parse_kml_polygon(KML_POLY)
    print(f"  Polygon: {len(poly_pts)} vertices")
    sites = parse_kml_sites(KML_SITES)
    print(f"  Sites:   {len(sites)}")

    poly = Polygon(poly_pts)
    # Note: do NOT use poly.buffer(0) to fix invalid polygons -- it inverts orientation

    print("Computing cell centres in lat/lon...")
    LAT, LON, xc, yc, x_edges, y_edges = cell_centers_latlon(m)

    print("Classifying cells (land vs sea)...")
    is_land = np.zeros((m['mnx'], m['mny']), dtype=bool)
    for ix in range(m['mnx']):
        for iy in range(m['mny']):
            is_land[ix, iy] = poly.contains(Point(LON[ix, iy], LAT[ix, iy]))
    n_land = int(is_land.sum())
    n_sea  = is_land.size - n_land
    print(f"  Land cells: {n_land}  ({100*n_land/is_land.size:.1f}%)")
    print(f"  Sea cells:  {n_sea}  ({100*n_sea/is_land.size:.1f}%)")

    # Apply mask: sea cells get RHO_SEA in top layers (depth_top < SEA_DEPTH_M)
    print(f"Applying sea mask: top layers down to {SEA_DEPTH_M} m get {RHO_SEA} ohm.m")
    new_vals = m['vals'].copy()
    depth_top = np.concatenate([[0.0], np.cumsum(m['dz'])])[:-1]   # top of each layer
    sea_layers = np.where(depth_top < SEA_DEPTH_M)[0] if SEA_DEPTH_M else np.arange(m['mnz'])
    print(f"  Layers replaced with sea water: {sea_layers.tolist()}")
    print(f"  Their depth ranges: {[(depth_top[k], depth_top[k]+m['dz'][k]) for k in sea_layers]}")
    for k in sea_layers:
        for ix in range(m['mnx']):
            for iy in range(m['mny']):
                if not is_land[ix, iy]:
                    new_vals[k, ix, iy] = math.log(RHO_SEA)

    # ---- Visualization
    print("Drawing check plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ---- Left: lat/lon view with polygon
    ax = axes[0]
    # cell-centre dots, coloured by land/sea
    ax.scatter(LON[is_land], LAT[is_land], s=4, c='#7BA86F', label='Land cell')
    ax.scatter(LON[~is_land], LAT[~is_land], s=4, c='#5B8AB8', label='Sea cell')
    # polygon outline
    px, py = zip(*poly_pts)
    ax.plot(px, py, '-', color='#C0392B', lw=1.5, label='Coastline polygon')
    # sites
    for name, lon, lat in sites:
        ax.plot(lon, lat, 'k^', ms=5)
    ax.plot([], [], 'k^', ms=5, label='MT sites')
    # geographic origin
    ax.plot(LON_ORIGIN, LAT_ORIGIN, 'rx', ms=10, mew=2, label='Origin')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Land/Sea classification (geographic view)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect(1.0 / math.cos(math.radians(LAT_ORIGIN)))
    ax.grid(alpha=0.3)

    # ---- Right: zoom on station cluster, local metres
    ax = axes[1]
    # Plot cells as pcolormesh: integer 0=sea, 1=land  (use natural indexing)
    # In local metres, ix is along x (N), iy is along y (E)
    # x_edges, y_edges already correspond to ix and iy edges
    # is_land[ix, iy]: rows are ix (south->north), cols are iy (west->east)
    # For pcolormesh we want X = y (east), Y = x (north) so transpose:
    # extent: y on horizontal, x on vertical
    Y_edges, X_edges = np.meshgrid(y_edges, x_edges)  # shapes (mnx+1, mny+1)
    ax.pcolormesh(Y_edges/1000, X_edges/1000, is_land.astype(float),
                  cmap='RdYlGn', vmin=0, vmax=1, alpha=0.6, shading='flat')

    # Sites in local metres
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(LAT_ORIGIN)))
    for name, lon, lat in sites:
        sx = (lat - LAT_ORIGIN) / deg_per_m_lat   # north
        sy = (lon - LON_ORIGIN) / deg_per_m_lon   # east
        ax.plot(sy/1000, sx/1000, 'k^', ms=5)
    # polygon in local metres
    px_loc = [(lon - LON_ORIGIN) / deg_per_m_lon / 1000 for lon, lat in poly_pts]
    py_loc = [(lat - LAT_ORIGIN) / deg_per_m_lat / 1000 for lon, lat in poly_pts]
    ax.plot(px_loc, py_loc, '-', color='#C0392B', lw=1.5)
    # origin
    ax.plot(0, 0, 'rx', ms=10, mew=2)

    ax.set_xlabel('y - East (km)')
    ax.set_ylabel('x - North (km)')
    ax.set_title('Land/Sea map in local metres (green=land, red=sea)')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    plt.suptitle(f'Sea-mask check — {RHO_IN.split("/")[-1]}\n'
                 f'{n_land} land cells, {n_sea} sea cells, sea depth = {SEA_DEPTH_M} m',
                 y=1.02)
    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=120, bbox_inches='tight')
    print(f"  Saved {PNG_OUT}")

    # ---- Write modified .rho
    print(f"Writing {RHO_OUT}...")
    write_rho(RHO_OUT, m, new_vals)
    print("Done.")


if __name__ == '__main__':
    main()
