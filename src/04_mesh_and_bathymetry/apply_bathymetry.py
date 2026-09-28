"""
apply_bathymetry.py
-------------------
Reads EMODnet bathymetry CSV and applies it to a ModEM .rho model that already
has a coastline-aware seamask (e.g. Mallorca_Homo2_seamask.rho).

For each marine column (cells that the current model already classifies as sea):
  - Look up the actual seafloor depth from EMODnet at the cell-center lat/lon.
  - Layers ABOVE the seafloor → water (RHO_SEA, 0.3 Ω·m)
  - Layers BELOW the seafloor → marine sediments (RHO_SEDIMENT, 30 Ω·m)

Land columns (currently 100 Ω·m everywhere) are left untouched.

The bathymetry CSV may cover a larger area than the model — that's fine, only
the cells that fall inside the model grid are queried.
"""
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import os

# ---------------------------------------------------------------- CONFIG
# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))

RHO_IN     = Path(str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat_opt/Mallorca_Homo2_seamask.rho"))
BATHY_CSV  = Path(str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat_opt/Mean_depth.csv"))
RHO_OUT    = Path(str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat_opt/Mallorca_Homo2_seamask_bathy.rho"))

# Geographic origin (must match what was used for the .rho and the .dat)
LAT_ORIGIN = 39.476669
LON_ORIGIN = 2.895278

# Resistivities (Ω·m, NOT log)
RHO_LAND     = 100.0   # already in input — kept untouched
RHO_SEA      = 0.3     # water column above seafloor
RHO_SEDIMENT = 30.0    # NEW — replaces "land below water" with marine sediments

# Tolerance for identifying current sea cells in LOGE values
SEA_TOL = 0.01
# -----------------------------------------------------------------------


# ============ .rho I/O (same convention as apply_seamask.py) ============
def read_rho(path: Path) -> dict:
    with open(path) as f:
        text = f.read()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    i = 0
    while lines[i].lstrip().startswith('#'):
        i += 1
    header = lines[i].split()
    mnx, mny, mnz = int(header[0]), int(header[1]), int(header[2])
    rotation_in_header = int(header[3]) if len(header) > 3 else 0
    fmt = header[4] if len(header) > 4 else 'LOGE'
    i += 1

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

    nvals = mnx * mny * mnz
    flat = []
    while len(flat) < nvals and i < len(lines):
        toks = lines[i].split()
        if len(toks) <= 4 and len(flat) >= nvals - 4:
            break
        flat.extend(float(x) for x in toks)
        i += 1
    flat = np.array(flat[:nvals])
    vals_file = flat.reshape((mnz, mny, mnx))
    vals = vals_file.swapaxes(1, 2)[:, ::-1, :].copy()

    rest = []
    while i < len(lines):
        rest.append(lines[i].strip())
        i += 1
    origin = [float(x) for x in rest[0].split()]
    rotation = float(rest[1].split()[0]) if len(rest) > 1 else 0.0

    return {
        'mnx': mnx, 'mny': mny, 'mnz': mnz,
        'dx': dx, 'dy': dy, 'dz': dz,
        'vals': vals, 'origin': origin, 'rotation': rotation,
        'fmt': fmt, 'rotation_in_header': rotation_in_header,
    }


def write_rho(path: Path, m: dict, vals_natural: np.ndarray):
    mnx, mny, mnz = m['mnx'], m['mny'], m['mnz']
    vals_file = vals_natural[:, ::-1, :].swapaxes(1, 2)
    with open(path, 'w') as f:
        f.write('# 3D MT model - bathymetry + 30 ohm.m sediment layer\n')
        f.write(f' {mnx}  {mny}  {mnz}  {m["rotation_in_header"]} {m["fmt"]}\n')
        f.write(' '.join(f'{v:.3f}' for v in m['dx']) + ' \n')
        f.write(' '.join(f'{v:.3f}' for v in m['dy']) + ' \n')
        f.write(' '.join(f'{v:.3f}' for v in m['dz']) + ' \n')
        f.write('\n')
        for k in range(mnz):
            for r in range(mny):
                row = vals_file[k, r, :]
                f.write(' '.join(f'{v:.5E}' for v in row) + ' \n')
            f.write('\n')
        f.write(f'{m["origin"][0]:.3f}    {m["origin"][1]:.3f}    {m["origin"][2]:.3f}\n')
        f.write(f'{m["rotation"]:.1f}\n')


def cell_centers_latlon(m: dict):
    """Return (LAT, LON) of cell centres, shape (mnx, mny)."""
    x0, y0 = m['origin'][0], m['origin'][1]
    x_edges = np.concatenate([[x0], x0 + np.cumsum(m['dx'])])
    y_edges = np.concatenate([[y0], y0 + np.cumsum(m['dy'])])
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(LAT_ORIGIN)))
    lat = LAT_ORIGIN + xc * deg_per_m_lat
    lon = LON_ORIGIN + yc * deg_per_m_lon
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')
    return LAT, LON


# ============ EMODnet bathymetry CSV ============
def read_bathymetry(path: Path):
    """Read EMODnet 'no land' CSV. Returns (lats_sorted, lons_sorted, elev_grid).
    Land cells are NaN in the grid (no entry in the CSV)."""
    print(f"  Loading {path.name} ...")
    df = pd.read_csv(path)
    df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude', 'elevation'])

    lats = np.sort(df['latitude'].unique())
    lons = np.sort(df['longitude'].unique())
    print(f"  {len(df)} marine cells on a {len(lats)} × {len(lons)} grid")
    print(f"    lat: [{lats.min():.4f}, {lats.max():.4f}]")
    print(f"    lon: [{lons.min():.4f}, {lons.max():.4f}]")
    print(f"    elevation (m): [{df['elevation'].min():.1f}, {df['elevation'].max():.1f}]")

    elev = np.full((len(lats), len(lons)), np.nan)
    lat_to_i = {v: i for i, v in enumerate(lats)}
    lon_to_j = {v: j for j, v in enumerate(lons)}
    for _, row in df.iterrows():
        elev[lat_to_i[row['latitude']], lon_to_j[row['longitude']]] = row['elevation']
    return lats, lons, elev


# ============ MAIN ============
def main():
    print("== apply_bathymetry ==")
    print("Reading model...")
    m = read_rho(RHO_IN)
    print(f"  Grid: {m['mnx']}x{m['mny']}x{m['mnz']}, format {m['fmt']}")
    print(f"  Origin (m): {m['origin']}, rotation {m['rotation']}")

    print("Reading bathymetry...")
    lats, lons, elev_csv = read_bathymetry(BATHY_CSV)

    # Interpolator on the regular bathymetry grid; nearest neighbour is fine
    # because EMODnet ~115 m is comparable to the model's 150 m fine cells.
    interp = RegularGridInterpolator(
        (lats, lons), elev_csv,
        method='nearest', bounds_error=False, fill_value=np.nan,
    )

    # Cell-centre lat/lon for the entire grid → query bathymetry once per (ix, iy)
    print("Sampling bathymetry at every model cell-centre...")
    LAT, LON = cell_centers_latlon(m)
    pts = np.stack([LAT.ravel(), LON.ravel()], axis=-1)
    elev_at_cells = interp(pts).reshape(LAT.shape)
    seafloor_depth = -elev_at_cells   # positive depth, NaN for land

    n_in_grid = np.sum(~np.isnan(seafloor_depth))
    print(f"  Cells with valid bathymetry inside the model grid: {n_in_grid} / "
          f"{m['mnx']*m['mny']}  ({100*n_in_grid/(m['mnx']*m['mny']):.1f}%)")
    valid = ~np.isnan(seafloor_depth) & (seafloor_depth > 0)
    if valid.any():
        print(f"  Seafloor depth range over the grid: "
              f"{seafloor_depth[valid].min():.1f} – {seafloor_depth[valid].max():.1f} m")

    # Cumulative depth at the TOP of each layer (in m)
    depth_top = np.concatenate([[0.0], np.cumsum(m['dz'])])[:-1]   # shape (mnz,)

    # Identify marine columns from the CURRENT model (any cell == log(0.3))
    log_sea      = math.log(RHO_SEA)
    log_sediment = math.log(RHO_SEDIMENT)
    log_land     = math.log(RHO_LAND)

    is_sea_cell  = np.abs(m['vals'] - log_sea) < SEA_TOL          # (mnz, mnx, mny)
    is_marine_col = np.any(is_sea_cell, axis=0)                   # (mnx, mny)

    # Marine columns where we additionally have bathymetry
    use_bathy = is_marine_col & ~np.isnan(seafloor_depth) & (seafloor_depth > 0)
    print(f"  Marine columns to update with bathymetry: {use_bathy.sum()} "
          f"(out of {is_marine_col.sum()} total marine columns in model)")

    # Apply per layer
    new_vals = m['vals'].copy()
    n_water_now = 0
    n_sediment_now = 0
    for k in range(m['mnz']):
        layer_top = depth_top[k]
        # In columns flagged for update:
        above_sf = use_bathy & (layer_top <  seafloor_depth)
        below_sf = use_bathy & (layer_top >= seafloor_depth)
        new_vals[k][above_sf] = log_sea
        new_vals[k][below_sf] = log_sediment
        n_water_now    += int(above_sf.sum())
        n_sediment_now += int(below_sf.sum())

    print(f"  Cells set to water    (0.3 Ω·m):  {n_water_now}")
    print(f"  Cells set to sediment ({RHO_SEDIMENT:.0f} Ω·m):  {n_sediment_now}")
    print(f"  Land columns left untouched:  {(~is_marine_col).sum()}")

    # Quick before/after summary of unique values
    before = np.unique(np.round(np.exp(m['vals']), 3))
    after  = np.unique(np.round(np.exp(new_vals), 3))
    print(f"  Resistivities before: {before}")
    print(f"  Resistivities after:  {after}")

    print(f"Writing {RHO_OUT}...")
    write_rho(RHO_OUT, m, new_vals)
    print("Done.")


if __name__ == '__main__':
    main()
