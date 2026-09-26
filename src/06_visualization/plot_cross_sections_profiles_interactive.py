"""
Interactive vertical cross-sections along the MT profiles
=========================================================

Self-contained, single-file HTML viewer (same visual style as
plot_depth_slices_interactive.py) showing a vertical resistivity
cross-section for each survey profile.

The MT stations were laid out along lines that are perpendicular and
parallel to the geo-electric strike.  For each profile this script:

  * reads the station coordinates from the .edi files,
  * fits a straight line (PCA / total-least-squares) to obtain the
    profile orientation (azimuth) and end points,
  * samples the 3-D inversion model (.rho) along that oblique vertical
    plane, from the surface down to MAX_DEPTH_KM,
  * draws a log-scale resistivity heat-map (jet, reversed) with iso-
    contours labelled in ohm.m and the projected station positions.

Buttons switch between profiles.  Horizontal axis = distance along the
profile (km), vertical axis = depth (km, 0 .. MAX_DEPTH_KM).

Only numpy is required (Plotly is loaded from a CDN inside the HTML).

Output:
    C:\\Users\\alber\\TFG\\visualizations\\cross_sections_profiles_interactive.html
"""
import os
import re
import json
import glob
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#RHO_FILE = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Smooth3_Bat\Run_475_results\job_475_Mallorca_Inv_coast_bat_sm3\mallorca_coast_Bat_sm3_inv_NLCG_129.rho"
#DAT_FILE = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Smooth3_Bat\Run_475_results\job_475_Mallorca_Inv_coast_bat_sm3\mallorca_coast_Bat_sm3_inv_NLCG_129.dat"

#RHO_FILE = r"c:\Users\alber\TFG\MT_Llucmajor_results\Model_Comparable_Tesis_Arango\Run_632_results\mallorca_17km_seafixed_ef5_cov_inv_NLCG_067.rho"
#DAT_FILE = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_Comparable_Tesis_Arango\Run_632_results\mallorca_17km_seafixed_ef5_cov_inv_NLCG_067.dat"

RHO_FILE = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_recond_diag_dep_10s_ef5_ef7\Run_635\mallorca_recond_inv_NLCG_092.rho"
DAT_FILE = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_recond_diag_dep_10s_ef5_ef7\Run_635\mallorca_recond_inv_NLCG_092.dat"

#RHO_FILE = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_recond_diag_dep_10s_ODef5_D15\mallorca_10s_od5_d15_inv_NLCG_086.rho"
#DAT_FILE = r"C:\Users\alber\TFG\MT_Llucmajor_results\Model_recond_diag_dep_10s_ODef5_D15\mallorca_10s_od5_d15_inv_NLCG_086.dat"

EDI_DIR  = r"C:\Users\alber\TFG\LLUCMAJOR_DADES_edi"
OUT_HTML = r"C:\Users\alber\TFG\visualizations\cross_sections_profiles_interactive.html"

PAD_THRESHOLD_M = 6_000.0            # crop padding cells beyond this distance (m) from centre
MAX_DEPTH_KM    = 1.5               # maximum depth shown on the sections
COLOR_VMIN, COLOR_VMAX = 1.0, 1e4   # resistivity colour-scale range (ohm.m, log)
CONTOUR_STEP_LOG = 0.5              # iso-contour spacing in log10(rho) units (half-decade)
ROTATE_CLOCKWISE_TURNS = 1          # validated alignment of .rho grid vs stations (as in depth viewer)
N_SAMPLES = 160                     # samples along each profile
EXTEND_KM = 0.5                     # extend each section this far (km) beyond the end stations

# Profiles (station numbers).  'perp' = perpendicular to strike, 'par' = parallel.
PROFILES = [
    ('perfil1_perp', 'perp', [10, 4, 2, 3, 6, 5, 7, 8, 9, 24]),
    ('perfil2_perp', 'perp', [65, 1, 51, 67, 23]),
    ('perfil3_perp', 'perp', [64, 52, 66, 22]),
    ('perfil4_perp', 'perp', [63, 55, 41, 42]),
    ('perfil5_perp', 'perp', [62, 61, 53, 26]),
    ('perfil1_par',  'par',  [30, 32, 34, 33, 27, 28, 29, 26, 21, 24]),
    ('perfil2_par',  'par',  [42, 66, 67, 9]),
    ('perfil3_par',  'par',  [61, 52, 51, 7]),
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
    """station number -> (lat, lon) from REFLAT/REFLONG of each mallNN.edi."""
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
print('Parsing .rho file ...')
rho = parse_rho(RHO_FILE)
print(f"  Model grid : {rho['nx']} x {rho['ny']} x {rho['nz']} ({rho['log_mode']})")

centre_lat, centre_lon = parse_dat_centre(DAT_FILE)
print(f'  Survey centre : ({centre_lat:.6f}, {centre_lon:.6f})')

edi = parse_edi_coords(EDI_DIR)
print(f'  EDI stations  : {len(edi)}')


# ---------------------------------------------------------------------------
# 2) Build aligned model grid (same prep as the depth-slice viewer)
# ---------------------------------------------------------------------------
ox, oy, oz = rho['origin']
north_nodes = ox + np.concatenate(([0.0], np.cumsum(rho['dx'])))
east_nodes  = oy + np.concatenate(([0.0], np.cumsum(rho['dy'])))
z_nodes     = oz + np.concatenate(([0.0], np.cumsum(rho['dz'])))
north_centres = (north_nodes[:-1] + north_nodes[1:]) / 2
east_centres  = (east_nodes[:-1]  + east_nodes[1:])  / 2
z_centres     = (z_nodes[:-1]     + z_nodes[1:])     / 2

n_core = np.where(np.abs(north_centres) < PAD_THRESHOLD_M)[0]
e_core = np.where(np.abs(east_centres)  < PAD_THRESHOLD_M)[0]
n0, n1 = n_core[0], n_core[-1] + 1
e0, e1 = e_core[0], e_core[-1] + 1
north_km = north_centres[n0:n1] / 1000.0
east_km  = east_centres[e0:e1]  / 1000.0
z_km_all = z_centres / 1000.0
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

# Keep depth layers down to (just past) MAX_DEPTH_KM
zsel = np.where(z_km_all <= MAX_DEPTH_KM)[0]
zlast = (zsel[-1] + 2) if len(zsel) else 4          # one extra layer for the bottom edge
zlast = min(zlast, len(z_km_all))
z_km = z_km_all[:zlast]
log_res = np.log10(np.clip(res_core[:, :, :zlast], COLOR_VMIN, COLOR_VMAX))   # (n_n, n_e, nz)
nz_sel = log_res.shape[2]
print(f'  Aligned core : {res_core.shape[0]} x {res_core.shape[1]} cells | '
      f'east {east_km[0]:+.1f}..{east_km[-1]:+.1f}  north {north_km[0]:+.1f}..{north_km[-1]:+.1f} km')
print(f'  Depth layers <= {MAX_DEPTH_KM} km : {nz_sel} (to {z_km[-1]:.3f} km)')


# ---------------------------------------------------------------------------
# 3) Geographic helpers + station positions (survey-centre frame)
# ---------------------------------------------------------------------------
DEG_PER_M_LAT = 1.0 / 111_000.0
DEG_PER_M_LON = 1.0 / (111_000.0 * np.cos(np.radians(centre_lat)))
lonlat_of = lambda e_km, n_km: (centre_lon + e_km * 1000.0 * DEG_PER_M_LON,
                                centre_lat + n_km * 1000.0 * DEG_PER_M_LAT)

def station_en(num):
    """station number -> (east_km, north_km) in survey-centre frame, or None."""
    if num not in edi:
        return None
    lat, lon = edi[num]
    e = (lon - centre_lon) / (DEG_PER_M_LON * 1000.0)
    n = (lat - centre_lat) / (DEG_PER_M_LAT * 1000.0)
    return e, n


# ---------------------------------------------------------------------------
# 4) Bilinear sampler of the aligned model over (north_km, east_km)
# ---------------------------------------------------------------------------
def sample_section(line_e, line_n):
    """Return log-resistivity array of shape (nz_sel, N) along the line."""
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
print('Building profiles ...')
profiles_payload = []
for name, kind, nums in PROFILES:
    present = [num for num in nums if station_en(num) is not None]
    missing = [num for num in nums if station_en(num) is None]
    pts = np.array([station_en(num) for num in present])          # (M, 2) east, north
    c = pts.mean(axis=0)
    d = pts - c
    _, _, vt = np.linalg.svd(d)
    axis = vt[0]                                                  # unit direction (east, north)
    if np.degrees(np.arctan2(axis[0], axis[1])) % 360.0 >= 180.0:
        axis = -axis                                             # +distance points toward bearing 0..180
    proj = d @ axis
    pmin = float(proj.min()) - EXTEND_KM                         # extend past the first station
    pmax = float(proj.max()) + EXTEND_KM                         # extend past the last station
    length = pmax - pmin
    az = np.degrees(np.arctan2(axis[0], axis[1])) % 360.0        # bearing of +distance direction (0..180)
    start = c + axis * pmin
    end = c + axis * pmax
    dist = np.linspace(0.0, length, N_SAMPLES)
    line_e = start[0] + dist * axis[0]
    line_n = start[1] + dist * axis[1]
    sec = sample_section(line_e, line_n)                          # (nz_sel, N)
    st_list = sorted(
        [{'num': int(num), 'dist': round(float((np.array(station_en(num)) - c) @ axis - pmin), 3)}
         for num in present],
        key=lambda r: r['dist'])
    slon, slat = lonlat_of(*start)
    elon, elat = lonlat_of(*end)
    profiles_payload.append(dict(
        name=name, kind=kind,
        az=round(float(az), 1),
        length=round(float(length), 3),
        dist=[round(float(v), 3) for v in dist],
        z=np.round(sec, 2).tolist(),
        stations=st_list,
        missing=missing,
        start=[round(slat, 5), round(slon, 5)],
        end=[round(elat, 5), round(elon, 5)],
    ))
    print(f'  {name:13s} ({kind}) n={len(present):2d} miss={missing} '
          f'len={length:4.2f}km az={az:5.1f}deg')


# ---------------------------------------------------------------------------
# 6) Payload
# ---------------------------------------------------------------------------
model_label = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(RHO_FILE))))
DATA = dict(
    profiles=profiles_payload,
    depth=[round(float(v), 4) for v in z_km],
    maxdepth=float(MAX_DEPTH_KM),
    vmin=float(np.log10(COLOR_VMIN)), vmax=float(np.log10(COLOR_VMAX)),
    cstep=float(CONTOUR_STEP_LOG),
    title='Vertical Cross-Sections along MT Profiles',
    subtitle=(f'Mallorca MT survey · {model_label} · resistivity sampled along '
              f'strike-perpendicular / -parallel profiles · centre '
              f'({centre_lat:.4f}, {centre_lon:.4f})'),
)
data_json = json.dumps(DATA, separators=(',', ':'))


# ---------------------------------------------------------------------------
# 7) HTML / JS viewer template  (Plotly from CDN)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MT profile cross-sections</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js" charset="utf-8"></script>
<style>
  :root{
    --bg:#0f172a; --bg2:#111c33; --card:#ffffff; --ink:#0f172a;
    --muted:#64748b; --line:#e2e8f0; --accent:#2563eb; --accent2:#1d4ed8;
    --perp:#2563eb; --par:#0e9488; --shadow:0 12px 40px rgba(2,6,23,.18);
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0}
  body{font-family:"Segoe UI",system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif;
    color:var(--ink);
    background:
      radial-gradient(1200px 600px at 80% -10%, #1e2a4a 0%, rgba(30,42,74,0) 60%),
      radial-gradient(1000px 500px at -10% 110%, #16284d 0%, rgba(22,40,77,0) 55%),
      linear-gradient(160deg,var(--bg) 0%, var(--bg2) 100%);
    min-height:100vh; padding:26px 18px 40px;}
  .app{max-width:1120px;margin:0 auto;background:var(--card);border-radius:18px;
       box-shadow:var(--shadow);overflow:hidden}
  header{padding:20px 26px 14px;border-bottom:1px solid var(--line);
         background:linear-gradient(180deg,#ffffff,#fbfdff)}
  header h1{margin:0;font-size:19px;font-weight:700;letter-spacing:.2px}
  header p{margin:5px 0 0;font-size:12.5px;color:var(--muted)}
  .stage{padding:14px 18px 2px;display:flex;gap:16px;align-items:stretch}
  .plotwrap{position:relative;flex:1 1 auto;min-width:0}
  #plot{width:100%;height:520px}
  .side{flex:0 0 180px;display:flex;flex-direction:column;align-items:center;
        padding:14px 10px;border:1px solid var(--line);border-radius:14px;
        background:linear-gradient(180deg,#ffffff,#f8fafc)}
  .side-lbl{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;
            color:var(--muted);text-align:center;line-height:1.3;margin-bottom:6px}
  .azread{margin-top:6px;font-size:22px;font-weight:800;letter-spacing:.5px;font-variant-numeric:tabular-nums}
  .azsub{font-size:10.5px;color:var(--muted);margin-top:0;text-align:center}
  .side .pill{margin:12px 0 0}
  .controls{padding:10px 26px 16px}
  .grp{display:flex;align-items:center;gap:12px;margin-top:10px;flex-wrap:wrap}
  .glab{flex:0 0 168px;font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:.6px}
  .glab.perp{color:var(--perp)} .glab.par{color:var(--par)}
  .btns{display:flex;gap:7px;flex-wrap:wrap}
  .btn{border:1px solid var(--line);background:#fff;color:#334155;border-radius:9px;
       padding:7px 12px;font-size:12.5px;font-weight:600;cursor:pointer;transition:.15s}
  .btn:hover{border-color:var(--accent);color:var(--accent);background:#f8fbff}
  .btn.active{color:#fff;border-color:transparent}
  .btn.active.perp{background:var(--perp)} .btn.active.par{background:var(--par)}
  .pinfo{margin-top:14px;font-size:12.5px;color:#334155;line-height:1.5}
  .pinfo b{color:var(--ink)}
  .pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;
        font-weight:700;color:#fff;margin-right:6px}
  .pill.perp{background:var(--perp)} .pill.par{background:var(--par)}
  .warn{color:#b45309}
  footer{padding:12px 26px 20px;border-top:1px solid var(--line);
         font-size:11.5px;color:var(--muted)}
</style>
</head>
<body>
<div class="app">
  <header>
    <h1 id="title"></h1>
    <p id="subtitle"></p>
  </header>

  <div class="stage">
    <div class="plotwrap">
      <div id="plot"></div>
    </div>
    <aside class="side">
      <div class="side-lbl">Profile orientation<br/>azimuth from North</div>
      <div id="compass"></div>
      <div id="azread" class="azread"></div>
      <div class="azsub">clockwise from N</div>
      <div id="kindPill"></div>
    </aside>
  </div>

  <div class="controls">
    <div class="grp"><span class="glab perp">Perpendicular to strike</span><span id="btnsPerp" class="btns"></span></div>
    <div class="grp"><span class="glab par">Parallel to strike</span><span id="btnsPar" class="btns"></span></div>
    <div id="pinfo" class="pinfo"></div>
  </div>

  <footer>
    <span>&#9661; MT stations projected onto the profile &middot; contours labelled in &#937;&middot;m &middot; depth axis 0&ndash;1 km</span>
  </footer>
</div>

<script>
const DATA = __DATA_JSON__;

// ---------- state ----------
const P = DATA.profiles;
let cur = 0;
const gd = document.getElementById('plot');

const colorbar = {
  title:{text:'ρ (Ω·m)', side:'top', font:{size:12.5}},
  tickvals:[0,1,2,3,4], ticktext:['1','10','100','1k','10k'],
  len:0.9, thickness:14, x:1.0, xpad:6, outlinewidth:0, tickfont:{size:11},
  ticks:'outside', ticklen:4,
};

function ohmLabel(v){
  const val = Math.pow(10, v);
  let s;
  if (val >= 1000){ let k = val/1000; k = k>=10 ? Math.round(k) : Math.round(k*10)/10; s = k+'k'; }
  else if (val >= 100) s = String(Math.round(val/10)*10);
  else if (val >= 10)  s = String(Math.round(val));
  else                 s = String(Math.round(val*10)/10);
  return s + ' Ω·m';
}
function relabelContours(){
  gd.querySelectorAll('g.contourlabels text').forEach(t => {
    if (t.textContent.indexOf('Ω') !== -1) return;
    const v = parseFloat(t.textContent.trim().replace('−','-').replace(',','.'));
    if (isFinite(v)) t.textContent = ohmLabel(v);
  });
}

function traces(p){
  const heat = {
    type:'heatmap', x:p.dist, y:DATA.depth, z:p.z, zsmooth:'best',
    colorscale:'Jet', reversescale:true, zmin:DATA.vmin, zmax:DATA.vmax,
    colorbar:colorbar, name:'rho',
    hovertemplate:'Distance %{x:.2f} km · Depth %{y:.3f} km<br>ρ ≈ 10<sup>%{z:.2f}</sup> Ω·m<extra></extra>'};
  const cont = {
    type:'contour', x:p.dist, y:DATA.depth, z:p.z,
    contours:{start:DATA.vmin, end:DATA.vmax, size:DATA.cstep, coloring:'none',
              showlabels:true, labelfont:{size:9.5, color:'#1f2937'}},
    line:{color:'rgba(15,23,42,0.55)', width:0.9, smoothing:0},
    showscale:false, hoverinfo:'skip', name:'iso'};
  const sx = p.stations.map(s => s.dist), snum = p.stations.map(s => String(s.num));
  const sta = {
    type:'scatter', x:sx, y:sx.map(()=>0), mode:'markers+text',
    marker:{symbol:'triangle-down', size:11, color:'#ffffff', line:{color:'#0f172a', width:1.3}},
    text:snum, textposition:'top center', textfont:{size:9.5, color:'#0f172a'},
    hovertemplate:'<b>mall%{text}</b><br>at %{x:.2f} km<extra></extra>',
    showlegend:false, name:'MT'};
  return [heat, cont, sta];
}
function layout(p){
  return {
    margin:{l:62, r:96, t:30, b:50},
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#ffffff',
    xaxis:{title:{text:'Distance along profile (km)', font:{size:12}},
           range:[0, p.length], zeroline:false, showgrid:false, mirror:false},
    yaxis:{title:{text:'Depth (km)', font:{size:12}},
           range:[DATA.maxdepth, -0.05*DATA.maxdepth], zeroline:false, showgrid:false, mirror:false},
    hovermode:'closest', hoverlabel:{bgcolor:'#0f172a', font:{color:'#fff', size:12}},
    dragmode:'pan', showlegend:false,
  };
}
const config = {responsive:true, displaylogo:false,
  modeBarButtonsToRemove:['select2d','lasso2d'],
  toImageButtonOptions:{filename:'cross_section', scale:2}};

// ---------- professional compass (outside the plot) ----------
function updateCompass(p){
  const az = p.az, col = p.kind === 'perp' ? '#2563eb' : '#0e9488';
  const C = 60, R = 54;
  let s = '';
  s += `<circle cx="60" cy="60" r="${R}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>`;
  s += `<circle cx="60" cy="60" r="${R-8}" fill="none" stroke="#eef2f8" stroke-width="1"/>`;
  // graduated ticks every 10° (major every 30°)
  for (let a = 0; a < 360; a += 10){
    const t = a*Math.PI/180, dx = Math.sin(t), dy = -Math.cos(t);
    const major = (a % 30 === 0), len = major ? 8 : 4.5, w = major ? 1.3 : 0.7;
    s += `<line x1="${(C+dx*R).toFixed(1)}" y1="${(C+dy*R).toFixed(1)}" x2="${(C+dx*(R-len)).toFixed(1)}" y2="${(C+dy*(R-len)).toFixed(1)}" stroke="#94a3b8" stroke-width="${w}"/>`;
  }
  // cardinal letters
  for (const [lab, a, cc, fs] of [['N',0,'#ef4444',12.5],['E',90,'#475569',10.5],['S',180,'#475569',10.5],['W',270,'#475569',10.5]]){
    const t = a*Math.PI/180, rr = R-17;
    s += `<text x="${(C+Math.sin(t)*rr).toFixed(1)}" y="${(C-Math.cos(t)*rr+fs/3).toFixed(1)}" text-anchor="middle" font-size="${fs}" font-weight="700" fill="${cc}" font-family="system-ui">${lab}</text>`;
  }
  // angle arc from N to the azimuth
  const t = az*Math.PI/180, ra = 17;
  s += `<path d="M ${C} ${C-ra} A ${ra} ${ra} 0 ${az>180?1:0} 1 ${(C+Math.sin(t)*ra).toFixed(1)} ${(C-Math.cos(t)*ra).toFixed(1)}" fill="none" stroke="${col}" stroke-width="1.6"/>`;
  // single-direction arrow (needle) pointing at the azimuth
  const dx = Math.sin(t), dy = -Math.cos(t), nx = -dy, ny = dx;
  const tipX = C+dx*(R-9),  tipY = C+dy*(R-9);
  const baseX = C+dx*(R-21), baseY = C+dy*(R-21), hw = 6;
  s += `<line x1="${C}" y1="${C}" x2="${baseX.toFixed(1)}" y2="${baseY.toFixed(1)}" stroke="${col}" stroke-width="3.4" stroke-linecap="round"/>`;
  s += `<polygon points="${tipX.toFixed(1)},${tipY.toFixed(1)} ${(baseX+nx*hw).toFixed(1)},${(baseY+ny*hw).toFixed(1)} ${(baseX-nx*hw).toFixed(1)},${(baseY-ny*hw).toFixed(1)}" fill="${col}"/>`;
  s += `<circle cx="60" cy="60" r="3.4" fill="#0f172a"/>`;
  document.getElementById('compass').innerHTML = `<svg viewBox="0 0 120 120" width="156" height="156">${s}</svg>`;
  const ar = document.getElementById('azread'); ar.textContent = az.toFixed(0)+'°'; ar.style.color = col;
  document.getElementById('kindPill').innerHTML =
    `<span class="pill ${p.kind}">${p.kind === 'perp' ? 'PERP. TO STRIKE' : 'PAR. TO STRIKE'}</span>`;
}

// ---------- info line ----------
function fmtLL(a){ return a[0].toFixed(4)+'°, '+a[1].toFixed(4)+'°'; }
function updateInfo(p){
  const miss = p.missing.length ? ` <span class="warn">(stations ${p.missing.join(', ')} have no EDI &rarr; skipped)</span>` : '';
  document.getElementById('pinfo').innerHTML =
    `<span class="pill ${p.kind}">${p.kind === 'perp' ? 'PERPENDICULAR' : 'PARALLEL'}</span>` +
    `<b>${p.name}</b> &middot; azimuth <b>${p.az.toFixed(0)}°</b> &middot; length <b>${p.length.toFixed(2)} km</b> ` +
    `&middot; ${p.stations.length} stations${miss}<br>` +
    `start ${fmtLL(p.start)} &rarr; end ${fmtLL(p.end)}`;
}

// ---------- buttons ----------
function buildButtons(){
  const gp = document.getElementById('btnsPerp'), ga = document.getElementById('btnsPar');
  P.forEach((p, i) => {
    const b = document.createElement('button');
    b.className = 'btn ' + p.kind;
    b.textContent = p.name.replace('perfil', 'P').replace('_perp','').replace('_par','');
    b.dataset.idx = i;
    b.onclick = () => { cur = i; render(); };
    (p.kind === 'perp' ? gp : ga).appendChild(b);
  });
}
function updateButtons(){
  document.querySelectorAll('.btns .btn').forEach(b => {
    b.classList.toggle('active', +b.dataset.idx === cur);
  });
}

// ---------- render ----------
function render(){
  const p = P[cur];
  Plotly.react(gd, traces(p), layout(p), config);
  setTimeout(relabelContours, 50);
  updateCompass(p); updateInfo(p); updateButtons();
}

// ---------- init ----------
document.getElementById('title').textContent = DATA.title;
document.getElementById('subtitle').textContent = DATA.subtitle;
buildButtons();
render();
gd.on('plotly_afterplot', relabelContours);
</script>
</body>
</html>
"""

html = HTML_TEMPLATE.replace('__DATA_JSON__', data_json)
os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = os.path.getsize(OUT_HTML) / 1024 / 1024
print(f'\nSaved: {OUT_HTML}  ({size_mb:.2f} MB)')
print(f'  Profiles: {len(profiles_payload)}   depth layers: {nz_sel}')
print('Done!')
