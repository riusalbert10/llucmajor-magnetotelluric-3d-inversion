"""
Interactive depth-slice viewer for the 3D resistivity model  (v3)
=================================================================

Self-contained, single-file HTML viewer:

  * Depth slider + play/pause over the real inverted layers (no vertical
    interpolation) plus a log-scaled depth gauge on the left.
  * Original colour map: jet (reversed) on a log10(rho) scale.
  * CRISP rendering — no smoothing of the heat-map or of the iso-contours.
  * Iso-contour lines labelled directly in Ω·m.
  * Dual coordinates: distance in km from the survey centre (bottom / left)
    and geographic longitude / latitude (top / right).
  * A fixed compass rose (North is up; the map is not rotated).
  * 12 x 12 km field of view, centred on the survey centre, that encloses
    every MT station.

Only numpy is required (Plotly is loaded from a CDN inside the HTML).

Outputs:
    C:\\Users\\alber\\TFG\\visualizations\\depth_slices_interactive.html
"""
import os
import re
import json
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))

#RHO_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_recond_diag_dep_10s_ODef5_D15/mallorca_10s_od5_d15_inv_NLCG_086.rho")
#DAT_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_recond_diag_dep_10s_ODef5_D15/mallorca_10s_od5_d15_inv_NLCG_086.dat")

RHO_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_recond_diag_dep_10s_ef5_ef7/Run_635/mallorca_recond_inv_NLCG_092.rho")
DAT_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_recond_diag_dep_10s_ef5_ef7/Run_635/mallorca_recond_inv_NLCG_092.dat")

#RHO_FILE = r"c:\Users\alber\TFG\MT_Llucmajor_results\Model_Comparable_Tesis_Arango\Run_632_results\mallorca_17km_seafixed_ef5_cov_inv_NLCG_067.rho"
#DAT_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_Comparable_Tesis_Arango/Run_632_results/mallorca_17km_seafixed_ef5_cov_inv_NLCG_067.dat")

#RHO_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat/Run_475_results/job_475_Mallorca_Inv_coast_bat_sm3/mallorca_coast_Bat_sm3_inv_NLCG_129.rho")
#DAT_FILE = str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat/Run_475_results/job_475_Mallorca_Inv_coast_bat_sm3/mallorca_coast_Bat_sm3_inv_NLCG_129.dat")

OUT_HTML = str(TFG_DIR / "visualizations/depth_slices_interactive.html")

PAD_THRESHOLD_M = 5_000.0            # crop padding cells beyond this distance (m) from centre (as in the original)
COLOR_VMIN, COLOR_VMAX = 1.0, 1e4    # resistivity colour-scale range (ohm·m, log)
CONTOUR_STEP_LOG = 1.0               # iso-contour spacing in log10(res) units (one per decade)

# Fixed 90° CW alignment of the resistivity grid w.r.t. the (fixed) station
# positions — this is the data correction validated in the original script,
# NOT an interactive rotation.  Leave at 1.
ROTATE_CLOCKWISE_TURNS = 1

# Target display depths (m): 50 m steps down to 700 m, then 100 m steps to 2 km.
# Each target is shown using the nearest real model layer (no interpolation).
DEPTH_TARGETS_M = np.concatenate([np.arange(50.0, 700.0 + 1, 50.0),
                                  np.arange(800.0, 2000.0 + 1, 100.0)])


# ---------------------------------------------------------------------------
# Parsers  (unchanged behaviour)
# ---------------------------------------------------------------------------
def parse_rho(path):
    """Parse a ModEM/WS-format .rho file into a dict."""
    with open(path) as f:
        lines = f.read().splitlines()

    idx = 0
    while idx < len(lines) and (lines[idx].strip().startswith('#')
                                or lines[idx].strip() == ''):
        idx += 1

    parts = lines[idx].split()
    nx, ny, nz = int(parts[0]), int(parts[1]), int(parts[2])
    log_mode = parts[4].upper() if len(parts) >= 5 else 'LINEAR'
    idx += 1

    def read_floats(start, n):
        vals, i = [], start
        while len(vals) < n and i < len(lines):
            stripped = lines[i].strip()
            if stripped:
                vals.extend(float(v) for v in stripped.split())
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
        stripped = lines[idx].strip()
        if stripped:
            leftover.extend(stripped.split())
        idx += 1

    origin = (0.0, 0.0, 0.0)
    rotation = 0.0
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
    """Read the .dat header -> (centre_lat, centre_lon, stations)."""
    centre_lat = centre_lon = None
    stations = {}

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('>'):
                parts = stripped[1:].split()
                if len(parts) == 2:
                    try:
                        lat = float(parts[0]); lon = float(parts[1])
                        if -90 <= lat <= 90 and -180 <= lon <= 180 \
                                and centre_lat is None:
                            centre_lat, centre_lon = lat, lon
                    except ValueError:
                        pass
                continue
            if stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) >= 11:
                sid = parts[1].split('_')[0]
                if sid in stations:
                    continue
                try:
                    stations[sid] = dict(
                        name=sid,
                        lat=float(parts[2]),
                        lon=float(parts[3]),
                        x_m=float(parts[4]),
                        y_m=float(parts[5]),
                    )
                except ValueError:
                    pass

    return centre_lat, centre_lon, list(stations.values())


# ---------------------------------------------------------------------------
# 1) Read data
# ---------------------------------------------------------------------------
print('Parsing .rho file ...')
rho = parse_rho(RHO_FILE)
print(f"  Model grid : {rho['nx']} x {rho['ny']} x {rho['nz']} ({rho['log_mode']})")

print('Parsing .dat header (coordinates) ...')
centre_lat, centre_lon, stations = parse_dat_header(DAT_FILE)
print(f'  Survey centre : ({centre_lat:.6f}, {centre_lon:.6f})')
print(f'  Stations read : {len(stations)}')


# ---------------------------------------------------------------------------
# 2) Build grid in metres relative to survey centre, crop, align
# ---------------------------------------------------------------------------
ox, oy, oz = rho['origin']                  # ModEM: X=north, Y=east, Z=down

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
z_km     = z_centres            / 1000.0
res_core = rho['res'][n0:n1, e0:e1, :]

# Fixed 90° CW alignment (square-trim then rot90) — preserves validated orientation
if ROTATE_CLOCKWISE_TURNS:
    n_n_full, n_e_full = res_core.shape[:2]
    if n_n_full != n_e_full:
        n_keep = min(n_n_full, n_e_full)
        if n_n_full > n_keep:
            lo = (n_n_full - n_keep) // 2
            res_core = res_core[lo:lo + n_keep, :, :]
            north_km = north_km[lo:lo + n_keep]
        if n_e_full > n_keep:
            lo = (n_e_full - n_keep) // 2
            res_core = res_core[:, lo:lo + n_keep, :]
            east_km = east_km[lo:lo + n_keep]
    res_core = np.rot90(res_core, k=ROTATE_CLOCKWISE_TURNS, axes=(0, 1))

print(f'  Core grid : {res_core.shape[0]} x {res_core.shape[1]} cells')
print(f'  east  {east_km[0]:+.1f} .. {east_km[-1]:+.1f} km | '
      f'north {north_km[0]:+.1f} .. {north_km[-1]:+.1f} km')
print(f'  Depths    : {z_km.min():.3f} - {z_km.max():.2f} km ({rho["nz"]} layers)')


# ---------------------------------------------------------------------------
# 3) Geographic conversion + payload
# ---------------------------------------------------------------------------
DEG_PER_M_LAT = 1.0 / 111_000.0
DEG_PER_M_LON = 1.0 / (111_000.0 * np.cos(np.radians(centre_lat)))
km_to_lon = lambda km: centre_lon + km * 1000.0 * DEG_PER_M_LON
km_to_lat = lambda km: centre_lat + km * 1000.0 * DEG_PER_M_LAT

zlog_min, zlog_max = float(np.log10(COLOR_VMIN)), float(np.log10(COLOR_VMAX))
log_res = np.log10(np.clip(res_core, COLOR_VMIN, COLOR_VMAX))          # (n,e,z)

# Map each target depth to the nearest real model layer (no interpolation)
z_centres_m = z_centres                                  # cell-centre depths (m)
sel = [int(np.argmin(np.abs(z_centres_m - t))) for t in DEPTH_TARGETS_M]
print('  Depth targets -> nearest model layers:')
for t, k in zip(DEPTH_TARGETS_M, sel):
    print(f'    {t:6.0f} m  ->  layer {k+1:2d} (centre {z_centres_m[k]:7.1f} m)')

cube = [np.round(log_res[:, :, k], 2).tolist() for k in sel]
depths_disp_km = [round(float(t) / 1000.0, 4) for t in DEPTH_TARGETS_M]
depths_real_km = [round(float(z_centres_m[k]) / 1000.0, 4) for k in sel]
layer_idx = [k + 1 for k in sel]

# per-cell [lon, lat] for hover (constant across depth); shape [n_north][n_east][2]
cd = [[[round(float(km_to_lon(e)), 4), round(float(km_to_lat(n)), 4)]
       for e in east_km] for n in north_km]

# Axis extent = cropped data extent, origin = survey centre (framing as in the original)
xr = [round(float(east_km[0]), 4),  round(float(east_km[-1]), 4)]
yr = [round(float(north_km[0]), 4), round(float(north_km[-1]), 4)]
x_step = max(1, int(np.ceil((east_km[-1] - east_km[0]) / 10)))
y_step = max(1, int(np.ceil((north_km[-1] - north_km[0]) / 10)))
xticks = [float(t) for t in np.arange(np.ceil(east_km[0]),  np.floor(east_km[-1]) + 1, x_step)]
yticks = [float(t) for t in np.arange(np.ceil(north_km[0]), np.floor(north_km[-1]) + 1, y_step)]
lonticks = [f'{km_to_lon(t):.3f}deg'.replace('deg', chr(176)) for t in xticks]
latticks = [f'{km_to_lat(t):.3f}deg'.replace('deg', chr(176)) for t in yticks]

stations_payload = [
    dict(name=s['name'],
         x=round(s['y_m'] / 1000.0, 4),   # east  (km)
         y=round(s['x_m'] / 1000.0, 4),   # north (km)
         lat=round(s['lat'], 5),
         lon=round(s['lon'], 5))
    for s in stations
]

m = re.search(r'_(\d+)\.rho$', os.path.basename(RHO_FILE))
iter_no = (m.group(1).lstrip('0') or '0') if m else '?'
model_label = os.path.basename(os.path.dirname(os.path.dirname(RHO_FILE)))

DATA = dict(
    east_km=[round(float(v), 4) for v in east_km],
    north_km=[round(float(v), 4) for v in north_km],
    cube=cube, cd=cd,
    depths=depths_disp_km, real_depths=depths_real_km, layer_idx=layer_idx,
    vmin=zlog_min, vmax=zlog_max, cstep=float(CONTOUR_STEP_LOG),
    stations=stations_payload,
    xticks=xticks, yticks=yticks, lonticks=lonticks, latticks=latticks,
    xr=xr, yr=yr,
    title='3D Resistivity Model — Depth Slices',
    subtitle=(f'Mallorca MT survey · ModEM NLCG iter {iter_no} · {model_label} · '
              f'centre ({centre_lat:.4f}, {centre_lon:.4f})'),
)
data_json = json.dumps(DATA, separators=(',', ':'))


# ---------------------------------------------------------------------------
# 4) HTML / JS viewer template  (Plotly from CDN)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Resistivity depth slices</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js" charset="utf-8"></script>
<style>
  :root{
    --bg:#0f172a; --bg2:#111c33; --card:#ffffff; --ink:#0f172a;
    --muted:#64748b; --line:#e2e8f0; --accent:#2563eb; --accent2:#1d4ed8;
    --shadow:0 12px 40px rgba(2,6,23,.18);
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0}
  body{
    font-family:"Segoe UI",system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif;
    color:var(--ink);
    background:
      radial-gradient(1200px 600px at 80% -10%, #1e2a4a 0%, rgba(30,42,74,0) 60%),
      radial-gradient(1000px 500px at -10% 110%, #16284d 0%, rgba(22,40,77,0) 55%),
      linear-gradient(160deg,var(--bg) 0%, var(--bg2) 100%);
    min-height:100vh; padding:26px 18px 40px;
  }
  .app{max-width:1080px;margin:0 auto;background:var(--card);border-radius:18px;
       box-shadow:var(--shadow);overflow:hidden}
  header{padding:20px 26px 14px;border-bottom:1px solid var(--line);
         background:linear-gradient(180deg,#ffffff,#fbfdff)}
  header h1{margin:0;font-size:19px;font-weight:700;letter-spacing:.2px}
  header p{margin:5px 0 0;font-size:12.5px;color:var(--muted)}
  .stage{display:flex;gap:8px;padding:16px 18px 4px;align-items:stretch}
  .gauge-wrap{flex:0 0 80px;display:flex;flex-direction:column;align-items:center}
  .gauge-wrap .lbl{font-size:10.5px;color:var(--muted);text-align:center;margin-bottom:2px;line-height:1.2}
  .plotwrap{position:relative;flex:1 1 auto;min-width:0}
  #plot{width:100%;height:580px}
  .compass{position:absolute;top:60px;left:74px;width:72px;height:72px;
           filter:drop-shadow(0 4px 10px rgba(2,6,23,.18));user-select:none;
           pointer-events:none;z-index:5}
  .hint{position:absolute;left:50%;transform:translateX(-50%);bottom:6px;font-size:11px;color:var(--muted);
        background:rgba(255,255,255,.8);backdrop-filter:blur(3px);
        padding:3px 8px;border-radius:8px;border:1px solid var(--line);pointer-events:none}
  .controls{padding:6px 26px 20px}
  .row{display:flex;align-items:center;gap:14px;margin-top:12px}
  .row .tag{flex:0 0 96px;font-size:12px;font-weight:600;color:#334155;
            text-transform:uppercase;letter-spacing:.6px}
  .row .val{flex:0 0 120px;font-size:13px;color:var(--ink);font-variant-numeric:tabular-nums}
  .row .val b{font-size:15px}
  input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:999px;
     background:linear-gradient(90deg,var(--accent) 0%,var(--accent) var(--fill,0%),#dbe3ef var(--fill,0%));
     outline:none;flex:1 1 auto;cursor:pointer}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:18px;height:18px;
     border-radius:50%;background:#fff;border:3px solid var(--accent);box-shadow:0 2px 6px rgba(2,6,23,.25);cursor:pointer}
  input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;
     border:3px solid var(--accent);box-shadow:0 2px 6px rgba(2,6,23,.25);cursor:pointer}
  .btn{border:1px solid var(--line);background:#fff;color:#334155;border-radius:10px;
       padding:7px 12px;font-size:12.5px;font-weight:600;cursor:pointer;transition:.15s}
  .btn:hover{border-color:var(--accent);color:var(--accent);background:#f8fbff}
  .btn.primary{background:var(--accent);border-color:var(--accent);color:#fff;min-width:46px}
  .btn.primary:hover{background:var(--accent2)}
  footer{padding:12px 26px 20px;border-top:1px solid var(--line);
         font-size:11.5px;color:var(--muted);display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
  footer b{color:#334155}
</style>
</head>
<body>
<div class="app">
  <header>
    <h1 id="title"></h1>
    <p id="subtitle"></p>
  </header>

  <div class="stage">
    <div class="gauge-wrap">
      <div class="lbl">Depth<br/>(km · log)</div>
      <svg id="gauge" width="80" height="560" aria-label="depth gauge"></svg>
    </div>
    <div class="plotwrap">
      <div id="plot"></div>
      <div class="compass" id="compass"></div>
    </div>
  </div>

  <div class="controls">
    <div class="row">
      <div class="tag">Depth</div>
      <div class="val"><b id="depthVal">—</b> km</div>
      <input type="range" id="depth" min="0" max="0" value="0" step="1"/>
      <button class="btn primary" id="play">▶</button>
      <button class="btn" id="resetZoom" title="Back to initial view">⤢ Reset zoom</button>
    </div>
  </div>

  <footer>
    <span id="footNote">▽ MT stations · North up · contours in Ω·m · origin = survey centre</span>
    <span id="layerInfo"></span>
  </footer>
</div>

<script>
const DATA = __DATA_JSON__;

// ---------- state ----------
let curK = 0;
const nz = DATA.depths.length;
const XR = DATA.xr, YR = DATA.yr;            // initial (max) extent
const EK = DATA.east_km, NK = DATA.north_km;

// ---------- traces ----------
const colorbar = {
  title:{text:'ρ (Ω·m)', side:'top', font:{size:12.5}},
  tickvals:[0,1,2,3,4], ticktext:['1','10','100','1k','10k'],
  len:0.9, thickness:14, x:1.2, outlinewidth:0, tickfont:{size:11},
  ticks:'outside', ticklen:4,
};
function heat(k){
  return {type:'heatmap', x:EK, y:NK, z:DATA.cube[k], zsmooth:'best',
    colorscale:'Jet', reversescale:true, zmin:DATA.vmin, zmax:DATA.vmax,
    colorbar:colorbar, customdata:DATA.cd, xaxis:'x', yaxis:'y', name:'rho',
    hovertemplate:'Lon %{customdata[0]:.4f}° · Lat %{customdata[1]:.4f}°<br>'+
                  'E %{x:.2f} · N %{y:.2f} km<br>ρ ≈ 10<sup>%{z:.2f}</sup> Ω·m<extra></extra>'};
}
function cont(k){
  return {type:'contour', x:EK, y:NK, z:DATA.cube[k], xaxis:'x', yaxis:'y',
    contours:{start:DATA.vmin, end:DATA.vmax, size:DATA.cstep, coloring:'none',
              showlabels:true, labelfont:{size:9.5, color:'#1f2937'}},
    line:{color:'rgba(15,23,42,0.55)', width:0.9, smoothing:0},
    showscale:false, hoverinfo:'skip', name:'iso'};
}
const stationsTrace = {type:'scatter',
  x:DATA.stations.map(s=>s.x), y:DATA.stations.map(s=>s.y),
  xaxis:'x', yaxis:'y', mode:'markers',
  marker:{symbol:'triangle-down', size:9, color:'#ffffff', line:{color:'#0f172a', width:1.2}},
  customdata:DATA.stations.map(s=>[s.name, s.lat, s.lon]),
  hovertemplate:'<b>%{customdata[0]}</b><br>Lat %{customdata[1]:.4f}° · Lon %{customdata[2]:.4f}°<br>'+
                'E %{x:.2f} · N %{y:.2f} km<extra></extra>',
  showlegend:false, name:'MT'};
const phantom = {type:'scatter', x:[XR[0],XR[1]], y:[YR[0],YR[1]], xaxis:'x2', yaxis:'y2',
  mode:'markers', marker:{opacity:0}, hoverinfo:'skip', showlegend:false};

const axBase = {zeroline:false, showgrid:false, constrain:'domain'};
const layout = {
  margin:{l:64, r:170, t:50, b:54},
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#ffffff',
  xaxis:Object.assign({}, axBase, {title:{text:'East (km)', font:{size:12}},
        side:'bottom', range:XR.slice(), tickmode:'array', tickvals:DATA.xticks, ticktext:DATA.xticks.map(String)}),
  yaxis:Object.assign({}, axBase, {title:{text:'North (km)', font:{size:12}},
        side:'left', range:YR.slice(), scaleanchor:'x', scaleratio:1, tickmode:'array', tickvals:DATA.yticks, ticktext:DATA.yticks.map(String)}),
  xaxis2:Object.assign({}, axBase, {overlaying:'x', side:'top', range:XR.slice(),
        tickmode:'array', tickvals:DATA.xticks, ticktext:DATA.lonticks,
        title:{text:'Longitude', font:{size:11.5}}, tickfont:{size:10}}),
  yaxis2:Object.assign({}, axBase, {overlaying:'y', side:'right', range:YR.slice(),
        tickmode:'array', tickvals:DATA.yticks, ticktext:DATA.latticks,
        title:{text:'Latitude', font:{size:11.5}}, tickfont:{size:10}}),
  hovermode:'closest', hoverlabel:{bgcolor:'#0f172a', font:{color:'#fff', size:12}},
  dragmode:'zoom', showlegend:false,
};
const config = {responsive:true, displaylogo:false, scrollZoom:false,
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'],
  toImageButtonOptions:{filename:'depth_slice', scale:2}};

const gd = document.getElementById('plot');
Plotly.newPlot(gd, [heat(0), cont(0), stationsTrace, phantom], layout, config);

// ---------- contour labels in Ω·m ----------
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
  const texts = gd.querySelectorAll('g.contourlabels text');
  texts.forEach(t => {
    if (t.textContent.indexOf('Ω') !== -1) return;          // already converted
    const v = parseFloat(t.textContent.trim().replace('−','-').replace(',','.'));
    if (isFinite(v)) t.textContent = ohmLabel(v);
  });
}
gd.on('plotly_afterplot', relabelContours);
setTimeout(relabelContours, 60);

function render(){ Plotly.restyle(gd, {z:[DATA.cube[curK]]}, [0,1]); }

// ---------- zoom: never zoom out past the initial extent, keep geo axes synced ----------
let _busy = false;
function applyRanges(xr, yr){
  _busy = true;
  Plotly.relayout(gd, {
    'xaxis.range':xr.slice(),  'yaxis.range':yr.slice(),
    'xaxis2.range':xr.slice(), 'yaxis2.range':yr.slice(),
    'xaxis.autorange':false,   'yaxis.autorange':false
  }).then(() => { _busy = false; });
}
function clampToView(){
  if (_busy) return;
  const xr = (gd.layout.xaxis.range || XR), yr = (gd.layout.yaxis.range || YR);
  const x0 = Math.max(Math.min(xr[0], xr[1]), XR[0]);
  const x1 = Math.min(Math.max(xr[0], xr[1]), XR[1]);
  const y0 = Math.max(Math.min(yr[0], yr[1]), YR[0]);
  const y1 = Math.min(Math.max(yr[0], yr[1]), YR[1]);
  applyRanges([x0, x1], [y0, y1]);
}
gd.on('plotly_relayout', (ev) => {
  if (_busy || !ev) return;
  const touched = ('xaxis.range[0]' in ev) || ('yaxis.range[0]' in ev) ||
                  ev['xaxis.autorange'] || ev['yaxis.autorange'] ||
                  ('xaxis.range' in ev) || ('yaxis.range' in ev);
  if (touched) clampToView();
});
document.getElementById('resetZoom').onclick = () => applyRanges(XR, YR);

// ---------- compass (static, North up) ----------
document.getElementById('compass').innerHTML = `
<svg viewBox="0 0 100 100" width="72" height="72">
  <defs><radialGradient id="cg" cx="50%" cy="38%" r="65%">
    <stop offset="0%" stop-color="#ffffff"/><stop offset="100%" stop-color="#eef2f8"/>
  </radialGradient></defs>
  <circle cx="50" cy="50" r="46" fill="url(#cg)" stroke="#cbd5e1" stroke-width="1.5"/>
  <circle cx="50" cy="50" r="38" fill="none" stroke="#e2e8f0" stroke-width="1"/>
  <g stroke="#94a3b8" stroke-width="1">
    <line x1="50" y1="9" x2="50" y2="17"/><line x1="50" y1="83" x2="50" y2="91"/>
    <line x1="9" y1="50" x2="17" y2="50"/><line x1="83" y1="50" x2="91" y2="50"/>
  </g>
  <polygon points="50,12 43,52 57,52" fill="#ef4444"/>
  <polygon points="50,88 43,48 57,48" fill="#64748b"/>
  <circle cx="50" cy="50" r="4.2" fill="#0f172a"/>
  <text x="50" y="27" text-anchor="middle" font-size="12" font-weight="700" fill="#ef4444" font-family="system-ui">N</text>
  <text x="50" y="80" text-anchor="middle" font-size="9" font-weight="600" fill="#64748b" font-family="system-ui">S</text>
  <text x="79" y="54" text-anchor="middle" font-size="9" font-weight="600" fill="#64748b" font-family="system-ui">E</text>
  <text x="21" y="54" text-anchor="middle" font-size="9" font-weight="600" fill="#64748b" font-family="system-ui">W</text>
</svg>`;

// ---------- depth gauge (log scale) ----------
function buildGauge(){
  const svg = document.getElementById('gauge');
  const H2 = 560, top = 22, bot = H2 - 26, x = 31;
  const dmin = DATA.depths[0], dmax = DATA.depths[nz-1];
  const lmin = Math.log10(dmin), lmax = Math.log10(dmax);
  window._gy = d => top + (Math.log10(d) - lmin) / (lmax - lmin) * (bot - top);
  const ticks = [0.05,0.1,0.2,0.5,1,2,5,10,20,30].filter(t => t >= dmin*0.999 && t <= dmax*1.001);
  let s = '';
  s += `<rect x="${x-3}" y="${top}" width="6" height="${bot-top}" rx="3" fill="#eef2f8" stroke="#e2e8f0"/>`;
  s += `<rect id="gfill" x="${x-3}" y="${top}" width="6" height="0" rx="3" fill="#bfdbfe"/>`;
  for (const t of ticks){
    const y = window._gy(t);
    s += `<line x1="${x-7}" y1="${y}" x2="${x+7}" y2="${y}" stroke="#94a3b8" stroke-width="1"/>`;
    s += `<text x="${x+12}" y="${y+3}" font-size="9.5" fill="#64748b" font-family="system-ui">${t}</text>`;
  }
  s += `<g id="gmark" transform="translate(0,${top})">
          <polygon points="${x-13},-6 ${x-13},6 ${x-4},0" fill="#2563eb"/>
          <line x1="${x-4}" y1="0" x2="${x+9}" y2="0" stroke="#2563eb" stroke-width="2"/></g>`;
  svg.innerHTML = s;
}
function setGauge(d){
  const y = window._gy(d), top = 22;
  const m = document.getElementById('gmark'), f = document.getElementById('gfill');
  if (m) m.setAttribute('transform', `translate(0,${y})`);
  if (f) f.setAttribute('height', Math.max(0, y - top));
}

// ---------- UI ----------
const depthEl = document.getElementById('depth');
const depthVal = document.getElementById('depthVal');
const layerInfo = document.getElementById('layerInfo');
function setFill(el){ const pct = (el.value - el.min) / (el.max - el.min) * 100; el.style.setProperty('--fill', pct + '%'); }
function updateDepthUI(){
  const d = DATA.depths[curK];
  depthVal.textContent = d.toFixed(d < 1 ? 3 : 2);
  layerInfo.innerHTML = `Slice <b>${curK+1}/${nz}</b> · target z = <b>${d.toFixed(d<1?3:2)} km</b>` +
    ` · model layer ${DATA.layer_idx[curK]} (centre ${DATA.real_depths[curK].toFixed(3)} km)`;
  depthEl.value = curK; setFill(depthEl); setGauge(d);
}
depthEl.min = 0; depthEl.max = nz - 1;
depthEl.addEventListener('input', () => { curK = +depthEl.value; updateDepthUI(); render(); });

let playing = false, timer = null;
const playBtn = document.getElementById('play');
function play(){ playing = true; playBtn.textContent = '⏸';
  timer = setInterval(() => { curK = (curK + 1) % nz; updateDepthUI(); render(); }, 320); }
function pause(){ playing = false; playBtn.textContent = '▶'; clearInterval(timer); }
playBtn.onclick = () => playing ? pause() : play();

// ---------- init ----------
document.getElementById('title').textContent = DATA.title;
document.getElementById('subtitle').textContent = DATA.subtitle;
buildGauge(); updateDepthUI();
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
print(f'  View          : east {DATA["xr"][0]:+.1f}..{DATA["xr"][1]:+.1f} km, '
      f'north {DATA["yr"][0]:+.1f}..{DATA["yr"][1]:+.1f} km   slices: {len(cube)}   '
      f'stations: {len(stations_payload)}')
print('Done!')
