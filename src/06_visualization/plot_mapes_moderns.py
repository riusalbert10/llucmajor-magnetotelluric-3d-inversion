# -*- coding: utf-8 -*-
"""
Mapes moderns per al TFG (estil càlid/teal):
  1) Xarxa d'estacions MT/AMT amb poble, carreteres i els 5 perfils perpendiculars.
  2) Gradient geotèrmic amb isotermes (30/40/50 C), punts mesurats i inset de Mallorca.

Estacions: coordenades reals del KML -> UTM31N (precises).
Perfils: llegits de plot_cross_sections_profiles_figure.py (ajust PCA + 0.5 km).
Poble/carreteres/isotermes/punts tèrmics: traçat aproximat (esquemàtic) de les
figures de referència.
"""
import re, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from matplotlib.lines import Line2D
from scipy.interpolate import splprep, splev
from pyproj import Transformer
import os
from pathlib import Path

CREAM="#FBF7EF"; INK="#2B2B2B"; TEAL="#147A72"; GRIDC="#D9D0C2"
ROAD="#B9AE9C"; TOWN="#C9BEA9"; PROF="#7A7A7A"
C30="#E9B44C"; C40="#DE7A3B"; C50="#B3402F"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
    "axes.edgecolor":INK,"axes.linewidth":1.0,"text.color":INK,
    "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK})

# Project folder holding the data and ModEM runs (not included in the repo).
# Defaults to the repository root; override it with the TFG_DIR environment variable.
TFG_DIR = Path(os.environ.get("TFG_DIR", Path(__file__).resolve().parents[2]))

KMLP=str(TFG_DIR / "MT_Llucmajor_results/Model_Smooth3_Bat/Generar_Costa_Batimetria/Coordenades_sites.KML")
CSP =str(TFG_DIR / "visualizations/plot_cross_sections_profiles_figure.py")
OUT1=str(TFG_DIR / "visualizations/mapa_estacions_modern.png")
OUT2=str(TFG_DIR / "visualizations/mapa_gradient_modern.png")

tr=Transformer.from_crs("EPSG:4326","EPSG:32631",always_xy=True)
MT_IDS={10:"K10",11:"K11",12:"K12",13:"K13",14:"K14",31:"K31"}
amt={}; mt={}; allpos={}
for m in re.finditer(r"<name>(mall\d+)</name>.*?<coordinates>([^<]+)</coordinates>",open(KMLP).read(),re.S):
    n=int(re.search(r'\d+',m.group(1)).group()); lo,la=[float(v) for v in m.group(2).split(",")[:2]]
    e,nth=tr.transform(lo,la); allpos[n]=(e,nth)
    (mt if n in MT_IDS else amt)[n]=(e,nth)

perp=[]
for m in re.finditer(r"\('perfil\d+_perp',\s*'perp',\s*\[([0-9,\s]+)\]\)",open(CSP).read()):
    perp.append([int(x) for x in m.group(1).split(",")])
EXTEND=500.0
def profile_line(nums):
    pts=np.array([allpos[n] for n in nums if n in allpos],float)
    c=pts.mean(0); d=pts-c; _,_,vt=np.linalg.svd(d); ax=vt[0]
    pr=d@ax; s=c+ax*(pr.min()-EXTEND); e=c+ax*(pr.max()+EXTEND)
    return [s[0],e[0]],[s[1],e[1]]

EXT=[487000,495000,4367400,4372100]
TOWN_POLY=[
(490861.7,4370760.0),(490677.2,4370874.3),(490585.0,4370862.9),(490412.1,4370908.6),
(490377.5,4370908.6),(490314.1,4370811.4),(490262.2,4370782.9),(490123.9,4370965.7),
(489985.6,4370977.1),(489922.2,4371028.6),(490118.2,4371325.7),(490095.1,4371360.0),
(490106.6,4371405.7),(490072.0,4371440.0),(490083.6,4371474.3),(490037.5,4371520.0),
(490014.4,4371611.4),(490147.0,4371777.1),(490210.4,4371760.0),(490245.0,4371897.1),
(490342.9,4371971.4),(490435.2,4371902.9),(490550.4,4371880.0),(490619.6,4371834.3),
(490683.0,4371942.9),(490711.8,4371971.4),(490734.9,4371948.6),(490804.0,4372040.0),
(490919.3,4371971.4),(490976.9,4372085.7),(491005.8,4372057.1),(491005.8,4371954.3),
(491167.1,4371794.3),(491144.1,4371737.1),(491409.2,4371531.4),(491374.6,4371451.4),
(491513.0,4371348.6),(491063.4,4370891.4),(490988.5,4370771.4),(490896.3,4370805.7),
(490861.7,4370760.0)
]
town=TOWN_POLY
roads=[]
def smooth_line(xy,n=200):
    p=np.array(xy,float)
    if len(p)<4: return p[:,0],p[:,1]
    (tck,u)=splprep([p[:,0],p[:,1]],s=0,k=min(3,len(p)-1)); uu=np.linspace(0,1,n)
    return splev(uu,tck)

def base(ax,title):
    ax.set_facecolor("white"); ax.set_xlim(EXT[0],EXT[1]); ax.set_ylim(EXT[2],EXT[3])
    ax.set_aspect("equal"); ax.grid(True,color=GRIDC,lw=0.6,alpha=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Est (UTM)"); ax.set_ylabel("Nord (UTM)")
    ax.set_xticks(range(487000,495001,1000)); ax.set_yticks(range(4368000,4372001,1000))
    ax.tick_params(labelsize=8); ax.ticklabel_format(style="plain",useOffset=False)
    ax.set_title(title,fontsize=14,fontweight="bold",pad=10)

# ===================== MAPA 1 =====================
fig,ax=plt.subplots(figsize=(9.8,6.4)); fig.patch.set_facecolor("white")
base(ax,"Xarxa d'estacions magnetotel·lúriques — Llucmajor")
for r in roads:
    x,y=smooth_line(r); ax.plot(x,y,color=ROAD,lw=1.4,zorder=1,solid_capstyle="round")
ax.add_patch(MPoly(town,closed=True,facecolor=TOWN,edgecolor="#A99A80",lw=0.8,zorder=2))
ax.text(491650,4371520,"Llucmajor",fontsize=15,fontstyle="italic",fontweight="bold",family="serif",ha="left",va="center",zorder=6)
for i,nums in enumerate(perp):
    xx,yy=profile_line(nums); ax.plot(xx,yy,color=PROF,lw=1.7,ls="--",alpha=0.6,zorder=3,solid_capstyle="round")
    j=0 if yy[0]<yy[1] else 1
    ax.annotate(str(i+1),(xx[j],yy[j]),xytext=(0,-11),textcoords="offset points",ha="center",va="top",fontsize=12,fontweight="bold",color="#444444",zorder=6)
ax.scatter([v[0] for v in amt.values()],[v[1] for v in amt.values()],s=42,facecolor=TEAL,edgecolor="white",linewidth=0.8,zorder=4)
for n,(e,nth) in amt.items():
    ax.annotate(str(n),(e,nth),xytext=(4,3),textcoords="offset points",fontsize=6.5,zorder=5)
for n,(e,nth) in mt.items():
    ax.scatter([e],[nth],s=95,marker="P",facecolor=INK,edgecolor="white",linewidth=0.8,zorder=5)
    ax.annotate(MT_IDS[n],(e,nth),xytext=(5,4),textcoords="offset points",fontsize=8,fontweight="bold",zorder=6)
for _n in (3,6,9,25):
    if _n in amt:
        _e,_nn=amt[_n]; ax.scatter([_e],[_nn],marker="+",s=150,c=INK,linewidths=2.0,zorder=6)
ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=TEAL,markeredgecolor='white',markersize=9,label='Estació AMT'),
     Line2D([0],[0],marker='P',color='none',markerfacecolor=INK,markeredgecolor='white',markersize=11,label='Estació MT'),
     Line2D([0],[0],color=PROF,lw=1.8,ls='--',alpha=0.6,label='Perfil')],
     loc="lower right",fontsize=9,framealpha=0.95,facecolor="white",edgecolor=GRIDC,title="Símbols",title_fontsize=9)
fig.tight_layout(); fig.savefig(OUT1,dpi=240,bbox_inches="tight"); plt.close(fig)

# ===================== MAPA 2 =====================
def smooth_iso(pts,closed,n=300):
    p=np.array(pts,float)
    if closed: p=np.vstack([p,p[0]])
    (tck,u)=splprep([p[:,0],p[:,1]],s=0,per=1 if closed else 0,k=3)
    uu=np.linspace(0,1,n); return splev(uu,tck)
iso30=[(489250,4370700),(488550,4370400),(488150,4369850),(488250,4369300),(488700,4368700),(489100,4368250),(489650,4368150)]
iso40=[(489650,4370680),(490250,4370550),(490550,4370200),(490350,4369750),(489800,4369500),(489250,4369650),(488950,4370050),(489150,4370500)]
iso50=[(489550,4370400),(489700,4370600),(490000,4370600),(490200,4370450),(490050,4370280),(489700,4370280)]
measured=[(487650,4369350),(488050,4369520),(488300,4370050),(488300,4369550),(489150,4369950),(489450,4370150),
 (489720,4370520),(489980,4370500),(490060,4370350),(490050,4370000),(490350,4370050),(490560,4370000),
 (490100,4369250),(490360,4369300),(490800,4370000),(490900,4369250),(489350,4369200),(490560,4368760)]
fig,ax=plt.subplots(figsize=(9.8,6.4)); fig.patch.set_facecolor("white")
base(ax,"Gradient geotèrmic — isotermes (°C)")
for r in roads:
    x,y=smooth_line(r); ax.plot(x,y,color=ROAD,lw=1.1,alpha=0.85,zorder=1)
ax.add_patch(MPoly(town,closed=True,facecolor=TOWN,edgecolor="#A99A80",lw=0.8,zorder=2))
ax.text(491650,4371520,"Llucmajor",fontsize=15,fontstyle="italic",fontweight="bold",family="serif",ha="left",va="center",zorder=6)
for pts,col,val,cl,lab in [(iso30,C30,"30",False,(488250,4369550)),(iso40,C40,"40",True,(489050,4370150)),(iso50,C50,"50",True,(489870,4370440))]:
    x,y=smooth_iso(pts,cl); ax.plot(x,y,color=col,lw=3.0,solid_capstyle="round",zorder=3)
    ax.text(*lab,val,color=col,fontsize=11,fontweight="bold",ha="center",va="center",zorder=6,
            bbox=dict(boxstyle="round,pad=0.1",fc=CREAM,ec="none",alpha=0.85))
ax.scatter([p[0] for p in measured],[p[1] for p in measured],marker="x",s=55,c=INK,linewidth=1.6,zorder=5)
ax.legend(handles=[Line2D([0],[0],marker='x',color=INK,lw=0,markersize=9,markeredgewidth=1.8,label='Punt mesurat'),
     Line2D([0],[0],color=C40,lw=3,label='Isoterma (°C)')],
     loc="lower right",fontsize=9,framealpha=0.95,facecolor="white",edgecolor=GRIDC,title="Símbols",title_fontsize=9)
mall=[(2.37,39.56),(2.70,39.86),(3.15,39.96),(3.45,39.76),(3.47,39.55),(3.20,39.35),(2.95,39.28),(2.75,39.32),(2.50,39.42)]
axi=ax.inset_axes([0.66,0.66,0.32,0.32]); axi.set_facecolor("#EAF1F2")
axi.add_patch(MPoly(mall,closed=True,facecolor=TOWN,edgecolor="#9c8f76",lw=0.8))
axi.plot(2.89,39.49,marker="*",ms=12,c=C50,mec="white",mew=0.8)
axi.text(2.92,39.36,"Mallorca",fontsize=8,fontstyle="italic",ha="center")
axi.set_xlim(2.3,3.5); axi.set_ylim(39.25,40.0); axi.set_xticks([]); axi.set_yticks([])
for s in axi.spines.values(): s.set_edgecolor(INK)
fig.tight_layout(); fig.savefig(OUT2,dpi=240,bbox_inches="tight"); plt.close(fig)
print("done")
