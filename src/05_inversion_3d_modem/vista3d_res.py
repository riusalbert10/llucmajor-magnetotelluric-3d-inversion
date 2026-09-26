#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vista 3D del model MT a ESCALA REAL (amb padding), coloreada per log10(resistivitat)
amb colorbar, estil software de malla (Voxler/3DGrid).
"""
import numpy as np, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

MODEL="/sessions/charming-upbeat-allen/mnt/Arxius ejecucio/Model_Bat_35km_recond"
DATA ="/sessions/charming-upbeat-allen/mnt/Arxius_ejecució/Mall_mask_no_tip"
OUT  ="/sessions/charming-upbeat-allen/mnt/Arxius_ejecució/vista3d_res.png"

f=open(MODEL).read().split("\n"); h=f[1].split(); Nx,Ny,Nz=int(h[0]),int(h[1]),int(h[2])
nums=lambda s:[float(x) for x in s.split()]
dx,dy,dz=nums(f[2]),nums(f[3]),nums(f[4]); ox,oy=-17503.0,-17503.0
rows=[nums(f[i]) for i in range(5,len(f)) if len(f[i].split())==Nx][:Nx*Nz]
A=np.transpose(np.array(rows).reshape(Nz,Nx,Ny),(1,2,0))     # [x,y,z] ln(rho)
L=A/math.log(10)                                             # log10(rho)

# vores reals (km)
xe=(ox+np.concatenate([[0],np.cumsum(dx)]))/1e3
ye=(oy+np.concatenate([[0],np.cumsum(dy)]))/1e3
ze=np.concatenate([[0],np.cumsum(dz)])/1e3                   # profunditat km

from matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list("res",
     ["#d00000","#ff8000","#ffff00","#22b000","#00c0c0","#1030ff"])   # vermell->groc->verd->cian->blau
norm=Normalize(0,3)
col=lambda v: cmap(norm(np.clip(v,0,3)))

polys=[]; cols=[]
def q(p,c): polys.append(p); cols.append(c)
# cara superior
for i in range(Nx):
    for j in range(Ny):
        q([(xe[i],ye[j],0),(xe[i+1],ye[j],0),(xe[i+1],ye[j+1],0),(xe[i],ye[j+1],0)], col(L[i,j,0]))
# 4 parets
for i in range(Nx):
    for k in range(Nz):
        q([(xe[i],ye[0],ze[k]),(xe[i+1],ye[0],ze[k]),(xe[i+1],ye[0],ze[k+1]),(xe[i],ye[0],ze[k+1])],col(L[i,0,k]))
        q([(xe[i],ye[Ny],ze[k]),(xe[i+1],ye[Ny],ze[k]),(xe[i+1],ye[Ny],ze[k+1]),(xe[i],ye[Ny],ze[k+1])],col(L[i,Ny-1,k]))
for j in range(Ny):
    for k in range(Nz):
        q([(xe[0],ye[j],ze[k]),(xe[0],ye[j+1],ze[k]),(xe[0],ye[j+1],ze[k+1]),(xe[0],ye[j],ze[k+1])],col(L[0,j,k]))
        q([(xe[Nx],ye[j],ze[k]),(xe[Nx],ye[j+1],ze[k]),(xe[Nx],ye[j+1],ze[k+1]),(xe[Nx],ye[j],ze[k+1])],col(L[Nx-1,j,k]))

fig=plt.figure(figsize=(12,8.5)); ax=fig.add_subplot(111,projection="3d")
ax.computed_zorder=False
ax.add_collection3d(Poly3DCollection(polys,facecolors=cols,edgecolor=(0,0,0,0.04),linewidths=0.08,zorder=1))

# estacions
cx=ox+np.concatenate([[0],np.cumsum(dx)]); cx=0.5*(cx[:-1]+cx[1:])
cy=oy+np.concatenate([[0],np.cumsum(dy)]); cy=0.5*(cy[:-1]+cy[1:])
st={}
for l in open(DATA):
    p=l.split()
    if len(p)>=11 and not l.strip().startswith(("#",">")):
        try: st[p[1]]=(float(p[4])/1e3,float(p[5])/1e3)
        except: pass
sx=[v[0] for v in st.values()]; sy=[v[1] for v in st.values()]
ax.scatter(sx,sy,np.full(len(sx),-0.8),c="k",s=9,depthshade=False,zorder=10)

ax.set_xlim(xe[0],xe[-1]); ax.set_ylim(ye[0],ye[-1]); ax.set_zlim(0,ze[-1])
ax.set_xlabel("X, Nord (km)"); ax.set_ylabel("Y, Est (km)"); ax.set_zlabel("Profunditat (km)")
ax.set_zticks([0,16,32,48,64])
ax.invert_zaxis(); ax.view_init(elev=22, azim=135); ax.set_box_aspect((1,1,1.35))
ax.text2D(0.01,0.98,"Nx=%d  Ny=%d  Nz=%d"%(Nx,Ny,Nz),transform=ax.transAxes,
          fontsize=9,fontweight="bold",va="top")

sm=ScalarMappable(norm=norm,cmap=cmap); sm.set_array([])
cb=fig.colorbar(sm,ax=ax,shrink=0.5,pad=0.02,ticks=[0,1,2,3])
cb.set_label("Log10 [Resistivitat (Ω·m)]",fontweight="bold")
fig.savefig(OUT,dpi=140,bbox_inches="tight"); plt.close(fig)
print("escrit:",OUT)
