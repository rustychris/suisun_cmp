import numpy as np
import pandas as pd
import six
import matplotlib.pyplot as plt
from matplotlib import collections
import xarray as xr
from stompy import utils
utils.path("/home/rusty/src/hor_flow_and_salmon/bathy")
import stream_tracer
from stompy.plot import plot_utils

##

ds=xr.open_dataset("current-summary.nc")
g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)
g.edge_to_cells()

##

# Load the CSDP data -- see how that traces.

csdp_df=pd.read_csv("/home/rusty/data/usace/csdp_data/1999_2000_utm83_Navd88.xyz",
                     names=['x','y','z','year','project'])
sel=(csdp_df.project=='DWR-EST')|(csdp_df.project=='DWR-ZIGJHO')
subset=csdp_df.iloc[sel.values,:]
csdp_xyz=subset.loc[:, ['x','y','z'] ].values

# relevant projects:
# DWR-ZIGJHO
# DWR-EST
# those contain way more than we need right now, though.
clip=(583734., 589138, 4226370, 4229150)
sel=utils.within_2d(csdp_xyz[:,:2],clip)

cutoff_xyz=csdp_xyz[sel,:]
    
##

# Unfortunately a good portion of the remaining time is in xarray. 
# %prun -s cumulative stream_tracer.steady_streamline_twoways(g,Uc,x0,max_t=20*3600,bidir=True,max_dist=500.)
plt.figure(2).clf()
fig,ax=plt.subplots(1,1,num=2)
g.edges['mark'] = (g.edge_to_cells().min(axis=1)<0)

g.plot_boundary()

# Trim first and last, as these are either at boundaries, so not very useful,
# or worse they are in dead-end cells which suffer from lack of terracing
# This gets >90% of the whiskers, but it's not bullet proof.  Pairs of
# dead-end cells conspire.
a_segs=[ a.x.values[1:-1] for a in alongs if len(a.x.values)>3]
acoll=collections.LineCollection(a_segs,color='r',lw=0.5,alpha=0.5)
ax.add_collection(acoll)


##

scat=ax.scatter(cutoff_xyz[:,0],
                cutoff_xyz[:,1],
                20,
                cutoff_xyz[:,2])

##

# What about packaging up more of the process, so there is more control over the
# initial run?

# A length scale defining the max distance that a sample can influence, and
# likewise defines the relevant domain for a set of samples
L=50.0    # These would yield something like 50k cells.
Lres=10.0 # 

from shapely import geometry
from shapely.ops import cascaded_union
from stompy.plot import plot_wkb

buffers=[geometry.Point(x).buffer(L,8) for x in cutoff_xyz[:,:2]]
total_poly=cascaded_union(buffers)

assert total_poly.type=='Polygon'

total_poly=total_poly.simplify(Lres/2.)

##

# But better to just use a subset of the existing grid.
select=g.select_nodes_intersecting(total_poly)

g_sub=g.copy()

for n in np.nonzero(~select)[0]:
    g_sub.delete_node_cascade(n)
g_sub.renumber()
g_sub.orient_edges()

##
plt.figure(1).clf()
g_sub.plot_edges()


##

# Define open boundary locations:
open_boundaries=[ np.array([[ 588459.4, 4226736.8],
                            [ 588481.1, 4226712.7],
                            [ 588543.8, 4226689.4]]),
                  np.array([[ 584185.8, 4227673.5],
                            [ 584232.2, 4227634.4],
                            [ 584261.8, 4227606.9]]) ]

##

z_hw=2.0
z_init=-5.0

g_sub.add_cell_field('z_bed',z_init*np.ones(g_sub.Ncells()))

##

# What do diffusion contours look like?
boundary=open_boundaries[0]
boundary_edges=g_sub.select_edges_by_polyline(boundary)
boundary_cells=g_sub.edge_to_cells(boundary_edges)[:,0]

##

from stompy.model import unstructured_diffuser

differ=unstructured_diffuser.Diffuser(grid=g_sub)
for c in boundary_cells:
    differ.set_flux(1.0,cell=c)
differ.set_decay_rate(1e-5)

differ.construct_linear_system()

differ.solve_linear_system()

##


differ.calc_fluxes()
differ.calc_flux_vectors_and_grad()
differ.flux_vector_c

##

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

g_sub.plot_cells(values=np.log(differ.C_solved),cmap='jet')
ax.axis('equal')

unit_flux_vec=utils.to_unit(differ.flux_vector_c)

ax.quiver( g_sub.cells_center()[:,0],
           g_sub.cells_center()[:,1],
           unit_flux_vec[:,0],unit_flux_vec[:,1])
           
##

# That's not too bad...
# See how it traces using just the one boundary, but will come back to layer in
# additional boundaries.
six.moves.reload_module(stream_tracer)
stream_tracer.prepare_grid(g_sub)

alongs=[]

for i in utils.progress(range(len(cutoff_xyz))):
    x0=cutoff_xyz[i,:2]
    along=stream_tracer.steady_streamline_twoways(g_sub,unit_flux_vec,
                                                  x0,max_t=20*3600,bidir=False,max_dist=500.)
    alongs.append(along)

##
acrosses=[]

unit_flux_vec_rot=utils.rot(np.pi/2,unit_flux_vec)

for i in utils.progress(range(len(cutoff_xyz))):
    x0=cutoff_xyz[i,:2]
    across=stream_tracer.steady_streamline_twoways(g_sub,unit_flux_vec_rot,
                                                   x0,max_t=20*3600,bidir=False,max_dist=100.)
    acrosses.append(across)

##

a_segs=[ a.x.values for a in alongs]
acoll=collections.LineCollection(a_segs,color='r',lw=0.5,alpha=0.5)
ax.add_collection(acoll)

x_segs=[ a.x.values for a in acrosses]
xcoll=collections.LineCollection(x_segs,color='b',lw=0.5,alpha=0.5)
ax.add_collection(xcoll)

# Generally speaking not terrible.

##
six.moves.reload_module(stream_tracer)

SD=stream_tracer.StreamDistance(g=g_sub,U=unit_flux_vec,U_rot=unit_flux_vec_rot,
                                alongs=alongs,acrosses=acrosses,source_ds=source_ds)
stream_distance=SD.stream_distance


source_ds=xr.Dataset()
source_ds['x']=('sample','xy'),cutoff_xyz[:,:2]
source_ds['z']=('sample',),cutoff_xyz[:,2]

def samples_for_target(SD,x_target,N=500):
    x_along=SD.trace_along(x_target)
    x_across=SD.trace_across(x_target)

    # nearby source samples
    dists=utils.mag( x_target-source_ds.x )
    close_samples=np.argsort(dists)[:N]

    close_distances=[]

    for s in close_samples:
        close_distances.append( SD.stream_distance(x_target,s,
                                                   x_along=x_along,
                                                   x_across=x_across) )
    close_distances=np.array(close_distances)
    ds=xr.Dataset()
    ds['target']=('xy',),x_target.copy()
    ds['target_z']=(), np.nan 
    ds['stream_dist']=('sample','st'),close_distances
    ds['sample_z']=('sample',),source_ds.z.values[close_samples]
    ds['sample_xy']=('sample','xy'),source_ds.x.values[close_samples]
    return ds

##
x_target=np.array([585942.0, 4227041.8])
samples=samples_for_target(SD,x_target)

##

plt.figure(1).clf()
g_sub.plot_edges(color='k',lw=0.6)
plt.scatter( samples.sample_xy.values[:,0],
             samples.sample_xy.values[:,1],
             20,
             samples.sample_z.values )

##

all_targets=[]
cc=g_sub.cells_center()
for c in range(g_sub.Ncells()):
    print(c)
    x_target=cc[c]
    target_samples=samples_for_target(SD,x_target)
    all_targets.append(target_samples)
    

import pickle
with open('cutoff-targets2.pkl','wb') as fp:
    pickle.dump(all_targets, fp)

##

