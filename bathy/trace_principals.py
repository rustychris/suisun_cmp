import numpy as np
import pandas as pd
import six
import matplotlib.pyplot as plt
from matplotlib import collections
import xarray as xr
from stompy import utils
utils.path("/home/rusty/src/hor_flow_and_salmon/bathy")
from stompy.model import stream_tracer
from stompy.plot import plot_utils

##

ds=xr.open_dataset("current-summary.nc")
g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)
g.edge_to_cells()

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
zoom=(583157., 588467., 4226178, 4230135)

# This has some issues in places where the starting bathy is bad, as those
# places start dry.  In most places, it's a decent check
#ccoll=g.plot_cells(values=ds.min_waterdepth.values,cmap='viridis',clip=zoom)
#ccoll.set_clim([0,0.1])
ccoll=g.plot_cells(values=ds.max_waterdepth.values,cmap='viridis',clip=zoom)
ccoll.set_clim([0,0.5])

cc=g.cells_center()
sel=g.cell_clip_mask(zoom) & (ds.urms.values>0.0)

plt.quiver(cc[sel,0],cc[sel,1],
           ds.princ.values[sel,0],
           ds.princ.values[sel,1],
           ds.urms.values[sel],
           headlength=0.0,headwidth=1,cmap='plasma',
           scale_units='xy',pivot='middle',scale=0.03,headaxislength=0.0)

ax.axis((583406.3318506933, 584328.3674565726, 4227095.339747712, 4227668.034620019))

##

Uc=(ds.urms*ds.princ).values


# easy point out in Montezuma
x0=np.array([587784.3, 4226320.0])

six.moves.reload_module(stream_tracer)
trace=stream_tracer.steady_streamline_twoways(g,Uc,x0,max_t=20*3600,bidir=True)

Ucrot=utils.rot(np.pi/2,Uc)
trace_perp=stream_tracer.steady_streamline_twoways(g,Ucrot,x0,max_t=20*3600,bidir=True)

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g.plot_cells(values=ds.urms.values,cmap='viridis',ax=ax)

ax.plot(trace.x.values[:,0],
        trace.x.values[:,1],
        'r-o',alpha=0.4)

ax.plot(trace_perp.x.values[:,0],
        trace_perp.x.values[:,1],
        'g-o',alpha=0.4)

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

ax.scatter(cutoff_xyz[:,0],
           cutoff_xyz[:,1],
           20,
           cutoff_xyz[:,2])
##
six.moves.reload_module(stream_tracer)
stream_tracer.prepare_grid(g)

u_min=1e-02

alongs=[]

for i in utils.progress(range(len(cutoff_xyz))):
    x0=cutoff_xyz[i,:2]
    along=stream_tracer.steady_streamline_twoways(g,Uc,x0,max_t=20*3600,bidir=True,max_dist=500.,
                                                  u_min=u_min)
    alongs.append(along)

acrosses=[]

for i in utils.progress(range(len(cutoff_xyz))):
    x0=cutoff_xyz[i,:2]
    across=stream_tracer.steady_streamline_twoways(g,Ucrot,x0,max_t=20*3600,bidir=True,max_dist=100.,
                                                   u_min=u_min)
    acrosses.append(across)
    
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

# Across traces shouldn't trim first/last segment, though.
# but there is maybe an issue with across traces hitting
# the shoreline -- a

x_segs=[ a.x.values for a in acrosses]
xcoll=collections.LineCollection(x_segs,color='b',lw=0.5,alpha=0.5)
ax.add_collection(xcoll)

##

# Since this isn't a steady run, it does end up running some traces into small coves
# Unclear whether this will be a problem with the interpolation.
# Take a quick look at umag, eccent, to see if there is anything useful for weeding these
# out.
#   umag is not specific enough,  and unsurprisingly eccent is useless.
# could also just trim the last segment from all traces.

g.plot_edges(clip=ax.axis(),ax=ax,color='k')

##

# Debugging across traces that hit the shore and run parallel.
x0=np.array([584317.7444372588, 4227767.260970293])

trace=stream_tracer.steady_streamline_oneway(g,-Ucrot,x0,max_t=20*3600,bidir=True,max_dist=100.)

clip=(584239, 584380, 4227723, 4227813)
plt.figure(2).clf()
fig,ax=plt.subplots(1,1,num=2)

g.plot_edges(color='k',clip=clip,ax=ax)
ax.plot(trace.x.values[:,0],trace.x.values[:,1],'r-o')
ax.axis(clip)

g.plot_cells(clip=clip,labeler=lambda i,r:str(i),color='none',ax=ax)

# Those cells do have a small velocity -- probably from wetting/drying
# during sloshing.
# 

sel_cell=g.cell_clip_mask(clip)

ax.quiver(g.cells_center()[sel_cell,0],
          g.cells_center()[sel_cell,1],
          ds.princ.values[sel_cell,0],
          ds.princ.values[sel_cell,1])

##

# Options:
#  - press on, hoping that bad traces are only in places that don't matter
#    that much.

#  - does the dual help here at all?
#    a little tricky, since I don't have the edge fluxes, just the cell-center
#    principal directions.
#    If I did have some sort of edge-based quantity --
#      maybe.  doesn't seem like a silver bullet.

#  - Look into ways of extracting a steady flow field obeying continuity
#    based on the cell-centered vectors.
#    if I want to deal with dead-end sloughs, there is no steady, conservative
#    field that will put flow into a dead-end slough (short of creating artificial
#    sources and sinks)

#  - Look into a simulation that could be run that would be more amenable
#    to this, like forcing steady flow through the domain, running a longer
#    simulation and filtering, etc.
#    - what about steady with inflows and evaporation?
#      that gets a little weird with multiple inflows.  potentially have
#      a stagnation point in the middle of the domain.
#    - maybe something as simple as running a longer tidal run and discarding
#      early sloshing.
#    - what about a propagating wave?  separately for each open boundary, force
#      an impulse

#  - Create a regionally consistent velocity field, i.e. within the range of
#    a trace.  falls apart in the vicinity of a cutoff slough.

#  - Filter out cells that were dry any/most of the time.
#    with newly updated current-summary.nc output, tried this.  With the existing
#    run it is not good, as there are areas that start off dry due to bad initial
#    bathy.

# How bad is it to solve for potential flow directly?  Again, this does no
# good in dead-end sloughs.

scat=ax.scatter(cutoff_xyz[:,0],
                cutoff_xyz[:,1],
                20,
                cutoff_xyz[:,2])

##

# What about packaging up more of the process, so there is more control over the
# initial run?


##

# Generating a grid:
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

plt.figure(3).clf()
fig,ax=plt.subplots(1,1,num=3)


scat=ax.scatter(cutoff_xyz[:,0],
                cutoff_xyz[:,1],
                20,
                cutoff_xyz[:,2])
plot_wkb.plot_wkb(total_poly,zorder=-2)

ax.axis('equal')

##
from stompy.spatial import field
from stompy.grid import paver
six.moves.reload_module(paver)

p=paver.Paving(geom=total_poly,density=field.ConstantField(Lres))
p.verbose=1
p.pave_all()

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


##
# from stompy.model.suntans import sun_driver
# model=sun_driver.SuntansModel()
# 
# model.set_grid(g_sub)
# 
# model.add_bc(sun_driver.StageBC(


