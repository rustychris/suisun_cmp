from stompy.spatial import field
import six
import numpy as np
import glob
import numpy as np
import skimage.graph
from scipy import ndimage
from stompy import memoize, utils
import matplotlib.pyplot as plt
import stompy.plot.cmap as scmap
turbo=scmap.load_gradient('turbo.cpt')
import subprocess

##

# 9k x 10k
dem=field.GdalGrid('../bathy/DEM_VegCorr_DWR_2m_NAVD88.tif')

##
from stompy.spatial import wkb2shp
six.moves.reload_module(wkb2shp)

gdb_fn="../bathy/levees/ESRI ArcGIS/SFBayShoreInventory__SFEI_2016.gdb"

levees=wkb2shp.shp2geom(gdb_fn,layer_patt='.*with_elevation',
                        target_srs='EPSG:26910')

# 'San_Francisco_Bay_Shore_Inventory__100ft_segments_with_elevation'

##

# Whittle down the levees to overlapping the DEM.
dem_xxyy=dem.extents
levee_bounds=[ g.bounds for g in levees['geom']]
levee_bounds=np.array(levee_bounds)

sel=( (levee_bounds[:,0]<=dem_xxyy[1]) &
      (levee_bounds[:,2]>=dem_xxyy[0]) &
      (levee_bounds[:,1]<=dem_xxyy[3]) &
      (levee_bounds[:,3]>=dem_xxyy[2]))

levee_clip=levees[sel] # 23k

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
dem.downsample(4).plot(cmap='gray',vmin=-1,vmax=5,ax=ax,zorder=-2)

ax.axis('off')

from stompy.plot import plot_wkb

for l in levee_clip['geom'][:1000]:
    plot_wkb.plot_wkb(l,ax=ax,color='m',lw=2,zorder=2)
ax.axis('equal')

##

# Would like to burn in the Z_Min from the levees to the DEM, but only
# where higher than existing values.

levee_burn_fn="levee_burn.tif"

if not os.path.exists(levee_burn_fn):
    levee_burn=dem.copy()
    levee_burn.F[:,:]=np.nan
    levee_burn.write_gdal(levee_burn_fn)

    cmd=( "gdal_rasterize -a Z_Min"
          " -l 'San_Francisco_Bay_Shore_Inventory__100ft_segments_with_elevation'"
          #" -a_nodata nan"
          f" '{gdb_fn}'"
          " levee_burn.tif"
    )
    subprocess.run(cmd,shell=True)

@memoize.memoize()
def levee_burn():
    return field.GdalGrid('levee_burn.tif')

##

force=False

@memoize.memoize()
def dem_and_levees():
    dem_and_levees_fn='dem_and_levees.tif'
    if force or not os.path.exists(dem_and_levees_fn):
        assert np.allclose( dem.extents, levee_burn.extents)
        assert dem.F.shape==levee_burn.F.shape

        leveed_dem=dem.copy()
        leveed_dem.F=np.fmax( dem.F, levee_burn.F)

        os.path.exists(dem_and_levees_fn) and os.unlink(dem_and_levees_fn)
        leveed_dem.write_gdal(dem_and_levees_fn)
    else:
        leveed_dem=field.GdalGrid(dem_and_levees_fn)
    return leveed_dem

##

# Buffer out the regions from the levees to get the region that will be
# pulled from the lidar:

@memoize.memoize()
def lidar_mask():
    mask=np.isfinite(levee_burn().F)

    mask_open = ndimage.binary_dilation( mask, iterations=5)

    lidar_mask=field.SimpleGrid( extents=levee_burn().extents, F=mask_open)
    return lidar_mask

#plt.figure(1).clf() ; lidar_mask.plot()

##

lidar_src=field.MultiRasterField(["../bathy/lidar/tiles/*.tif"])

@memoize.memoize()
def lidar_resamp():
    lidar_resamp_fn='lidar-levees.tif'

    if not os.path.exists(lidar_resamp_fn):
        # Will compile the lidar data here:
        lidar_levees=field.SimpleGrid(extents=lidar_mask().extents,
                                      F=np.zeros(lidar_mask().F.shape,np.float32))
        lidar_levees.F[:,:]=np.nan


        lidar_resamp=lidar_src.to_grid(bounds=lidar_mask().extents,
                                       dx=lidar_mask().dx,dy=lidar_mask().dy)

        lidar_resamp.F[~lidar_mask().F]=np.nan

        # There are seams, at least 1px wide, between lidar tiles.
        lidar_resamp.fill_by_convolution(iterations=1)

        lidar_resamp.write_gdal('lidar-levees.tif')
    else:
        lidar_resamp=field.GdalGrid('lidar-levees.tif')
    return lidar_resamp

##
leveed_dem=dem_and_levees()
leveed_dem.F=np.fmax( leveed_dem.F, lidar_resamp().F)

leveed_dem.write_gdal('dem_and_levees_and_lidar.tif')


## 

# Ready to starting poking around.
plt.figure(1).clf()
leveed_dem.plot(cmap=turbo,vmin=0.4,vmax=2.0)

##

# Out in Grizzly
seed_xy=np.array([583638, 4218191.])

##

z_inundate=dem.copy()
z_inundate.F[:,:]=np.nan

from scipy import ndimage

for thresh in np.arange(0,4.0,0.01):
    print(thresh)
    wet=(leveed_dem.F<=thresh).astype(np.int8)
    labels,n_features=ndimage.label(wet)
    labeled=field.SimpleGrid(extents=leveed_dem.extents,
                             F=labels)
    bay_label=labeled(seed_xy)
    sel=(labels==bay_label) & np.isnan(z_inundate.F)
    z_inundate.F[sel]=thresh

z_inundate.write_gdal('inundation_sea_level.tif')

##
plt.figure(2).clf()
fig,axs=plt.subplots(1,2,num=2)
img=z_inundate.plot(ax=axs[0],cmap=turbo)
img.set_clim(1.5,2.5)
topo_cmap=scmap.load_gradient('oc-sst.cpt')
img2=leveed_dem.plot(ax=axs[1],cmap=topo_cmap,vmin=0.0,vmax=3.0)
plt.colorbar(img,ax=axs[0],orientation='horizontal' ,fraction=0.05,label='SL of inundation')
plt.colorbar(img2,ax=axs[1],orientation='horizontal',fraction=0.05,label='Elevation')

for ax in axs:
    ax.axis('off')
    ax.axis('tight')
    ax.axis('equal')

fig.subplots_adjust( left=0.05,right=0.95, top=0.95,bottom=0.10)
fig.tight_layout()

##

zoom=(585696.6974036136, 586143.5123015448, 4224980.868623694, 4225279.00753271)
plt.figure(3).clf()
img=leveed_dem.crop(zoom).plot(cmap=turbo,vmin=0.5,vmax=2.5)
plt.colorbar(img)


##

# Troubleshooting:
plt.figure(1).clf()
fig,axs=plt.subplots(1,3,num=1)
img=z_inundate.plot(ax=axs[0],cmap=turbo)
img.set_clim(1.5,2.5)
plt.colorbar(img)

img2=leveed_dem.plot(ax=axs[1],cmap='jet')
img2.set_clim([0,3.0])

for ax in axs:
    ax.axis('off')
    ax.axis('tight')
    ax.axis('equal')


## 
if 1:
    test_xy=plt.ginput(1)[0] # [583303, 4228972]
    test_z=z_inundate(test_xy)
    test_ij=z_inundate.xy_to_indexes(test_xy)
    seed_ij=z_inundate.xy_to_indexes(seed_xy)
    valid=z_inundate.F<=test_z

    costs = np.where(valid, 1, np.inf)
    # Not particularly fast... 10s?
    path, cost = skimage.graph.route_through_array(
        costs, start=test_ij, end=seed_ij, fully_connected=True)

    # After bringing in the lidar, the area around Sheldrake floods
    # around 2.2m.
    # Still seems kind of low.
    # The breach is now in that other slough off to the southwest.

    p=np.array(path)
    x,y=z_inundate.xy()
    pxy=np.c_[ x[p[:,1]],y[p[:,0]]]
    
    axs[0].plot(pxy[:,0], pxy[:,1], 'k-',lw=2)
    axs[1].plot(pxy[:,0], pxy[:,1], 'k-',lw=2)

    
##
if 1:
    lidar=lidar_src.extract_tile(axs[0].axis())
    img=lidar.plot(cmap='jet',vmin=0.0,vmax=2.5,ax=axs[2])

axs[2].plot(pxy[:,0], pxy[:,1], 'k-',lw=2)

## 

# Identifies a point along the N. side of sheldrake that dips down
# just enough to let water through.

# LiDaR puts this point above 2m, though.

# One option is to bump up the levee elevations to a max-filtered version of the
# lidar.



# Might also be of interest to find a way to calculate network flow,
# to preemptively find the holes.

# Could just sample points, build up a network flow.

# If it's just shortest path, then the costs will be monotonically increasing
# moving towards the target.

# What I want is more like simulating the freesurface going up a small amount
# and how that translates to velocities throughout.
# So I have a change in volume for every pixel
# I want a flow on every edge such that the divergence of the flow
# gives the change in volume
# This sounds bit like a Helmholtz projection.

# I'm looking for the curl-free / irrotational field that gives me the
# divergence I dictate.
# an irrotational field can be defined as the gradient of a vector
# potential.

# That's probably a bit much.

# But really I just want to know what's going on for certain increases
# in elevation.

leveed_dem=field.GdalGrid('dem_and_levees_and_lidar.tif')


low_thresh=2.0
high_thresh=2.2

labeled=[]

low_wet=(leveed_dem.F<=low_thresh).astype(np.int8)
low_labels,low_n_features=ndimage.label(low_wet)
low_labels_f=low_labels.astype(np.float32)
low_labels_f[low_wet==0]=np.nan

high_wet=(leveed_dem.F<=high_thresh).astype(np.int8)
high_labels,high_n_features=ndimage.label(high_wet)

low_labeled=field.SimpleGrid(extents=leveed_dem.extents,F=low_labels_f)
high_labeled=field.SimpleGrid(extents=leveed_dem.extents,F=high_labels)

low_bay_label=low_labeled(seed_xy).astype(np.int32)
high_bay_label=high_labeled(seed_xy)

## 
z_inundate=field.GdalGrid('inundation_sea_level.tif')

#   enumerate features that are bay-connected at the high-threshold but not
#   bay-connected at the low threshold (incl. dry at low threshold)
#   For each of these that is "sufficiently large", 
#      pull a tile of low-bay-connected pixels, expand by 1.
#      overlap that with the new feature to get potential connecting pixels.
#      increment those pixels with the area of the connected feature

new_wet=(high_labels==high_bay_label)&(low_labels!=low_bay_label)
new_labels,new_n_features=ndimage.label(new_wet)
new_labels_f=new_labels.astype(np.float64)
new_labels_f[~new_wet]=np.nan
new_labeled=field.SimpleGrid(extents=leveed_dem.extents,F=new_labels_f)
new_bay_label=new_labeled(seed_xy)

nrows,ncols=new_labels.shape
new_label_extents=np.zeros( (new_n_features+1,4),np.int32)

for feat,idxs in utils.enumerate_groups(new_labels.ravel()):
    if feat==0: continue
    r=idxs//ncols
    c=idxs%ncols
    new_label_extents[feat]=[ r.min(), r.max(), c.min(), c.max() ]

##

pad=3
pad_extents=new_label_extents.copy()
pad_extents[:,0]=np.maximum(0,new_label_extents[:,0]-pad)
pad_extents[:,1]=np.minimum(new_labels.shape[0],new_label_extents[:,1]+pad)
pad_extents[:,2]=np.maximum(0,new_label_extents[:,2]-pad)
pad_extents[:,3]=np.minimum(new_labels.shape[1],new_label_extents[:,3]+pad)

connections=np.zeros(leveed_dem.shape,np.float32)

thresh=10000

def calc_connection_weight(new_f):
    sel=( slice( pad_extents[new_f,0], pad_extents[new_f,1] ),
          slice( pad_extents[new_f,2], pad_extents[new_f,3] ) )
    
    mask=(new_labels[sel]==new_f)
    if mask.sum() < thresh:
        return
    
    orig_bay=low_labels[sel]==low_bay_label
    orig_bay_exp=ndimage.binary_dilation(orig_bay)

    overlap=mask & orig_bay_exp
    assert overlap.sum()>0
    
    connection_size=mask.sum()

    connections[sel][overlap]+=connection_size
    

for new_f in utils.progress(range(1,new_n_features+1)):
    if new_f%5000==0: print(f"{new_f}/{new_n_features}")
    calc_connection_weight(new_f)

connections[connections==0]=np.nan

##

fig=plt.figure(2)
fig.clf()
fig.set_size_inches([10.63,  8.78],forward=True)

fig,ax=plt.subplots(num=2)
ax.axis('off')
ax.set_position([0,0,1,1])

clip=leveed_dem.extents

leveed_dem.crop(clip).plot(ax=ax,cmap='gray',vmin=-2,vmax=3.5)

x,y=leveed_dem.xy()
rows,cols=np.nonzero(np.isfinite(connections))
ax.plot(x[cols],y[rows],'o',color='k',ms=10,zorder=1.5)
ax.plot(x[cols],y[rows],'o',color='yellow',ms=8,zorder=2)

if 1:
    categories=np.nan*new_labels_f
    categories[low_labels_f==low_bay_label]=1

    # wet areas, not connected at low thresh, connected at high thresh
    categories[(low_labels_f!=low_bay_label)&(high_labels==high_bay_label)]=3
    # categories[(low_labels!=0)&(low_labels_f!=low_bay_label)&(high_labels==high_bay_label)]=3
    
    # newly wet areas 
    # categories[(low_labels==0)&(high_labels==high_bay_label)]=2

    cat_fld=field.SimpleGrid(extents=leveed_dem.extents,F=categories)
    img=cat_fld.crop(clip).plot(ax=ax,cmap='tab10',alpha=0.3)
    img.set_clim([0,20])
ax.axis('tight')
ax.axis('equal')
ax.axis(clip)

fig.savefig('inundation-2.0_to_2.2.png',dpi=200)

##

A_thresh=thresh*leveed_dem.dx*leveed_dem.dy

print(f"Only showing overtopping that connects >{A_thresh:.2e} m^2 ({A_thresh/1e6:.2f} km^2)")
