"""
Try a spinup, update the variables, and continue.
"""
# copy the original global MDU.
# TStop was 43200.  That now becomes TStart, and TStop set to 86400
# This next step may not be needed in the case of using a proper restart file.
# I think it's just for using a map file
# ['restart','RestartDateTime'] = 201804101200
# Note that processor ID is removed in this name, and will be re-inserted by partition
# ['restart','RestartFile'] = DFM_OUTPUT_FlowFM/FlowFM_20180410_120000_rst.nc

# Need to make sure the simulation name reflects the restart, so the output directories
# are kept separate

## 
import numpy as np
import os
import glob
import shutil
import stompy.model.delft.dflow_model as dfm
from stompy.grid import unstructured_grid
from stompy.spatial import wkb2shp
import local_config
import xarray as xr
from matplotlib.path import Path
import six
six.moves.reload_module(dfm)

local_config.install()

# the original
model_orig=dfm.DFlowModel.load(os.path.join(local_config.run_root_dir,
                                            '2018_Suisun_R05_res00',
                                            'FlowFM.mdu'))

model=model_orig.create_restart('FlowFM_R001.mdu')
model.run_stop = model.run_start+np.timedelta64(30,'D')

# Copy restart files to the new output directory and overwrite
# some variables

polys=wkb2shp.shp2geom('../../field/hypsometry/slough_polygons-v00.shp')

def modify_ic(ds,proc,model,**kw):
    # NOTE! subdomain grid read from the partitioned _net files does *NOT*
    # line up with the flow elements in the restart file. There is some
    # reordering that occurs.  It appears to be deterministic, and is
    # consistent between map outputs and restart files.  So either use
    # the coordinate information in the restart file, or load a map output
    # from a previous run.
    # NOPE: model.subdomain_grid(proc)
    cc=np.c_[ ds.FlowElem_xzw.values,
              ds.FlowElem_yzw.values ]

    for idx in range(len(polys)):
        scal=polys['code'][idx]
        geom=polys['geom'][idx]
        #sel=g.select_cells_intersecting(geom)
        p=Path(np.array(geom.exterior),closed=True)
        sel=p.contains_points(cc)
        ds[scal].values[0,sel]=1.0

    return ds

model.modify_restart_data(modify_ic)

##     
if 1:
    model.update_config()
    model.mdu.write() 
    model.partition() # but I want to partition only the mdu!
    os.environ['LD_LIBRARY_PATH']=os.path.join(local_config.dfm_root_dir,'lib')
    model.run_simulation(extra_args=['--processlibrary',os.path.join(local_config.dfm_root_dir,'share/delft3d/proc_def.def')])


# HERE
# Seems that the IC it's trying to set is not lining up correctly.
# The geographical extent is close.
# next step is to look at the limited geometry available in the rst
# file, see if it is consistent with the _net files or not.
