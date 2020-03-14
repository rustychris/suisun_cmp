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
import local_config
import xarray as xr
import six
six.moves.reload_module(dfm)

local_config.install()

# the original
model_orig=dfm.DFlowModel.load(os.path.join(local_config.run_root_dir,
                                            '2018_Suisun_R05_res00',
                                            'FlowFM.mdu'))

model=model_orig.create_restart('FlowFM_R001')
model.run_stop = model.run_start+np.timedelta64(6,'h')

# Copy restart files to the new output directory and overwrite
# some variables

self=model

def modify_ic(ds,**kw):
    ds['fm'].values[0,:]=np.random.random(ds.dims['nFlowElem'])
    return ds

self.modify_restart_data(modify_ic)

##     
if 1:
    model.mdu.write() 
    model.partition()
    os.environ['LD_LIBRARY_PATH']=os.path.join(local_config.dfm_root_dir,'lib')
    model.run_simulation(extra_args=['--processlibrary',os.path.join(local_config.dfm_root_dir,'share/delft3d/proc_def.def')])
