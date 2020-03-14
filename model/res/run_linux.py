"""
Try a spinup, update the variables, and continue.
"""
import numpy as np
import os
import shutil
import stompy.model.delft.dflow_model as dfm
import local_config

local_config.install()

model=dfm.DFlowModel.load('.')

model.num_procs=16
# on cws-linuxmodeling, 16 cores mpi, this 173 day run is expected
# to take ~5h for this period:
# '2018-04-10T00:00:00.000000'
# '2018-09-30T00:00:00.000000'

# For 10 days it's about 15 minutes.

# model class is not clean enough to handle a true load/write roundtrip,
# so reach in and be a bit manual
model.mdu.set_time_range(start=model.run_start,
                         #stop=np.datetime64('2018-04-20 00:00'))
                         #stop=np.datetime64('2018-09-30 00:00'))
                         stop=model.run_start + np.timedelta64(5*24,'h'))
                         
# Install the DWAQ parameters
model.mdu['processes','SubstanceFile']='sources.sub'
model.mdu['processes','DtProcesses']=60
model.mdu['processes','ProcessFluxIntegration'] = 1
model.mdu['processes','DtMassBalance'] = 86400.0

with open(model.mdu.filepath(('processes','SubstanceFile')),'wt') as fp:
    fp.write("""
substance 'fm' active
  description 'FM tracer'
  concentration-unit 'g/m3'
  waste-load-unit 'g'
end-substance
substance 'sd' active
  description 'SD tracer'
  concentration-unit 'g/m3'
  waste-load-unit 'g'
end-substance
substance 'hl' active
  description 'HL tracer'
  concentration-unit 'g/m3'
  waste-load-unit 'g'
end-substance
substance 'pt' active
  description 'PT tracer'
  concentration-unit 'g/m3'
  waste-load-unit 'g'
end-substance

active-processes
   name 'DYNDEPTH' 'Depth calculations'
   name 'TOTDEPTH' 'total depth calc'
end-active-processes
""")

# And doctor up the external forcing file
sentinel=b"# ---- auto generated below ----"
with open('FlowFM_bnd_old.ext','rb+') as fp:
    contents=fp.read()
    try:
        fp.seek(contents.index(sentinel))
    except ValueError:
        fp.seek(len(contents))
    fp.write(sentinel+b"\n")

    # At both boundaries specify 0 concentration for all 3 tracers
    for pli_src in ['Boundary01.pli','Boundary02.pli']:
        for tracer in ['fm','sd','hl','pt']:
            pli_out=pli_src.replace('.pli',tracer+'.pli')
            tim_out=pli_out.replace('.pli','_0001.tim')
            shutil.copyfile(pli_src,pli_out)
            fp.write(f"""
QUANTITY=tracerbnd{tracer}
FILENAME={pli_out}
FILETYPE=9
METHOD=3
OPERAND=O
            """.encode() )
            with open(tim_out,'wt') as tim_fp:
                tim_fp.write("""\
-1.0e+06 0
1.0e+06 0
""")

# Request restart files
model.mdu['output','RstInterval']=7200 # TODO: change to longer

##     
if 1:
    model.mdu.write()

    model.partition()
    # this to the environment
    os.environ['LD_LIBRARY_PATH']=os.path.join(local_config.dfm_root_dir,'lib')
    # This on to the command line
    # "--processlibrary $DFMROOT/share/delft3d/proc_def.def" 

    model.run_simulation(extra_args=['--processlibrary',os.path.join(local_config.dfm_root_dir,'share/delft3d/proc_def.def')])
