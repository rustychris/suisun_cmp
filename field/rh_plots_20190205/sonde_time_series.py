import pandas as pd
import os
import numpy as np
from stompy import utils
import matplotlib.pyplot as plt
import xarray as xr
import stompy.plot.cmap as scmap

##
cmap=scmap.load_gradient('hot_desaturated.cpt')

#%% 
##
sonde_dir="../sonde/CMP_hydro/sondes/Data_output"
sonde_fns=[
    "CMP_Data_FirstMallard_Sonde_2017_2018_Tidied.csv",
    "CMP_Data_Hill_Sonde_2017_2018_Tidied.csv",
    "CMP_Data_Peytonia_Sonde_2017_2018_Tidied.csv",
    "CMP_Data_Sheldrake_Sonde_2017_2018_Tidied.csv"
]

##
#%%
import six
six.moves.reload_module(utils)

sondes=[]
for sonde_fn in sonde_fns:
    sonde_df=pd.read_csv(os.path.join(sonde_dir,sonde_fn),
                         parse_dates=['datetime','datetime_DLS'])
    sonde_ds=xr.Dataset.from_dataframe(sonde_df)
    sonde_ds=sonde_ds.rename(index='time')
    sonde_ds['time']=sonde_ds['datetime'] # more standard in CF
    sonde_ds['fn']=(),sonde_fn
    sonde_ds['site']=(),sonde_ds['SiteName'][0]
    
    th=utils.hour_tide(utils.to_dnum(sonde_ds.time.values),
                       h=sonde_ds['Depth'])
    sonde_ds['tide_h']=('time',),th
                       
    # DEV - expand to 24 hour tide
    incrs=np.zeros(len(th),np.float64)
    wraps=np.nonzero(np.diff(th)<0)[0]
    th24=th.copy()
    h=sonde_ds['Depth']
    lwraps=[None] + list(wraps) + [None]
    slices=[slice(a,b) for a,b in zip(lwraps[:-1],lwraps[1:])]

    for i in range(0,len(slices)-1,2):
        winA=slices[i]
        winB=slices[i+1]
        if h[winA].max() > h[winB].max():
            th24[winA]+=12
        else:
            th24[winB]+=12
    
    sonde_ds['tide24_h']=('time',),th24

    sondes.append(sonde_ds)
##
#%%

def figure_tidal_phase(field,sel_name,selector):
    plt.figure(1).clf()
    periods=[
        (np.datetime64("2017-04-01"), np.datetime64("2017-11-01")),
        (np.datetime64("2018-04-01"), np.datetime64("2018-11-01"))
    ]

    fig,axs=plt.subplots(len(sondes),len(periods),num=1,sharex='col')
    fig.set_size_inches([12.75,9.25],forward=True)

    limits={'DO_mgl':[2,10],
            'Turb':[0,200],
            'fDOMQSU':[0,200],
            'ChlorophyllugL':[0,50],
            'pH':[7,9],
            'Sal':[0,8],
            'Temp':[14,25],
            'tide_h':[0,12],
            'tide24_h':[0,24]}

    for ax_periods,sonde in zip(axs,sondes):
        for ax,period in zip(ax_periods,periods):
            sel=(sonde.time>=period[0])&(sonde.time<=period[1])
            #sel=sel&(sonde.tide24_h>=12)
            sel=sel&selector(sonde)
            sonde_sel=sonde.isel(time=sel)
            scat=ax.scatter( sonde_sel.time.values, sonde_sel.Depth.values,10, sonde_sel[field].values,
                             cmap=cmap)
            if field in limits:
                scat.set_clim( limits[field] )
        ax_left=ax_periods[0]
        ax_left.text(0.02,0.96,sonde.site.item(),
                     transform=ax_left.transAxes,va='top')
        ax_left.set_ylabel('Depth')
        plt.colorbar(scat,ax=ax_periods[-1],
                     label=field)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

for field in ['DO_mgl','Turb','Temp','Sal','pH','ChlorophyllugL','fDOMQSU']:
    for sel_name,selector in [('12',lambda ds: ds.tide24_h>=12),
                              ('0', lambda ds: ds.tide24_h<12)]:
        fig=figure_tidal_phase(field,sel_name,selector)
        break
        fig.savefig('%s_tidal_patterns%s.png'%(field,sel_name))
#%%
        
fig=figure_tidal_phase("fDOMQSU",'12',lambda ds: ds.tide24_h>=12)
