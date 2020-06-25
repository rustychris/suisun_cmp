"""
Look at possibility of estimating dispersion or bulk mixing rates
from observations alone or observed scalars and modeled flow.
"""
from scipy.optimize import fmin
import six
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os
from stompy import utils, filters
from stompy.plot import plot_utils

import stompy.plot.cmap as scmap
turbo=scmap.load_gradient('turbo.cpt')
##

field_data="../field/from_sophie"
info=pd.read_csv(os.path.join(field_data,'Flux_4stations.csv'))

##

stn_info = info[ (info.Station=='FM') & (info.Year==2018) ].iloc[0,:]
#flow_src='model_clip_to_obs'
flow_src='obs'

def flow_obs():
    flow_obs_fn=os.path.join(field_data,stn_info['Flow_obs'].replace('\\','/'))
    return pd.read_csv(flow_obs_fn,parse_dates=['Time'],
                       infer_datetime_format=True).rename({'Time':'time'},axis=1)
def flow_model():
    # Need to get time,Flow,Stage
    df=pd.read_csv(os.path.join(field_data,'Processed_DF/Modeled_FLOW_STAGE.csv'),
                   parse_dates=['datetime']).rename({'datetime':'time',
                                                         'flow':'Flow',
                                                         'stage':'Stage'},axis=1)
    sel=(df.Site==stn_info.Station)&(df.Year==stn_info.Year)

    if stn_info.Year==2018:
        print("Shifting by -1h")
        df['time'] = df['time'].values - np.timedelta64(3600,'s')
    # Seems that the model is generally 6 minutes off.
    print("Shifting for model error by 6 minutes")
    # Without this, SD 2017, e.g., has lag_s of -360 and -354.
    df['time'] = df['time'].values + np.timedelta64(360,'s')

    return df[sel].copy()

if flow_src=='obs':
    flow_stage=flow_obs()
elif flow_src=='model':
    flow_stage=flow_model()
elif flow_src=='model_clip_to_obs':
    obs=flow_obs()
    flow_stage=flow_model()
    sel=((flow_stage.time.values>=obs.time.values[0])
         & (flow_stage.time.values<=obs.time.values[-1]))
    flow_stage=flow_stage.iloc[sel,:].copy()

##
import xarray as xr
from stompy.model import data_comparison
six.moves.reload_module(data_comparison)
flow_stage_model=xr.Dataset.from_dataframe(flow_model().set_index('time'))
flow_stage_obs=  xr.Dataset.from_dataframe(flow_obs().set_index('time'))

# getting 3290s lag.  sounds like a time zone
flow_metrics=data_comparison.calc_metrics(flow_stage_model.Flow,
                                          flow_stage_obs.Flow,
                                          combine=True)

# and 3234s here.  
stage_metrics=data_comparison.calc_metrics(flow_stage_model.Stage,
                                          flow_stage_obs.Stage,
                                          combine=True)
## 
# mini-validation
fig=plt.figure(10)
fig.clf()
gs=gridspec.GridSpec(2,5)
ax=fig.add_subplot( gs[0,:-1])
ax_eta=fig.add_subplot(gs[1,:-1],sharex=ax)
ax_txt=fig.add_subplot(gs[0,-1])
ax_eta_txt=fig.add_subplot(gs[1,-1])
ax_txt.axis('off')
ax_eta_txt.axis('off')

ax.plot(flow_stage_obs.time,
        flow_stage_obs.Flow,
        'k-',label='obs')
ax.plot(flow_stage_model.time,
        flow_stage_model.Flow,
        'g-',label='mod')
ax.set_ylabel('Flow (m$^3$/s)')
ax.legend()

# not perfect, since it doesn't match times
eta_offset=flow_stage_obs.Stage.mean() - flow_stage_model.Stage.mean()

ax_eta.plot(flow_stage_obs.time,
            flow_stage_obs.Stage - eta_offset,
            'k-',label='obs')
ax_eta.plot(flow_stage_model.time,
            flow_stage_model.Stage,
            'g-',label='mod')
ax_eta.set_ylabel('Stage (m, relative)')
ax_eta.legend()

for axt,metrics in zip( [ax_txt,ax_eta_txt], [flow_metrics,stage_metrics]):
    lines="\n".join( [ "%s:%.3g"%(k,metrics[k]) for k in metrics if k not in ['lag']] )
    axt.text(0.01,0.99,lines,va='top')

fig.text(0.5,0.95, "%s: %s"%( stn_info.Station, stn_info.Year) )

##

# TODO: Better estimation here
# Define a cross-section to scale this to a regular dispersion coefficient
if stn_info.Station=='FM':
    # Rough estimate from NWIS: tidal range 24ft to 28ft.
    # Flood velocity at 26 feet: -0.70  with Q=-306
    # Ebb velocity 26 feet:0.69 Q=301
    stn_info['Ayz']= (303/0.70) * 0.3048**2
elif stn_info.Station=='SD':
    # TODO: get real numbers here.
    # This is eyeballed from DEM in QGIS profile tool
    stn_info['Ayz']= 25*1.2
    
    
##

sonde_fn=os.path.join(field_data,'Processed_DF','MasterDF_sonde2.csv')

sonde=pd.read_csv(sonde_fn,parse_dates=['Datetime_UTC'],infer_datetime_format=True).rename({'Datetime_UTC':'time'},axis=1)

sonde=sonde[ (sonde.Site==stn_info.Station) & (sonde.Year==stn_info.Year) ]

##
# Interpolate flow and stage onto the sonde timeseries

flow_dnum=utils.to_dnum(flow_stage.time.values)
sonde_dnum=utils.to_dnum(sonde.time.values)

flow_at_sonde =utils.interp_near(sonde_dnum,flow_dnum,flow_stage.Flow.values,max_dx=0.5/24)
stage_at_sonde=utils.interp_near(sonde_dnum,flow_dnum,flow_stage.Stage.values,max_dx=0.5/24)

sonde['flow_obs']=flow_at_sonde
sonde['stage_obs']=stage_at_sonde

sonde_dt_h=0.25

def lowpass(s,t=None):
    # return filters.lowpass_fir(s, int(40/sonde_dt_h))
    return filters.lowpass_godin(s,mean_dt_h=sonde_dt_h,ends='nan')
def highpass(s,t=None):
    return s-lowpass(s,t)

sonde['Sal_lp']=lowpass(sonde['Sal'].values, sonde.time)
sonde['Sal_hp'] = sonde['Sal'] - sonde['Sal_lp']
##

plt.figure(2).clf()

params=['flow_obs','stage_obs','Sal']
fig,axs=plt.subplots(len(params),1,sharex=True,num=2)

for param,ax in zip(params,axs):
    ax.plot(sonde.time, sonde[param])
    ax.set_ylabel(param)


## 
plt.figure(1).clf()

fig,ax=plt.subplots(num=1)

# Phasing of salt, flow and stage
scat=ax.scatter( sonde.flow_obs,
                 sonde.stage_obs,
                 10,
                 sonde.Sal_hp,cmap=turbo)
ax.set_xlabel('Flow')
ax.set_ylabel('Stage')
plt.colorbar(scat,label='Salt anomaly (ppt)')
fig.tight_layout()

##

# This next part hasn't been totally nailed down, or shown to be
# robust.  Just trying to poke the data a bit.

sonde_sel=sonde[ np.isfinite(sonde.flow_obs.values) ].copy()

max_dt_sel_s=np.diff(sonde_sel.time).max() / np.timedelta64(1,'s')
assert max_dt_sel_s<=1800.0 # make sure we're not integrating over big missing chunks.

t_s=(sonde_sel.time.values - sonde_sel.time.values[0])/np.timedelta64(1,'s')

Qmid=0.5*(sonde_sel.flow_obs.values[:-1] + sonde_sel.flow_obs.values[1:])
dt_s=t_s[1:] - t_s[:-1]

Vrel=np.r_[ 0, np.cumsum(-Qmid*dt_s)]

sonde_sel['Vrel']=Vrel
sonde_sel['Vrel_lp']=lowpass(sonde_sel['Vrel'].values, sonde_sel.time.values)
sonde_sel['Vrel_hp']=sonde_sel.Vrel - sonde_sel.Vrel_lp

##

# Excursion plot using volume
fig=plt.figure(3)
fig.clf()
scat=plt.scatter( sonde_sel.time, sonde_sel['Vrel_hp'],20,sonde_sel['Sal_hp'], cmap=turbo)
plt.colorbar(scat,label='Salinity')
plt.ylabel('Volumetric excursion')
fig.tight_layout()

##

fig=plt.figure(4)
fig.clf()
scat=plt.scatter( sonde_sel['Vrel_hp'],sonde_sel['Sal_hp'], 10,
                  (sonde_sel.time-sonde_sel.time.values[0])/np.timedelta64(1,'D'),
                  cmap=turbo)
plt.xlabel('Volumetric excursion')
plt.ylabel('Salinity')
plt.colorbar(scat,label='Time (days)')
fig.tight_layout()

##

# Compare s(V(t)) for flood to subsequent ebb
plt.figure(5).clf()
fig,ax=plt.subplots(num=5)

ebbing=sonde_sel['flow_obs'].values>0
sonde_sel['phase']=np.where(ebbing,'ebb','flood')

ebb_mask=np.where( ( ebbing | np.roll(ebbing,-1)),
                   1, np.nan)
fld_mask=np.where( ( (~ebbing) | np.roll(~ebbing,-1)),
                   1,np.nan)


ax.plot( sonde_sel['Vrel_hp'].values,
         ebb_mask*sonde_sel['Sal_hp'].values,
         color='tab:blue',label='Ebb')
ax.plot( sonde_sel['Vrel_hp'].values,
          fld_mask*sonde_sel['Sal_hp'].values,
          color='tab:red',label='Flood')
ax.legend()
ax.set_ylabel('Salinity anomaly')
ax.set_xlabel('Volume anomaly')
ax.text(0.02,0.02,'Low water',transform=ax.transAxes,ha='left')
ax.text(0.98,0.02,'High water',transform=ax.transAxes,ha='right')

##

# Bins salt anomaly by volume excursion, plot for flood and ebb.
valid=np.isfinite(sonde_sel['Vrel_hp'].values)
Vbins=np.percentile( sonde_sel['Vrel_hp'].values[valid], np.linspace(0,100,20))

Vbin=np.searchsorted( Vbins, sonde_sel['Vrel_hp'].values[valid] ).clip(0,len(Vbins)-2)
Vbin_mid=0.5*(Vbins[:-1]+Vbins[1:])

import seaborn as sns
fig=plt.figure(6)
fig.clf()
sns.boxplot( x=Vbin_mid[Vbin],
             y=sonde_sel['Sal_hp'].values[valid],
             hue=sonde_sel['phase'].values[valid] )

ax=fig.axes[0]
ax.text(0.02,0.02,'Low water',transform=ax.transAxes,ha='left')
ax.text(0.98,0.02,'High water',transform=ax.transAxes,ha='right')
ax.set_ylabel('Salinity anomaly')
ax.set_xlabel('Vol. anomaly (10$^3$ m$^3$)')

ax.set_xticklabels( ["%.0f"%(v/1e3) for v in Vbin_mid])

##

# Previous figure is harder to interpret when the gradient reverses.
# Here, instead plot the square of the anomaly.  A step closer to the
# contribution to variance.
fig=plt.figure(7)
fig.clf()
sns.boxplot( x=Vbin_mid[Vbin],
             y=(sonde_sel['Sal_hp'].values[valid])**2,
             hue=sonde_sel['phase'].values[valid] )

ax=fig.axes[0]
ax.text(0.02,0.02,'Low water',transform=ax.transAxes,ha='left')
ax.text(0.98,0.02,'High water',transform=ax.transAxes,ha='right')
ax.set_ylabel('Salinity anomaly$^2$')
ax.set_xlabel('Vol. anomaly (10$^3$ m$^3$)')

ax.set_xticklabels( ["%.0f"%(v/1e3) for v in Vbin_mid])

print(sonde_sel.groupby('phase')['Sal_hp'].var())

##

# Potential ways of estimating either residence time or dispersion coefficient

# A: tidal average salinity and estimated ds/dx

# Or even simpler:
#   Relate flux to apparent gradient.
# With the current setup, focus right around V=0.

sonde_sel['Q_lp']=lowpass(sonde_sel['flow_obs'].values, sonde_sel.time.values)
sonde_sel['Q_hp']=sonde_sel['flow_obs'] - sonde_sel['Q_lp']

sonde_sel['J_salt']=lowpass( sonde_sel['flow_obs']*sonde_sel['Sal'],sonde_sel.time )
sonde_sel['J_salt_adv']=lowpass( sonde_sel['Q_lp']*sonde_sel['Sal_lp'],sonde_sel.time )
sonde_sel['J_salt_disp']=lowpass( sonde_sel['Q_hp']*sonde_sel['Sal_hp'],sonde_sel.time )

# Okay - now an estimate of ds/dV
# Start with fitting a single slope to just the excursion near zero
Vscale=np.std(sonde_sel['Vrel_hp'])
Vlow,Vhigh=[-0.1*Vscale,0.1*Vscale]
sel=(sonde_sel['Vrel_hp']>=Vlow)&(sonde_sel['Vrel_hp']<Vhigh)

# m ~ 3e-6.  Agrees with figure 4.
m,b=np.polyfit( sonde_sel['Vrel_hp'].values[sel],
                sonde_sel['Sal_hp'].values[sel],1)


# running lowpass of ds/dv
# Should probably verify the math here.
# Get a similar number in the mean, though this is using all of the data points,
# while the above is just points near the middle
dSdV = lowpass( sonde_sel['Sal_hp']*sonde_sel['Vrel_hp'] ) / lowpass( sonde_sel['Vrel_hp']**2 )
sonde_sel['dSdV']=dSdV

plt.figure(8).clf()
fig,(ax,ax_grad)=plt.subplots(2,1,num=8,sharex=True)

ax.plot( sonde_sel.time, sonde_sel.J_salt, label='net flux')
ax.plot( sonde_sel.time, sonde_sel.J_salt_adv, label='adv flux')
ax.plot( sonde_sel.time, sonde_sel.J_salt_disp, label='disp flux')
ax.axhline(0,color='0.6',lw=0.5,zorder=-2)
ax.legend()
ax_grad.plot( sonde_sel.time, sonde_sel.dSdV, label='dSdV')
ax_grad.legend()

##

plt.figure(9).clf()

springiness=lowpass( highpass(sonde_sel.stage_obs.values)**2)

    
Ayz=stn_info['Ayz']

fig,ax_diff=plt.subplots(num=9)
xscale=1e3

dSdx=sonde_sel.dSdV.values*Ayz

#ax_diff.plot( xscale*sonde_sel.dSdV*Ayz, sonde_sel.J_salt_disp, 'k-',lw=0.5,label='Dispersive')
scat=ax_diff.scatter( xscale*dSdx, sonde_sel.J_salt_disp,
                      20, springiness, cmap='inferno_r')
plt.colorbar(scat,label='Springiness')
ax_diff.set_xlabel('Apparent dS/dx (ppt/km)')
ax_diff.set_ylabel('J_salt')
valid=np.isfinite(dSdx)
K,offset=np.polyfit( -dSdx[valid],
                     sonde_sel.J_salt_disp.values[valid]/Ayz, 1)
# fit again with the intercept forced to zero
Kslope_lsq=fmin( lambda K: np.mean( (K*(-dSdx[valid]) - (sonde_sel.J_salt_disp.values[valid]/Ayz))**2 ),
                 [10] )

# FM 2017: Get a tidal average dispersion coefficient of 41 m2/s
# seems reasonable.

dSs=np.array([min(0,np.nanmin(dSdx)),
              max(0,np.nanmax(dSdx))])

l=ax_diff.plot(xscale*dSs, Ayz*( offset - K*dSs),'g-')
t=plot_utils.annotate_line(l[0],'K = %.1f m$^2$/s'%K,norm_position=0.5,
                           fontsize=13,
                           buff=dict(linewidth=3, foreground="w"))

l2=ax_diff.plot(xscale*dSs, Ayz*( - Kslope_lsq*dSs),'g--')
t2=plot_utils.annotate_line(l2[0],'K = %.1f m$^2$/s'%Kslope_lsq,norm_position=0.2,
                            fontsize=13,
                            buff=dict(linewidth=3, foreground="w"))

##

# FM, 2017
# ==> K ~ -44.7 + 295*springiness

# FM, 2018: 
# ==> K ~ -67.5 + 391*springiness

# SD, 2017:
# ==> K ~ -12.9 + 116*springiness

plt.figure(11).clf()

springiness=lowpass( highpass(sonde_sel.stage_obs.values)**2)

Ayz=stn_info['Ayz']


fig,ax_diff=plt.subplots(num=11)
xscale=1e3

dSdx=sonde_sel.dSdV.values*Ayz

valid=np.isfinite(dSdx)

# Fit a K ~ springiness model
K0,c_spring=fmin( lambda params: np.mean( ((params[0]+params[1]*springiness[valid])*(-dSdx[valid])
                                           - (sonde_sel.J_salt_disp.values[valid]/Ayz))**2 ),
                  [10,1] )

# (K0+c_spring*springiness) * dSdx ~ J_disp/Ayz
# What can I plot to see how well this collapses?
# dSdx
dSdx_adj=(dSdx*(K0+c_spring*springiness))
scat=ax_diff.scatter( -dSdx_adj,
                      sonde_sel.J_salt_disp.values/Ayz,
                      20,
                      springiness, cmap='inferno_r')
plt.colorbar(scat,label=r'var($\eta$)')
ax_diff.set_xlabel('Spring-adjusted dS/dx')
ax_diff.set_ylabel('Salt flux')


dSs=np.array([np.nanmin(-dSdx_adj),
              np.nanmax(-dSdx_adj)])
scale=dSs[1]-dSs[0]
dSs[0]-=0.2*scale
dSs[1]+=0.2*scale

l=ax_diff.plot(dSs,dSs,'b-')
t=plot_utils.annotate_line(l[0],r'K = %.1f + %.1f*var($\eta$)'%(K0,c_spring),
                           norm_position=0.5,
                           fontsize=13,
                           buff=dict(linewidth=3, foreground="w"))


## 
# With the time-shifted model results, get 32 m2/s compared to 24 or so with
# observed.  But spring tides is still messed -- in both cases spring tides
# show positive fluxes.
