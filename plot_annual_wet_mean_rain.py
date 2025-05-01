## Plotting Mean Surface Temperature 
#-----------------------------------------------------------------------------
# importing libraries
#-----------------------------------------------------------------------------
import xarray as xr 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#from netCDF4 import Dataset as dt  
#import netCDF4

#import time
#from datetime import datetime, timedelta
#from netCDF4 import num2date, date2num
#from scipy import stats

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import matplotlib as mpl
matplotlib.use('Agg')
#-----------------------------------------------------------------------------
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

fname1 = '/data2/CMIP_ANALYSIS/Analysis/paper_AP_DI/WORLD_SHP/World_Countries.shp'
shape_feature1 = ShapelyFeature(Reader(fname1).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#------- Plotting Defination function ----------------------------------------
#-----------------------------------------------------------------------------
def plt_mapa(av,xt=0,title=' ',yt=0, yl=0, xl=0):
    if yl ==1:
        av.set_ylabel('Longitude',fontsize=16)

    if xl ==1:
        av.set_xlabel('Latitude',fontsize=16)

    if xt ==1 :
        av.set_xticks(np.arange(30, 65, 10))

    if yt ==1:
        av.set_yticks(np.arange(10, 35.5, 10))    

    av.set_xlim([30, 66])
    av.set_ylim([10, 35.5])
    av.set_title(title,fontsize=16)
    av.add_feature(cfeature.COASTLINE)
    av.add_feature(shape_feature1, facecolor='none',lw=0.8)
    lon_formatter = LongitudeFormatter(dateline_direction_label=True)
    lat_formatter = LatitudeFormatter()
    av.xaxis.set_major_formatter(lon_formatter)
    av.yaxis.set_major_formatter(lat_formatter)
    av.tick_params(which='major', width=1.00, length=5, labelsize=15)
    av.tick_params(which='minor', width=0.75, length=2.5, labelsize=12)  
    av.xaxis.set_ticks_position('bottom')
    
    return
#-----------------------------------------------------------------------------
#-- Reading DATA -------------------------------------------------------------
#-----------------------------------------------------------------------------
## MODEL Temperature
datadir_mdl='/data2/CMIP_ANALYSIS/rainfall_paper/data/'
figs_dir='/data2/CMIP_ANALYSIS/rainfall_paper/Figures/'
fname_rera_ann='era_hist_annual_mean.nc'
fname_rera_wet='era_hist_wet_mean.nc'
fname_rsrr_ann='rsrr_hist_annual_mean.nc'
fname_rsrr_wet='rsrr_hist_wet_mean.nc'
fname_mswx_ann='mswx_hist_annual_mean.nc'
fname_mswx_wet='mswx_hist_wet_mean.nc'

rera_ann_file=datadir_mdl + fname_rera_ann
rera_wet_file=datadir_mdl + fname_rera_wet
rsrr_ann_file=datadir_mdl + fname_rsrr_ann
rsrr_wet_file=datadir_mdl + fname_rsrr_wet
mswx_ann_file=datadir_mdl + fname_mswx_ann
mswx_wet_file=datadir_mdl + fname_mswx_wet

ofile=figs_dir + 'rain_annual_wet_means_era_rsrr.png'

rera_ann = xr.open_dataset(rera_ann_file)
rera_wet = xr.open_dataset(rera_wet_file)
rsrr_ann = xr.open_dataset(rsrr_ann_file)
rsrr_wet = xr.open_dataset(rsrr_wet_file)
mswx_ann = xr.open_dataset(mswx_ann_file)
mswx_wet = xr.open_dataset(mswx_wet_file)
print(rera_ann_file)
print(rera_wet_file)
print(rsrr_ann_file)
print(rsrr_wet_file)

#  Lat/ Lon Info 
llats_rera = rera_ann.variables['latitude']
llons_rera = rera_ann.variables['longitude']
llats_rsrr = rsrr_ann.variables['XLAT'][:,0]
llons_rsrr = rsrr_ann.variables['XLONG'][0,:]
llats_mswx = mswx_ann.variables['lat']
llons_mswx = mswx_ann.variables['lon']

rera_ann.close()
rsrr_ann.close()
mswx_ann.close()
rera_wet.close()
rsrr_wet.close()
mswx_wet.close()

rera_ann_var = rera_ann['tp'] * 365.0
rsrr_ann_var = rsrr_ann['rain'] * 365.0
rmsw_ann_var = mswx_ann['precipitation'] * 365.0
rera_wet_var = rera_wet['tp'] * 183.0
rsrr_wet_var = rsrr_wet['rain'] * 183.0
rmsw_wet_var = mswx_wet['precipitation'] * 183.0
###############################
###############################
####### ERA Plotting  #######
###############################
x2, y2 = np.meshgrid(llons_rera,llats_rera)
x1, y1 = np.meshgrid(llons_rsrr,llats_rsrr)
x3, y3 = np.meshgrid(llons_mswx,llats_mswx)

origin = 'lower'
us = np.arange(10,301.,20)

fig, axes = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},nrows=2,ncols=3, figsize=(13.4,7.2))

print(rsrr_ann_var.shape)
## ---- <<< RSRR-Annual >>>> -------------------------------------------------------------------
axes[0][0].contourf(x1,y1,rsrr_ann_var[0,:,:],
              levels=us,cmap='Blues',extend="both",origin=origin,transform=ccrs.PlateCarree()) 
plt_mapa(axes[0][0],title=' RSSS-Annual' ,yt=1, yl=1)

## ---- <<< RSRR-Wet  >>>> -------------------------------------------------------------------
axes[1][0].contourf(x1,y1,rsrr_wet_var[0,:,:],
              levels=us,cmap='Blues',extend="both",origin=origin,transform=ccrs.PlateCarree())
plt_mapa(axes[1][0],title=' RSRR-Wet' ,yt=1, yl=1,xt=1,xl=1)

## ---- <<< ERA-Annual >>>> -------------------------------------------------------------------
axes[0][1].contourf(x2,y2,rera_ann_var[0,:,:],
              levels=us,cmap='Blues',extend="both",origin=origin,transform=ccrs.PlateCarree()) 
plt_mapa(axes[0][1],title=' ERA-Annual' ,yt=0, yl=0, xt=0, xl=0)

## ---- <<< ERA-Wet >>>> -------------------------------------------------------------------
axes[1][1].contourf(x2,y2,rera_wet_var[0,:,:],
              levels=us,cmap='Blues',extend="both",origin=origin,transform=ccrs.PlateCarree())
plt_mapa(axes[1][1],title=' ERA-Wet' ,yt=0, yl=0,xt=1, xl=1)
#---------------------------------------------------------------------------------------------------
## ---- <<< MSWX-Annual >>>> -------------------------------------------------------------------
axes[0][2].contourf(x3,y3,rmsw_ann_var[0,:,:],
              levels=us,cmap='Blues',extend="both",origin=origin,transform=ccrs.PlateCarree()) 
plt_mapa(axes[0][2],title=' MSWX-Annual' ,yt=0, yl=0, xt=0, xl=0)

## ---- <<< MSWX-Wet >>>> -------------------------------------------------------------------
im=axes[1][2].contourf(x3,y3,rmsw_wet_var[0,:,:],
              levels=us,cmap='Blues',extend="both",origin=origin,transform=ccrs.PlateCarree())
plt_mapa(axes[1][2],title=' MSWX-Wet' ,yt=0, yl=0,xt=1, xl=1)
#---------------------------------------------------------------------------------------------------


## Colorbar
ust = np.arange(10,301.,20)
cb1=fig.colorbar(im, ax=axes.ravel(),aspect=30,pad=0.02,shrink=0.96,ticks=ust)
cb1.ax.tick_params(labelsize=18)

## Main-Title
fig.suptitle('Mean Precipitation (mm)', y=0.94, fontsize=20)

## Save Figure-1
plt.savefig(ofile,format='png',dpi=300)
quit()
