
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import  cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature


file ='/home/dasarih/Desktop/karthik/RSRR_SA_daily_ts.nc'
ds=xr.open_dataset(file)
lat = ds.latitude
lon = ds.longitude
lon, lat = np.meshgrid(lon, lat)
tm  = ds.rain
#lons, lats = np.meshgrid(lons, lats)

print(lat.shape)
print(lon.shape)
print(rain.shape)
nframes=t2m.shape[0]

fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection=ccrs.PlateCarree())

xlabels=["35E","40E","45E","55E","60E"]
ylabels=["15N","20N","25N","30N"]
extent = [35.,60.,15.,30.]
xextent=np.arange(35.0,60.1,1.0)
yextent=np.arange(15.0,30.1,1.0)
levels=np.arange(0.0,40.0,2.0)


rain_vec=ax.contourf(lat, lon,rain[0,:,:], transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both')
cbar = plt.colorbar(rain_vec, ax=ax, orientation='vertical', shrink=0.9, pad=0.03, extend='both')
cbar.set_label('Rainfall')


def update(frame):
    ax.clear()
    ax.coastlines()
    ax.set_extent(extent, ccrs.PlateCarree())

    ax.coastlines(linewidths=0.5)
    ax.set_yticks(yextent)
    ax.set_xticks(xextent)
    ax.set_xticklabels(xlabels,weight='bold')
    ax.set_yticklabels(ylabels,weight='bold')
    ax.set_title(str(ds.coords['time'].values[frame])[:-10])

    ax.contourf(lon, lat, rain[frame,:,:], transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels,extend='both')
    #ax.quiver(lon[::6, ::6], lat[::6,::6], U[frame,::6,::6], V[frame,::6,::6])

ani = animation.FuncAnimation(fig, update, frames=nframes, interval=500)
ani.save('rain_test.gif', writer='pillow', dpi=300)

