# code to plot precipitation animation 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

file='./winds_500mb_db.nc'
ds=xr.open_dataset(file)
lon = ds.longitude
lat = ds.latitude
lon, lat = np.meshgrid(lon, lat)
U=ds.u[:,0,:,:]
V=ds.v[:,0,:,:]
data=np.sqrt(np.square(U)+np.square(V))
precip_data = data

print(lat.shape)
print(lon.shape)
print(U.shape)
print(V.shape)
print(data.shape)
nframes=data.shape[0]
## Read UAE Boundary
#fname='./shapefile/united_arab_emirates_administrative.shp'
#shape_feature = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(),
#                               linewidth = 1, facecolor = 'none',
#                               edgecolor = 'k', linestyle='-', zorder=5, alpha=0.8)

# Set up the figure and projection
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

xlabels=["20E","30E","40E","50E","60E","70E"]
ylabels=["10N","15N","20N","25N","30N","35N","40N","45"]
extent = [20.,70.,10.,45.]
xextent=np.arange(20.0,70.1,10.0)
yextent=np.arange(10.0,45.1,5.0)
levels=np.arange(1.0,30.0,2.0)


# Add colorbar
#precip_contour=ax.contourf(lon, lat, ds.air[0,:,:], transform=ccrs.PlateCarree(), cmap='jet', alpha=0.5, levels=levels)
wind_vec=ax.contourf(lon, lat, precip_data[0,:,:], transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both')
cbar = plt.colorbar(wind_vec, ax=ax, orientation='vertical', shrink=0.9, pad=0.03, extend='both')
cbar.set_label('Wind Speed (m/s)')


# Function to update the plot for each frame
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
    #ax.add_feature(shape_feature,lw=0.7,alpha=0.8)

    # Plot precipitation contours
    ax.contourf(lon, lat, data[frame,:,:], transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels,extend='both')
    ax.quiver(lon[::6, ::6], lat[::6,::6], U[frame,::6,::6], V[frame,::6,::6])

# Create the animation (frames=no.of time steps)
ani = animation.FuncAnimation(fig, update, frames=nframes, interval=500)
ani.save('ws_500.gif', writer='pillow', dpi=300)

# Display the animation
#plt.show()
