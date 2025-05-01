import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import dask.array as da

# Load NetCDF file using xarray with Dask
file = '/home/dasarih/Desktop/karthik/RSRR_SA_daily_ts.nc'
ds = xr.open_dataset(file, chunks={'XTIME': 10})  # Chunk size is set to 10 time steps

lat = ds.
LAT = 
lon = ds.XLONG
lon, lat = np.meshgrid(lon, lat)
rain = ds.rain.data  # Get Dask array for rain data

print(lat.shape)
print(lon.shape)
print(rain.shape)

nframes = rain.shape[0]

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

xlabels = ["35E", "40E", "45E", "55E", "60E"]
ylabels = ["15N", "20N", "25N", "30N"]
extent = [35., 60., 15., 30.]
xextent = np.arange(35.0, 60.1, 1.0)
yextent = np.arange(15.0, 30.1, 1.0)
levels = np.arange(0.0, 40.0, 2.0)

# Function to compute rain data for each chunk
def process_chunk(chunk_index):
    # Determine the range of time steps to read
    start = max(0, chunk_index - 5)  # Read 5 time steps before the chunk
    end = min(nframes, chunk_index + 6)  # Read 6 time steps after the chunk
    
    # Load rain data for the specified range
    chunk_data = rain[start:end, :, :]
    
    # Process the chunk data (you can modify this part based on your computation)
    # For demonstration, let's compute the sum of rain data for each chunk
    return chunk_data.sum(axis=0)

# Function to update the plot for each frame
def update(frame):
    
     if frame % 100 == 0:
        print("Progress: {:.2f}%".format(frame / nframes * 100))

     ax.clear()
     ax.coastlines()
     ax.set_extent(extent, ccrs.PlateCarree())
     ax.coastlines(linewidths=0.5)
     ax.set_yticks(yextent)
     ax.set_xticks(xextent)
     ax.set_xticklabels(xlabels, weight='bold')
     ax.set_yticklabels(ylabels, weight='bold')
     ax.set_title(str(ds.coords['time'].values[frame])[:-10])

    # Compute the rain data for the current chunk
     chunk_index = frame
     rain_chunk = process_chunk(chunk_index)
    
    # Plot the rain data for the current chunk
     ax.contourf(lon, lat, rain_chunk, transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both')

# Call the update function to create animation
ani = animation.FuncAnimation(fig, update, frames=nframes, interval=100)
plt.show()
quit()




























































import numpy as np
import dask.array as da
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
import seaborn as sns
import multiprocessing

file = '/home/dasarih/Desktop/karthik/RSRR_SA_daily_ts.nc'
ds = xr.open_dataset(file)
lat = ds.XLONG
lon = ds.XLAT
lon, lat = np.meshgrid(lon, lat)



# Create a Dask array from the rainfall data
#chunks = {'time': 1000, 'latitude':'auto' , 'longitude':'auto'}
rain_dask = da.from_array(ds.rain, chunks={'XTIME':10})

rf = ds['rain']

def process_chunk(chunk):
    return chunk[~np.isnan(chunk)].flatten()
chunks = [rf.data[i] for i in range(rf.data.numblocks[0])]


num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(process_chunk)(chunk.compute()) for chunk in chunks)


rf_flat_np = np.concatenate(results)



fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

xlabels = ["35E", "40E", "45E", "50E"]
ylabels = ["15N", "20N", "25N", "30N"]
extent = [35., 60., 15., 30.]
xextent = np.arange(35.0, 50.1, 5.0)
yextent = np.arange(15.0, 30.1, 5.0)
levels = np.arange(0.0, 40.0, 2.0)

rain_vec = ax.contourf(lon, lat, rain_dask[0, :, :], transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both')
cbar = plt.colorbar(rain_vec, ax=ax, orientation='vertical', shrink=0.9, pad=0.03, extend='both')
cbar.set_label('Rainfall')

nframes = rain_dask.shape[0]  # Total number of time steps

def update(frame):

    if frame % 100 == 0:
        print("Progress: {:.2f}%".format(frame / nframes * 100))

    ax.clear()
    ax.coastlines()
    ax.set_extent(extent, ccrs.PlateCarree())
    ax.coastlines(linewidths=0.5)
    ax.set_yticks(yextent)
    ax.set_xticks(xextent)
    ax.set_xticklabels(xlabels, weight='bold')
    ax.set_yticklabels(ylabels, weight='bold')
    rain_vec.set_array(rain_dask[frame, :, :].compute().flatten())

ani = animation.FuncAnimation(fig, update, frames=nframes, interval=500)
ani.save('rain_test.gif', writer='pillow', dpi=300)































