import numpy as np
from dask.diagnostics import ProgressBar

import dask.array as da
import xarray as xr
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

# Load the dataset
file2 = '/home/dasarih/Desktop/karthik/wp20m_01_mean.nc'
ds = xr.open_dataset(file2, chunks={'time': 100})

# Compute time in seconds for regression
time_sec = (ds['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')





# Define function to compute trend and p-value for a single grid point using Xarray
def compute_trend_xr(lat_val, lon_val, time_sec, ds):
    data = ds.sel(latitude=lat_val, longitude=lon_val)['wp20m']
    time_data = time_sec
    valid_indices = np.isfinite(time_data) & np.isfinite(data)
    time_data = time_data[valid_indices]
    data = data[valid_indices]
    if len(time_data) > 0:
        slope, intercept, r_value, p_value, std_err = linregress(time_data, data)
    else:
        slope, p_value = np.nan, np.nan
    return slope, p_value

# Compute trends and p-values using Xarray with tqdm progress bar
xr_results = []
start_time = time.perf_counter()
for lat_val in tqdm(ds['latitude'], desc='Xarray Latitude'):
    row_results = []
    for lon_val in tqdm(ds['longitude'], desc='Xarray Longitude', leave=False):
        slope, p_value = compute_trend_xr(lat_val, lon_val, time_sec, ds)
        row_results.append((slope, p_value))
    xr_results.append(row_results)
xr_time = time.perf_counter() - start_time








# Define function to compute trend and p-value for a single grid point using NumPy
def compute_trend_numpy(lat_val, lon_val, time_sec, ds):
    # Extract data as NumPy arrays
    time_data = time_sec.values
    data = ds.sel(latitude=lat_val, longitude=lon_val)['wp20m'].values

    # Remove NaN values from time_data and corresponding data
    valid_indices = np.isfinite(time_data) & np.isfinite(data)
    time_data = time_data[valid_indices]
    data = data[valid_indices]

    # Compute linear regression if there are valid data points
    if len(time_data) > 0:
        slope, intercept, r_value, p_value, std_err = linregress(time_data, data)
    else:
        # Handle case where no valid data points are available
        slope, p_value = np.nan, np.nan

    return slope, p_value

# Compute trends and p-values using NumPy with tqdm progress bar
lat_vals = ds['latitude'].values
lon_vals = ds['longitude'].values

numpy_results = []
start_time = time.perf_counter()
for lat_val in tqdm(lat_vals, desc='NumPy Latitude'):
    row_results = []
    for lon_val in tqdm(lon_vals, desc='NumPy Longitude', leave=False):
        slope, p_value = compute_trend_numpy(lat_val, lon_val, time_sec, ds)
        row_results.append((slope, p_value))
    numpy_results.append(row_results)
numpy_time = time.perf_counter() - start_time

# Define function to compute trend and p-value for a single grid point using Dask

#def compute_trend_dask(lat_val, lon_val, time_sec, ds):
#    # Extract data as Dask arrays
#    time_data = da.from_array(time_sec, chunks=time_sec.chunks)
#    data = ds.sel(latitude=lat_val, longitude=lon_val)['wp20m']
#
#    # Remove NaN values from time_data and corresponding data
#    valid_indices = np.isfinite(time_data) & np.isfinite(data)
#    time_data = time_data[valid_indices]
#    data = data[valid_indices]
#
#    # Compute linear regression if there are valid data points
#    if len(time_data) > 0:
#        slope, intercept, r_value, p_value, std_err = linregress(time_data.compute(), data.compute())
#    else:
#        # Handle case where no valid data points are available
#        slope, p_value = np.nan, np.nan
#
#    return slope, p_value

def compute_trend(lat_val, lon_val, time_sec, ds):
    # Extract data as NumPy arrays
    time_data = time_sec.values
    data = ds.sel(latitude=lat_val, longitude=lon_val)['wp20m'].values

    # Remove NaN values from time_data and corresponding data
    valid_indices = np.isfinite(time_data) & np.isfinite(data)
    time_data = time_data[valid_indices]
    data = data[valid_indices]

    # Compute linear regression if there are valid data points
    if len(time_data) > 0:
        slope, intercept, r_value, p_value, std_err = linregress(time_data, data)
    else:
        # Handle case where no valid data points are available
        slope, p_value = np.nan, np.nan

    return slope, p_value

# Use Dask's map_blocks to parallelize the computation
def compute_trend_block(lat_vals, lon_vals, time_sec, ds):
    results = []
    for lat_val in lat_vals:
        row_results = []
        for lon_val in lon_vals:
            slope, p_value = compute_trend(lat_val, lon_val, time_sec, ds)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])



# Compute trends and p-values using Dask with tqdm progress bar
dask_results = []
start_time = time.perf_counter()


result_blocks = da.map_blocks(compute_trend_block, lat_vals, lon_vals, time_sec, ds, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()










#for lat_val in tqdm(lat_vals, desc='Dask Latitude'):
#    row_results = []
#    for lon_val in tqdm(lon_vals, desc='Dask Longitude', leave=False):
#        slope, p_value = compute_trend_dask(lat_val, lon_val, time_sec, ds)
#        row_results.append((slope, p_value))
#    dask_results.append(row_results)
dask_time = time.perf_counter() - start_time


# Define function to compute trend and p-value for a single grid point using Xarray
#def compute_trend_xr(lat_val, lon_val, time_sec, ds):
#   data = ds.sel(latitude=lat_val, longitude=lon_val)['wp20m']
#   time_data = time_sec
#   valid_indices = np.isfinite(time_data) & np.isfinite(data)
#   time_data = time_data[valid_indices]
#   data = data[valid_indices]
#   if len(time_data) > 0:
#       slope, intercept, r_value, p_value, std_err = linregress(time_data, data)
#   else:
#       slope, p_value = np.nan, np.nan
#   return slope, p_value

# Compute trends and p-values using Xarray with tqdm progress bar
#r_results = []
#tart_time = time.perf_counter()
#or lat_val in tqdm(ds['latitude'], desc='Xarray Latitude'):
#   row_results = []
#   for lon_val in tqdm(ds['longitude'], desc='Xarray Longitude', leave=False):
#       slope, p_value = compute_trend_xr(lat_val, lon_val, time_sec, ds)
#       row_results.append((slope, p_value))
#   xr_results.append(row_results)
#r_time = time.perf_counter() - start_time

# Create a bar chart
labels = ['Xarray', 'Numpy', 'Dask']
times = [xr_time, numpy_time, dask_time]

plt.bar(labels, times)
plt.xlabel('Library')
plt.ylabel('Computation Time (seconds)')
plt.title('Computation Time Comparison')
for label, time in zip(labels, times):
    plt.text(label, time, f'{label} Time = {time:.2f} s', ha='center', va='bottom')
plt.savefig('libraries_time_comparison.png')
plt.show()
