import dask
import dask.array as da
import numpy as np
import xarray as xr
import dask.array as da
from scipy.stats import linregress
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import matplotlib as mpl
matplotlib.use('Agg')
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from scipy.stats import linregress

fname1 = '/data2/CMIP_ANALYSIS/Analysis/paper_AP_DI/WORLD_SHP/World_Countries.shp'
shape_feature1 = ShapelyFeature(Reader(fname1).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')


def plt_mapa(av,xt=0,title=' ',yt=0, yl=0, xl=0):
    if yl ==1:
        av.set_ylabel('Longitude',fontsize=16)

    if xl ==1:
        av.set_xlabel('Latitude',fontsize=16)

    if xt ==1 :
        av.set_xticks(np.arange(30, 65, 10))

    if yt ==1:
        av.set_yticks(np.arange(5, 36, 5))

    av.set_xlim([25, 65])
    av.set_ylim([5, 36])
    av.set_title(title,fontsize=16)
    av.add_feature(cfeature.COASTLINE)
    av.add_feature(shape_feature1, facecolor='none',lw=0.8)
    lon_formatter = LongitudeFormatter(dateline_direction_label=True)
    lat_formatter = LatitudeFormatter()
    av.xaxis.set_major_formatter(lon_formatter)
    av.yaxis.set_major_formatter(lat_formatter)
    av.tick_params(which='major', width=1.00, length=5, labelsize=10)
    av.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
    av.xaxis.set_ticks_position('bottom')

    return


file_paths = {f"file{i}": f"/home/dasarih/Desktop/karthik/wp20m_{i:02d}_mean.nc" for i in range(1, 13)}

file1  ='/home/dasarih/Desktop/karthik/wp20m_01_mean.nc'
file2  ='/home/dasarih/Desktop/karthik/wp20m_02_mean.nc'
file3  ='/home/dasarih/Desktop/karthik/wp20m_03_mean.nc'
file4  ='/home/dasarih/Desktop/karthik/wp20m_04_mean.nc'
file5  ='/home/dasarih/Desktop/karthik/wp20m_05_mean.nc'
file6  ='/home/dasarih/Desktop/karthik/wp20m_06_mean.nc'
file7  ='/home/dasarih/Desktop/karthik/wp20m_07_mean.nc'
file8  ='/home/dasarih/Desktop/karthik/wp20m_08_mean.nc'
file9  ='/home/dasarih/Desktop/karthik/wp20m_09_mean.nc'
file10 ='/home/dasarih/Desktop/karthik/wp20m_10_mean.nc'
file11 ='/home/dasarih/Desktop/karthik/wp20m_11_mean.nc'
file12 ='/home/dasarih/Desktop/karthik/wp20m_12_mean.nc'

d1=xr.open_dataset(file1)
d2=xr.open_dataset(file2)
d3=xr.open_dataset(file3)
d4=xr.open_dataset(file4)
d5=xr.open_dataset(file5)
d6=xr.open_dataset(file6)
d7=xr.open_dataset(file7)
d8=xr.open_dataset(file8)
d9=xr.open_dataset(file9)
d10=xr.open_dataset(file10)
d11=xr.open_dataset(file11)
d12=xr.open_dataset(file12)

d1.close()
d2.close()
d3.close()
d4.close()
d5.close()
d6.close()
d7.close()
d8.close()
d9.close()
d10.close()
d11.close()
d12.close()







time_sec1 = (d1['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val1, lon_val1, time_sec1, d1):
    # Extract data as NumPy arrays
    time_data = time_sec1.values
    data = d1.sel(latitude=lat_val1, longitude=lon_val1)['wp20m'].values

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
def compute_trend_block(lat_vals1, lon_vals1, time_sec1, d1):
    results = []
    for lat_val1 in lat_vals1:
        row_results = []
        for lon_val1 in lon_vals1:
            slope, p_value = compute_trend(lat_val1, lon_val1, time_sec1, d1)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals1 = d1['latitude'].values
lon_vals1 = d1['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals1, lon_vals1, time_sec1, d1, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks1 = result['slope']
p_values_blocks1 = result['p_value']

###############     2     

time_sec2 = (d2['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val2, lon_val2, time_sec2, d2):
    # Extract data as NumPy arrays
    time_data = time_sec2.values
    data = d2.sel(latitude=lat_val2, longitude=lon_val2)['wp20m'].values

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
def compute_trend_block(lat_vals2, lon_vals2, time_sec2, d2):
    results = []
    for lat_val2 in lat_vals2:
        row_results = []
        for lon_val2 in lon_vals2:
            slope, p_value = compute_trend(lat_val2, lon_val2, time_sec2, d2)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals2 = d2['latitude'].values
lon_vals2 = d2['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals2, lon_vals2, time_sec2, d2, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(10, 10))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks2 = result['slope']
p_values_blocks2 = result['p_value']

#########################################   3


time_sec3 = (d3['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val3, lon_val3, time_sec3, d3):
    # Extract data as NumPy arrays
    time_data = time_sec3.values
    data = d3.sel(latitude=lat_val3, longitude=lon_val3)['wp20m'].values

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
def compute_trend_block(lat_vals3, lon_vals3, time_sec3, d3):
    results = []
    for lat_val3 in lat_vals3:
        row_results = []
        for lon_val3 in lon_vals3:
            slope, p_value = compute_trend(lat_val3, lon_val3, time_sec3, d3)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals3 = d3['latitude'].values
lon_vals3 = d3['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals3, lon_vals3, time_sec3, d3, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(10, 10))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks3 = result['slope']
p_values_blocks3 = result['p_value']

###############################################    4

time_sec4 = (d4['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val4, lon_val4, time_sec4, d4):
    # Extract data as NumPy arrays
    time_data = time_sec4.values
    data = d4.sel(latitude=lat_val4, longitude=lon_val4)['wp20m'].values

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
def compute_trend_block(lat_vals4, lon_vals4, time_sec4, d4):
    results = []
    for lat_val4 in lat_vals4:
        row_results = []
        for lon_val4 in lon_vals4:
            slope, p_value = compute_trend(lat_val4, lon_val4, time_sec4, d4)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals4 = d4['latitude'].values
lon_vals4 = d4['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals4, lon_vals4, time_sec4, d4, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(10, 10))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks4 = result['slope']
p_values_blocks4 = result['p_value']


############################################      5
time_sec5 = (d5['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val5, lon_val5, time_sec5, d5):
    # Extract data as NumPy arrays
    time_data = time_sec5.values
    data = d5.sel(latitude=lat_val5, longitude=lon_val5)['wp20m'].values

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
def compute_trend_block(lat_vals5, lon_vals5, time_sec5, d5):
    results = []
    for lat_val5 in lat_vals5:
        row_results = []
        for lon_val5 in lon_vals5:
            slope, p_value = compute_trend(lat_val5, lon_val5, time_sec5, d5)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals5 = d5['latitude'].values
lon_vals5 = d5['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals5, lon_vals5, time_sec5, d5, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks5 = result['slope']
p_values_blocks5 = result['p_value']

##############################################     6

time_sec6 = (d6['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val6, lon_val6, time_sec6, d6):
    # Extract data as NumPy arrays
    time_data = time_sec6.values
    data = d6.sel(latitude=lat_val6, longitude=lon_val6)['wp20m'].values

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
def compute_trend_block(lat_vals6, lon_vals6, time_sec6, d6):
    results = []
    for lat_val6 in lat_vals6:
        row_results = []
        for lon_val6 in lon_vals6:
            slope, p_value = compute_trend(lat_val6, lon_val6, time_sec6, d6)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals6 = d6['latitude'].values
lon_vals6 = d6['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals6, lon_vals6, time_sec6, d6, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks6 = result['slope']
p_values_blocks6 = result['p_value']


###################################                7          

time_sec7 = (d7['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val7, lon_val7, time_sec7, d7):
    # Extract data as NumPy arrays
    time_data = time_sec7.values
    data = d7.sel(latitude=lat_val7, longitude=lon_val7)['wp20m'].values

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
def compute_trend_block(lat_vals7, lon_vals7, time_sec7, d7):
    results = []
    for lat_val7 in lat_vals7:
        row_results = []
        for lon_val7 in lon_vals7:
            slope, p_value = compute_trend(lat_val7, lon_val7, time_sec7, d7)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals7 = d7['latitude'].values
lon_vals7 = d7['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals7, lon_vals7, time_sec7, d7, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute: the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks7 = result['slope']
p_values_blocks7 = result['p_value']

##########################################################   8

time_sec8 = (d8['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val8, lon_val8, time_sec8, d8):
    # Extract data as NumPy arrays
    time_data = time_sec8.values
    data = d8.sel(latitude=lat_val8, longitude=lon_val8)['wp20m'].values

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
def compute_trend_block(lat_vals8, lon_vals8, time_sec8, d8):
    results = []
    for lat_val8 in lat_vals8:
        row_results = []
        for lon_val8 in lon_vals8:
            slope, p_value = compute_trend(lat_val8, lon_val8, time_sec8, d8)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals8 = d8['latitude'].values
lon_vals8 = d8['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals8, lon_vals8, time_sec8, d8, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks8 = result['slope']
p_values_blocks8 = result['p_value']


###############################################################   9

time_sec9 = (d9['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val9, lon_val9, time_sec9, d9):
    # Extract data as NumPy arrays
    time_data = time_sec9.values
    data = d9.sel(latitude=lat_val9, longitude=lon_val9)['wp20m'].values

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
def compute_trend_block(lat_vals9, lon_vals9, time_sec9, d9):
    results = []
    for lat_val9 in lat_vals9:
        row_results = []
        for lon_val9 in lon_vals9:
            slope, p_value = compute_trend(lat_val9, lon_val9, time_sec9, d9)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals9 = d9['latitude'].values
lon_vals9 = d9['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals9, lon_vals9, time_sec9, d9, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks9 = result['slope']
p_values_blocks9 = result['p_value']


########################################################   10

time_sec10 = (d10['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val10, lon_val10, time_sec10, d10):
    # Extract data as NumPy arrays
    time_data = time_sec10.values
    data = d10.sel(latitude=lat_val10, longitude=lon_val10)['wp20m'].values

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
def compute_trend_block(lat_vals10, lon_vals10, time_sec10, d10):
    results = []
    for lat_val10 in lat_vals10:
        row_results = []
        for lon_val10 in lon_vals10:
            slope, p_value = compute_trend(lat_val10, lon_val10, time_sec10, d10)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals10 = d10['latitude'].values
lon_vals10 = d10['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals10, lon_vals10, time_sec10, d10, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks10 = result['slope']
p_values_blocks10 = result['p_value']

#######################################################        11

time_sec11 = (d11['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val11, lon_val11, time_sec11, d11):
    # Extract data as NumPy arrays
    time_data = time_sec11.values
    data = d11.sel(latitude=lat_val11, longitude=lon_val11)['wp20m'].values

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
def compute_trend_block(lat_vals11, lon_vals11, time_sec11, d11):
    results = []
    for lat_val11 in lat_vals11:
        row_results = []
        for lon_val11 in lon_vals11:
            slope, p_value = compute_trend(lat_val11, lon_val11, time_sec11, d11)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals11 = d11['latitude'].values
lon_vals11 = d11['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals11, lon_vals11, time_sec11, d11, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks11 = result['slope']
p_values_blocks11 = result['p_value']

############################################################# 12

time_sec12 = (d12['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

############
def compute_trend(lat_val12, lon_val12, time_sec12, d12):
    # Extract data as NumPy arrays
    time_data = time_sec12.values
    data = d12.sel(latitude=lat_val12, longitude=lon_val12)['wp20m'].values

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
def compute_trend_block(lat_vals12, lon_vals12, time_sec12, d12):
    results = []
    for lat_val12 in lat_vals12:
        row_results = []
        for lon_val12 in lon_vals12:
            slope, p_value = compute_trend(lat_val12, lon_val12, time_sec12, d12)
            row_results.append((slope, p_value))
        results.append(row_results)
    return np.array(results, dtype=[('slope', np.float64), ('p_value', np.float64)])

# Compute trends and p-values using Dask
lat_vals12 = d12['latitude'].values
lon_vals12 = d12['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals10, lon_vals10, time_sec10, d10, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks12 = result['slope']
p_values_blocks12 = result['p_value']

##########################################################################################################

x1, y1 = np.meshgrid(lon_vals1, lat_vals1)
x2, y2 = np.meshgrid(lon_vals2, lat_vals2)
x3, y3 = np.meshgrid(lon_vals3, lat_vals3)
x4, y4 = np.meshgrid(lon_vals4, lat_vals4)
x5, y5 = np.meshgrid(lon_vals5, lat_vals5)
x6, y6 = np.meshgrid(lon_vals6, lat_vals6)
x7, y7 = np.meshgrid(lon_vals7, lat_vals7)
x8, y8 = np.meshgrid(lon_vals8, lat_vals8)
x9, y9 = np.meshgrid(lon_vals9, lat_vals9)
x10, y10 = np.meshgrid(lon_vals10, lat_vals10)
x11, y11 = np.meshgrid(lon_vals11, lat_vals11)
x12, y12 = np.meshgrid(lon_vals12, lat_vals12)

##############################################
significance_mask1 = p_values_blocks1 < 0.05
significance_mask2 = p_values_blocks2 < 0.05
significance_mask3 = p_values_blocks3 < 0.05
significance_mask4 = p_values_blocks4 < 0.05
significance_mask5 = p_values_blocks5 < 0.05
significance_mask6 = p_values_blocks6 < 0.05
significance_mask7 = p_values_blocks7 < 0.05
significance_mask8 = p_values_blocks8 < 0.05
significance_mask9 = p_values_blocks9 < 0.05
significance_mask10 = p_values_blocks10 < 0.05
significance_mask11 = p_values_blocks11 < 0.05
significance_mask12 = p_values_blocks12 < 0.05

hatch_levels = [0.5, 1]
hatch_colors = ['none', 'black']  # Color for the hatches
hatch_styles = ['.', None]  # Hatching styles
###############################################

positive_slope_levels = np.linspace(0, 0.1, 3)  # Adjust levels as needed
negative_slope_levels = np.linspace(-0.1, 0, 3)  # Adjust levels as needed

origin = 'lower'
us = np.arange(100,1200.0,20.0)


#fig, axes = plt.subplots(1,2)
fig, axes = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},nrows=3,ncols=4, figsize=(13.4,7.2))
#print(wp1.shape)



axes[0][0].contourf(x1,y1, d1['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[0][0].contourf(x1,y1, significance_mask1,colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[0][0].contour(x1, y1, trend_blocks1, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[0][0].contour(x1, y1, trend_blocks1, levels=negative_slope_levels, colors='red', linestyles='dotted',                     transform=ccrs.PlateCarree())
plt_mapa(axes[0][0],title='July', yt=1, yl=1,xt=0,xl=0)

############################
axes[0][1].contourf(x2,y2, d2['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[0][1].contourf(x2,y2, significance_mask2,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[0][1].contour(x2, y2, trend_blocks2, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[0][1].contour(x2, y2, trend_blocks2, levels=negative_slope_levels, colors='red', linestyles='dotted',                     transform=ccrs.PlateCarree())

plt_mapa(axes[0][1],title='August' ,yt=0, yl=0,xt=0,xl=0)

#########################
axes[0][2].contourf(x3,y3, d3['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[0][2].contourf(x3,y3, significance_mask3,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[0][2].contour(x3, y3, trend_blocks3, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[0][2].contour(x3, y3, trend_blocks3, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt_mapa(axes[0][2],title='September' ,yt=0, yl=0, xt=0, xl=0)

#################################
axes[0][3].contourf(x4,y4, d4['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[0][3].contourf(x4,y4, significance_mask4,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[0][3].contour(x4, y4, trend_blocks4, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[0][3].contour(x4, y4, trend_blocks4, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt_mapa(axes[0][3],title='October' ,yt=0, yl=0,xt=0, xl=0)

######################################
axes[1][0].contourf(x5,y5, d5['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[1][0].contourf(x5,y5, significance_mask5,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[1][0].contour(x5, y5, trend_blocks5, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[1][0].contour(x5, y5, trend_blocks5, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt_mapa(axes[1][0],title='November' ,yt=1, yl=1, xt=0, xl=0)

##############################
axes[1][1].contourf(x6,y6, d6['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[1][1].contourf(x6,y6, significance_mask6,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[1][1].contour(x6, y6, trend_blocks6, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[1][1].contour(x6, y6, trend_blocks6, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt_mapa(axes[1][1],title=' December' ,yt=0, yl=0,xt=0, xl=0)

################################
axes[1][2].contourf(x7,y7, d7['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[1][2].contourf(x7,y7, significance_mask7,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[1][2].contour(x7, y7, trend_blocks7, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[1][2].contour(x7, y7, trend_blocks7, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt_mapa(axes[1][2], title='January',yt=0, yl=0, xt=0, xl=0)

###############################
axes[1][3].contourf(x8,y8, d8['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[1][3].contourf(x8,y8, significance_mask8,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[1][3].contour(x8, y8, trend_blocks8, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[1][3].contour(x8, y8, trend_blocks8, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt_mapa(axes[1][3],title='February' ,yt=0, yl=0, xt=0, xl=0)

###################################
axes[2][0].contourf(x9,y9, d9['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[2][0].contourf(x9,y9, significance_mask9,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[2][0].contour(x9, y9, trend_blocks9, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[2][0].contour(x9, y9, trend_blocks9, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt_mapa(axes[2][0],title='March' ,yt=1, yl=1,xt=1, xl=1)

#####################################
axes[2][1].contourf(x10,y10, d10['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[2][1].contourf(x10,y10, significance_mask10,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[2][1].contour(x10, y10, trend_blocks10, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[2][1].contour(x10, y10, trend_blocks10, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt_mapa(axes[2][1],title='April' ,yt=0, yl=0, xt=1, xl=1)

#######################################
axes[2][2].contourf(x11,y11, d11['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[2][2].contourf(x11,y11, significance_mask11,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[2][2].contour(x11, y11, trend_blocks11, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[2][2].contour(x11, y11, trend_blocks11, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt_mapa(axes[2][2],title='May' ,yt=0, yl=0,xt=1, xl=1)

#######################################
im=axes[2][3].contourf(x12,y12, d12['wp20m'][0,:,:],
              levels=us,cmap='Spectral_r',extend="both",origin=origin,transform=ccrs.PlateCarree())
axes[2][3].contourf(x12,y12, significance_mask12,
                    colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0,transform=ccrs.PlateCarree())
axes[2][3].contour(x12, y12, trend_blocks12, levels=positive_slope_levels, colors='Yellow', linestyles='solid',                    transform=ccrs.PlateCarree())
axes[2][3].contour(x12, y12, trend_blocks12, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt_mapa(axes[2][3], title='June',yt=0, yl=0, xt=1, xl=1)


ust = np.arange(100,400.,5)
cbar = plt.colorbar(im, ax=axes.ravel(), orientation='vertical', shrink=0.9, pad=0.03, extend='both')
cbar.set_label('wind power')

#cb1=fig.colorbar(im, ax=axes.ravel(),aspect=30,pad=0.02,shrink=0.96,ticks=ust)
#cb1.ax.tick_params(labelsize=18)

## Main-Title
fig.suptitle('Wind Power', y=0.97, fontsize=25)

plt.savefig('Climate_dask_masking.png',format='png',dpi=300)
quit()

