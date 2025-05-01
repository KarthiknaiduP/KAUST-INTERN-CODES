import numpy as np
import xarray as xr
import dask.array as da
from scipy.stats import linregress
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar

# Load your dataset
file2 = '/home/dasarih/Desktop/karthik/wp20m_01_mean.nc'
ds = xr.open_dataset(file2, chunks={'time': 100})

# Compute time in seconds for regression
time_sec = (ds['time'].astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

# Define function to compute trend and p-value for a single grid point
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

# Compute trends and p-values using Dask
lat_vals = ds['latitude'].values
lon_vals = ds['longitude'].values

result_blocks = da.map_blocks(compute_trend_block, lat_vals, lon_vals, time_sec, ds, dtype=np.dtype([('slope', np.float64), ('p_value', np.float64)]), chunks=(100, 100))

# Compute the results using Dask
with ProgressBar():
    result = result_blocks.compute()

# Extract the trend and p-values from the computed result
trend_blocks = result['slope']
p_values_blocks = result['p_value']

# Plotting
fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection=ccrs.PlateCarree())

xlabels=["25E","35E","45E","55E","65E"]
ylabels=["5N","15N","25N","35N"]
extent = [25.,65.,5.,35.]
xextent=np.arange(25.0,65.1,10.0)
yextent=np.arange(5.0,35.1,10.0)
levels=np.arange(100.0,1200.0,20.0)

wind_vec1=ax.contourf(lon_vals, lat_vals, ds['wp20m'][0,:,:], transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels, extend='both')
cbar1 = plt.colorbar(wind_vec1, ax=ax, orientation='vertical', shrink=0.9, pad=0.03, extend='both')

ax.clear()
ax.coastlines()
ax.set_extent(extent, ccrs.PlateCarree())

ax.coastlines(linewidths=0.5)
ax.set_yticks(yextent)
ax.set_xticks(xextent)
ax.set_xticklabels(xlabels,weight='bold')
ax.set_yticklabels(ylabels,weight='bold')
ax.set_title('Jan')

ax.contourf(lon_vals, lat_vals, ds['wp20m'][0,:,:], transform=ccrs.PlateCarree(), cmap='Spectral_r', levels=levels,extend='both')

significance_mask = p_values_blocks < 0.05
hatch_levels = [0.5, 1]
hatch_colors = ['none', 'black']  # Color for the hatches
hatch_styles = ['.', None]  # Hatching styles
mask_plot = ax.contourf(lon_vals, lat_vals, significance_mask, transform=ccrs.PlateCarree(), colors=hatch_colors, hatches=hatch_styles, levels=hatch_levels, alpha=0)

positive_slope_levels = np.linspace(0, 0.1, 3)  # Adjust levels as needed
positive_contour_lines = ax.contour(lon_vals, lat_vals, trend_blocks, levels=positive_slope_levels, colors='Yellow', linestyles='solid', transform=ccrs.PlateCarree())

# Overlay contour lines representing negative slope as black dotted lines
negative_slope_levels = np.linspace(-0.1, 0, 3)  # Adjust levels as needed
negative_contour_lines = ax.contour(lon_vals, lat_vals, trend_blocks, levels=negative_slope_levels, colors='red', linestyles='dotted', transform=ccrs.PlateCarree())

plt.tight_layout()

plt.savefig('Trends_wp.png', format='png', dpi=300)
plt.show()

