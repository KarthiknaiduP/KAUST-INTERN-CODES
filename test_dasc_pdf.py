import numpy as np
import xarray as xr
import dask.array as da
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Load NetCDF file using xarray with Dask
file_path = '/data2/CMIP_ANALYSIS/rainfall_paper/data/RSRR_SA_daily_ts.nc'
ofile='/data2/CMIP_ANALYSIS/rainfall_paper/Figures/fig_rainfall_pdf_AP_RSRR_ERA5.png'
#dataset = xr.open_dataset(file_path)
#dataset = xr.open_dataset(file_path)
ds = xr.open_dataset(file_path, chunks={'XTIME': 10})  # Adjust chunk size as necessary

# Assuming the variable name in the NetCDF file is 'rf'
rf = ds['rain']

# Function to process chunks and flatten data
def process_chunk(chunk):
    return chunk[~np.isnan(chunk)].flatten()

# Split the Dask array into manageable chunks
chunks = [rf.data[i] for i in range(rf.data.numblocks[0])]

# Use joblib to parallelize processing of chunks
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(process_chunk)(chunk.compute()) for chunk in chunks)

# Concatenate the results
rf_flat_np = np.concatenate(results)

# Function to compute KDE
def compute_kde_chunk(data_chunk, x_range):
    kde = gaussian_kde(data_chunk)
    return kde(x_range)

# Compute percentiles
pdf_95 = np.percentile(rf_flat_np, 99.5)
pdf_99 = np.percentile(rf_flat_np, 99.9)

# Prepare for parallel KDE computation
#x_range = np.linspace(0, 50, 1000)
x_range = np.linspace(rf_flat_np.min(), rf_flat_np.max(), 1000)
num_chunks = num_cores
data_chunks = np.array_split(rf_flat_np, num_chunks)

# Compute KDE in parallel
kde_values_list = Parallel(n_jobs=num_cores)(
    delayed(compute_kde_chunk)(chunk, x_range) for chunk in data_chunks
)

# Aggregate KDE values
pdf_values = np.mean(kde_values_list, axis=0)

# Compute CDF from PDF
cdf_values = np.cumsum(pdf_values)
cdf_values /= cdf_values[-1]  # Normalize to get values between 0 and 1

# Plot PDF and CDF
fig, ax = plt.subplots(figsize=(10, 6), dpi=320)

#ax2 = ax1.twinx()
#sns.histplot(rf_flat_np, bins=50, kde=False, stat='density', ax=ax, color='blue')  # Disable automatic KDE for speed
#ax1.plot(x_range, pdf_values, label='PDF', color='blue')
ax.plot(x_range, pdf_values, label='PDF', color='blue')
#ax.plot(x_range, cdf_values, label='CDF', color='red')

# Plot vertical lines at 95% and 99% values of PDF
ax.axvline(x=pdf_95, color='green', linestyle='--', label='95% PDF')
ax.axvline(x=pdf_99, color='purple', linestyle='--', label='99% PDF')

print('P99.5 = ' , pdf_95)
print('P99.9 = ' , pdf_99)

ax.set_xlim(1, 40) 
ax.set_ylim(0.0, 0.0301) 
#ax.set_yscale('log') 
#ax2.set_xlim(0, 50) 

# Labels and title
ax.set_xlabel('Rainfal (mm / day)')
#ax1.set_ylabel('PDF', color='blue')
ax.set_ylabel('PDF', color='red')
ax.set_title('CDF of RSRR Rainfall over AP')

# Adding legends
ax.legend(loc='lower right')
#ax2.legend(loc='upper right')

# Save the plot
#plt.savefig('pdf_cdf_rsrr_plot.png')
#plt.savefig(ofile,dpi=300)
plt.show()

# Close the xarray dataset
ds.close()
