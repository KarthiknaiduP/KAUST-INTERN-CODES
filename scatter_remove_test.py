import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess
from pwlf import PiecewiseLinFit

# Open datasets
file1 = '/home/dasarih/Desktop/karthik/day_wp100m_1980-2023_file.nc'
file2 = '/home/dasarih/Desktop/karthik/observations/ncfiles/AlBada_1_F3_WS_TFCA_file.nc'

ds = xr.open_dataset(file1)
df = xr.open_dataset(file2)

# Extract variables
lat = ds.latitude
lon = ds.longitude
wp100m = ds.wp100m
v = df.F3_WS_TFCA
wp100 = (0.5) * (1.225) * (v) * (v) * (v)

# Print shapes
print(lat.shape)
print(lon.shape)
print(wp100m.shape)
print(v.shape)
print(wp100.shape)

# Define specific latitude and longitude point
lat1 = 28.498333
lon1 = 34.977778

# Find indices of closest points
index_latitude1 = np.abs(lat - lat1).argmin()
index_longitude1 = np.abs(lon - lon1).argmin()

# Extract values at specified point
x = wp100m[:, index_latitude1, index_longitude1].values
y = wp100.values

# Remove outliers
x_clipped = np.clip(x, 0, 1000)
y_clipped = np.clip(y, 0, 1000)

# Calculate leverage
X_design = np.vstack([np.ones(len(x_clipped)), x_clipped]).T
hat_matrix = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
leverage = np.diag(hat_matrix)
mask = leverage < 0.5
x_filtered = x_clipped[mask]
y_filtered = y_clipped[mask]

# Log-log transformation
x_log = np.log(x_filtered)
y_log = np.log(y_filtered)

# Fit a linear model to the log-transformed data
m_log, b_log = np.polyfit(x_log, y_log, 1)

# Convert the slope and intercept back to the original scale
m = np.exp(m_log)
b = np.exp(b_log)

# Non-parametric regression using LOESS
y_smooth = lowess(y_filtered, x_filtered, frac=0.5)

# Piecewise regression
pwlf = PiecewiseLinFit(x_filtered, y_filtered)
pwlf.fit(n_segments=2)

# Create scatter plot
fig = plt.figure(figsize=(10, 6))
colors = np.arange(0, len(x_filtered), 1)
plt.scatter(x_filtered, y_filtered, alpha=0.5, c=colors, cmap='rainbow')
plt.colorbar()
plt.xlabel('WP100m')
plt.ylabel('Wind Power')

# Add linear regression line
plt.axline(xy1=(0, b), slope=m, color='r', label=f'$y = {m:.2f}x {b:+.2f}$')

# Add LOESS smoothing line
#plt.plot(x_filtered, y_smooth, 'g-', label='LOESS Smoothing')

# Add piecewise regression line
#plt.plot(x_filtered, pwlf.predict(x_filtered), 'b-', label='Piecewise Regression')

plt.legend()

# Show plot
plt.show()
