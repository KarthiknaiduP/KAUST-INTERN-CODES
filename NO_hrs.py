import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# Load NetCDF data
file1 = '/home/dasarih/Desktop/karthik/observations/ncfiles/AlBada_2_F6_WS_TFCA_hourmean_updated.nc'
df = xr.open_dataset(file1)

# Filter data for one year, assuming the dataset has a 'time' variable
year_to_filter = '2022'
df_year = df.sel(time=slice(f'{year_to_filter}-01-01', f'{year_to_filter}-12-31'))

# Assuming 'F6_WS_TFCA' is the variable containing wind speeds
wind_speeds = df_year['F6_WS_TFCA'].values

# Round wind speeds to the nearest integer
rounded_wind_speeds = np.round(wind_speeds).astype(int)

# Create a lookup table for wind speed and power output
wind_speed_power = {
    1: -0.5, 2: -0.5, 3: 1.2, 4: 7.2, 5: 14.5, 6: 24.7, 7: 37.9, 8: 58.7, 9: 74.8, 10: 85.1,
    11: 90.2, 12: 94.7, 13: 95.3, 14: 95.1, 15: 94.2, 16: 92.9, 17: 91.2, 18: 88.9, 19: 87.1, 
    20: 84.1, 21: 81.3, 22: 78.6, 23: 75.1, 24: 74.3, 25: 71.7
}

# Map the rounded wind speeds to their corresponding power outputs
power_values = np.array([wind_speed_power[speed] if speed in wind_speed_power else 0 for speed in rounded_wind_speeds])

# Calculate number of hours at each rounded wind speed
unique_rounded_speeds, counts = np.unique(rounded_wind_speeds, return_counts=True)

# Create a pandas DataFrame for wind speed data
data_wind = {
    'Rounded Wind Speed (m/s)': unique_rounded_speeds,
    'Number of Hours': counts
}
df_rounded_hours = pd.DataFrame(data_wind)

# Calculate the sum of power at each rounded wind speed
power_sum = np.array([power_values[rounded_wind_speeds == speed].sum() for speed in unique_rounded_speeds])

# Create a pandas DataFrame for power data
data_power = {
    'Rounded Wind Speed (m/s)': unique_rounded_speeds,
    'Power (kW)': power_sum
}
df_power = pd.DataFrame(data_power)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the bar plot for the number of hours at each rounded wind speed
ax1.bar(df_rounded_hours['Rounded Wind Speed (m/s)'], df_rounded_hours['Number of Hours'], color='skyblue', label='Number of Hours')
ax1.set_xlabel('Rounded Wind Speed (m/s)')
ax1.set_ylabel('Number of Hours')
ax1.grid(axis='y', linestyle='--')
ax1.set_xticks(df_rounded_hours['Rounded Wind Speed (m/s)'])

# Create a second y-axis to plot the power curve
ax2 = ax1.twinx()
ax2.plot(df_power['Rounded Wind Speed (m/s)'], df_power['Power (kW)'], color='red', label='Power (kW)')
ax2.set_ylabel('Power (kW)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add title and legend
fig.suptitle(f'Number of Hours at Each Rounded Wind Speed and Corresponding Power for {year_to_filter}')
ax1.legend(['Number of Hours'], loc='upper left')
ax2.legend(['Power (kW)'], loc='upper right')

plt.tight_layout()
plt.show()

# Print all power values
print("Power values (kW):", power_values)

# Close the NetCDF file
df.close()

