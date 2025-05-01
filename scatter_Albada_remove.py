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




file11 ='/home/dasarih/Desktop/karthik/day_wp100m_1980-2023_file.nc'
file3 ='/home/dasarih/Desktop/karthik/observations/ncfiles/AlBada_1_F3_WS_TFCA_file.nc'
file4 ='/home/dasarih/Desktop/karthik/observations/ncfiles/AlBada_1_F4_WS_TFCA_file.nc'

file12 ='/home/dasarih/Desktop/karthik/day_wp150m_1980-2023_file.nc'
file1 ='/home/dasarih/Desktop/karthik/observations/ncfiles/AlBada_1_F1_WS_TFCA_file.nc'
file2 ='/home/dasarih/Desktop/karthik/observations/ncfiles/AlBada_1_F2_WS_TFCA_file.nc'

file13 ='/home/dasarih/Desktop/karthik/day_wp80m_1980-2023_file.nc'
file5 ='/home/dasarih/Desktop/karthik/observations/ncfiles/AlBada_1_F5_WS_TFCA_file.nc'
file6 ='/home/dasarih/Desktop/karthik/observations/ncfiles/AlBada_1_F6_WS_TFCA_file.nc'

file14 ='/home/dasarih/Desktop/karthik/day_wp30m_1980-2023_file.nc'
file7 ='/home/dasarih/Desktop/karthik/observations/ncfiles/AlBada_1_F7_WS_TFCA_file.nc'
file8 ='/home/dasarih/Desktop/karthik/observations/ncfiles/AlBada_1_F8_WS_TFCA_file.nc'


ds11=xr.open_dataset(file11)
wp100  = ds11.wp100m
wp100m  = wp100*26/100
lat = ds11.latitude
lon = ds11.longitude

ds12=xr.open_dataset(file12)
wp150  = ds12.wp150m
wp150m  = wp150*26/100

ds13=xr.open_dataset(file13)
wp80  = ds13.wp80m
wp80m  = wp80*30/100

ds14=xr.open_dataset(file14)
wp30  = ds14.wp30m
wp30m  = wp30*31/100


df3 = xr.open_dataset(file3)
v3  = df3.F3_WS_TFCA
wp100_1 = (0.5)*(1.2065)*(v3)*(v3)*(v3)
df4 = xr.open_dataset(file4)
v4  = df4.F4_WS_TFCA
wp100_2 = (0.5)*(1.2065)*(v4)*(v4)*(v4)

df1 = xr.open_dataset(file1)
v1  = df1.F1_WS_TFCA
wp150_1 = (0.5)*(1.2065)*(v1)*(v1)*(v1)

df2 = xr.open_dataset(file2)
v2  = df2.F2_WS_TFCA
wp150_2 = (0.5)*(1.2065)*(v2)*(v2)*(v2)

df5 = xr.open_dataset(file5)
v5  = df5.F5_WS_TFCA
wp80_1 = (0.5)*(1.2065)*(v5)*(v5)*(v5)

df6 = xr.open_dataset(file6)
v6  = df6.F6_WS_TFCA
wp80_2 = (0.5)*(1.2065)*(v6)*(v6)*(v6)

df7 = xr.open_dataset(file7)
v7  = df7.F7_WS_TFCA
wp30_1 = (0.5)*(1.2065)*(v7)*(v7)*(v7)

df8 = xr.open_dataset(file8)
v8  = df8.F8_WS_TFCA
wp30_2 = (0.5)*(1.2065)*(v8)*(v8)*(v8)




#lat1=28.2954
#lon1=34.5804

lat1=28.258333
lon1=34.967778


index_latitude1 = np.abs(lat - lat1).argmin()
index_longitude1 = np.abs(lon - lon1).argmin()

ds11.close()
ds12.close()
ds13.close()
df1.close()
df2.close()
df3.close()
df4.close()
df5.close()
df6.close()
df7.close()
df8.close()

colors=np.arange(0,793,1)
origin = 'lower'

fig, axs = plt.subplots(2, 4, figsize=(10, 6))
plt.subplots_adjust(left=0.06, right=0.9, bottom=0.1, top=0.85, wspace=0.2, hspace=0.35)
#################################################################
plt.subplot(2,4,3)
x1= wp100m[:,index_latitude1,index_longitude1]
y1 = wp100_1.values

x1_clipped = np.clip(x1, 0, 1000)
y1_clipped = np.clip(y1, 0, 1000)

X1_design = np.vstack([np.ones(len(x1_clipped)), x1_clipped]).T
hat_matrix = X1_design @ np.linalg.inv(X1_design.T @ X1_design) @ X1_design.T
leverage = np.diag(hat_matrix)
mask = leverage < 0.5
x1_filtered = x1_clipped[mask]
y1_filtered = y1_clipped[mask]

x1_log = np.log(x1_filtered)
y1_log = np.log(y1_filtered)

m1_log, b1_log = np.polyfit(x1_log, y1_log, 1)

# Convert the slope and intercept back to the original scale
m1 = np.exp(m1_log)
b1 = np.exp(b1_log)

# Create scatter plot
plt.scatter(x1_filtered, y1_filtered, alpha=0.3, c=colors, cmap='rainbow')

plt.axline(xy1=(0, b1), slope=m1, color='black', label=f'$y = {m1:.2f}x {b1:+.2f}$')
plt.legend(fontsize='small')
plt.gca().set_title("F3_WS_TFCA")
plt.tick_params(axis='x', labelsize=0, length=0, width=0, labelbottom=False)
plt.tick_params(axis='y', labelsize=0, length=0, width=0, labelleft=False)
plt.xlabel('WP100M')
plt.ylabel('WP-OBS')
###########################################################################
plt.subplot(2,4,4)
x2=wp100m[:,index_latitude1,index_longitude1]
y2=wp100_2.values


x2_clipped = np.clip(x2, 0, 1000)
y2_clipped = np.clip(y2, 0, 1000)

X2_design = np.vstack([np.ones(len(x2_clipped)), x2_clipped]).T
hat_matrix = X2_design @ np.linalg.inv(X2_design.T @ X2_design) @ X2_design.T
leverage = np.diag(hat_matrix)
mask = leverage < 0.5
x2_filtered = x2_clipped[mask]
y2_filtered = y2_clipped[mask]

x2_log = np.log(x2_filtered)
y2_log = np.log(y2_filtered)

m2_log, b2_log = np.polyfit(x2_log, y2_log, 1)

# Convert the slope and intercept back to the original scale
m2 = np.exp(m2_log)
b2 = np.exp(b2_log)

# Create scatter plot
plt.scatter(x2_filtered, y2_filtered, alpha=0.3, c=colors, cmap='rainbow')
plt.axline(xy1=(0, b2), slope=m2, color='black', label=f'$y = {m2:.2f}x {b2:+.2f}$')

plt.legend(fontsize='small')
plt.gca().set_title("F4_WS_TFCA")
plt.tick_params(axis='x', labelsize=0, length=0, width=0, labelbottom=False)
plt.tick_params(axis='y', labelsize=0, length=0, width=0, labelleft=False)
plt.xlabel('WP100M')
plt.ylabel('WP-OBS')

#########################################################################
plt.subplot(2,4,1)
x3=wp150m[:,index_latitude1,index_longitude1]
y3=wp150_1.values


x3_clipped = np.clip(x3, 0, 1000)
y3_clipped = np.clip(y3, 0, 1000)

X3_design = np.vstack([np.ones(len(x3_clipped)), x3_clipped]).T
hat_matrix = X3_design @ np.linalg.inv(X3_design.T @ X3_design) @ X3_design.T
leverage = np.diag(hat_matrix)
mask = leverage < 0.5
x3_filtered = x3_clipped[mask]
y3_filtered = y3_clipped[mask]

x3_log = np.log(x3_filtered)
y3_log = np.log(y3_filtered)

m3_log, b3_log = np.polyfit(x3_log, y3_log, 1)

# Convert the slope and intercept back to the original scale
m3 = np.exp(m3_log)
b3 = np.exp(b3_log)

# Create scatter plot
plt.scatter(x3_filtered, y3_filtered, alpha=0.3, c=colors, cmap='rainbow')
plt.axline(xy1=(0, b3), slope=m3, color='black', label=f'$y = {m3:.2f}x {b3:+.2f}$')

plt.legend(fontsize='small')
plt.gca().set_title("F1_WS_TFCA")
plt.tick_params(axis='y', labelsize=5)  # Decrease font size for x-axis tick labels
plt.tick_params(axis='x', labelsize=0, length=0, width=0, labelbottom=False)
plt.xlabel('WP150M')
plt.ylabel('WP-OBS')
#################################################################################


plt.subplot(2,4,2)
x4=wp150m[:,index_latitude1,index_longitude1]
y4=wp150_2.values

x4_clipped = np.clip(x4, 0, 1000)
y4_clipped = np.clip(y4, 0, 1000)

X4_design = np.vstack([np.ones(len(x4_clipped)), x4_clipped]).T
hat_matrix = X4_design @ np.linalg.inv(X4_design.T @ X4_design) @ X4_design.T
leverage = np.diag(hat_matrix)
mask = leverage < 0.5
x4_filtered = x4_clipped[mask]
y4_filtered = y4_clipped[mask]

x4_log = np.log(x4_filtered)
y4_log = np.log(y4_filtered)

m4_log, b4_log = np.polyfit(x4_log, y4_log, 1)

# Convert the slope and intercept back to the original scale
m4 = np.exp(m4_log)
b4 = np.exp(b4_log)

# Create scatter plot
plt.scatter(x4_filtered, y4_filtered, alpha=0.3, c=colors, cmap='rainbow')


plt.axline(xy1=(0, b4), slope=m4, color='black', label=f'$y = {m4:.2f}x {b4:+.2f}$')
plt.legend(fontsize='small')
plt.gca().set_title("F2_WS_TFCA")
plt.tick_params(axis='x', labelsize=0, length=0, width=0, labelbottom=False)
plt.tick_params(axis='y', labelsize=0, length=0, width=0, labelleft=False)
plt.xlabel('WP150M')
plt.ylabel('WP-OBS')


###################################################################################
plt.subplot(2,4,5)
#WP80m=wp80m[:,index_latitude1,index_longitude1]
x5= wp80m[:,index_latitude1,index_longitude1]
y5=wp80_1.values

x5_clipped = np.clip(x5, 0, 1000)
y5_clipped = np.clip(y5, 0, 1000)

X5_design = np.vstack([np.ones(len(x5_clipped)), x5_clipped]).T
hat_matrix = X5_design @ np.linalg.inv(X5_design.T @ X5_design) @ X5_design.T
leverage = np.diag(hat_matrix)
mask = leverage < 0.5
x5_filtered = x5_clipped[mask]
y5_filtered = y5_clipped[mask]

x5_log = np.log(x5_filtered)
y5_log = np.log(y5_filtered)

m5_log, b5_log = np.polyfit(x5_log, y5_log, 1)

# Convert the slope and intercept back to the original scale
m5 = np.exp(m5_log)
b5 = np.exp(b5_log)

# Create scatter plot
plt.scatter(x5_filtered, y5_filtered, alpha=0.3, c=colors, cmap='rainbow')

plt.axline(xy1=(0, b5), slope=m5, color='black', label=f'$y = {m5:.2f}x {b5:+.2f}$')
plt.legend(fontsize='small')
plt.gca().set_title("F5_WS_TFCA")
plt.tick_params(axis='x', labelsize=5)  # Decrease font size for x-axis tick labels
plt.tick_params(axis='y', labelsize=5)
plt.xlabel('WP80M')
plt.ylabel('WP-OBS')

####################################################################################
plt.subplot(2,4,6)
x6=wp80m[:,index_latitude1,index_longitude1]
y6=wp80_2.values

x6_clipped = np.clip(x6, 0, 1000)
y6_clipped = np.clip(y6, 0, 1000)

X6_design = np.vstack([np.ones(len(x6_clipped)), x6_clipped]).T
hat_matrix = X6_design @ np.linalg.inv(X6_design.T @ X6_design) @ X6_design.T
leverage = np.diag(hat_matrix)
mask = leverage < 0.5
x6_filtered = x6_clipped[mask]
y6_filtered = y6_clipped[mask]

x6_log = np.log(x6_filtered)
y6_log = np.log(y6_filtered)

m6_log, b6_log = np.polyfit(x6_log, y6_log, 1)

# Convert the slope and intercept back to the original scale
m6 = np.exp(m6_log)
b6 = np.exp(b6_log)

# Create scatter plot
plt.scatter(x6_filtered, y6_filtered, alpha=0.3, c=colors, cmap='rainbow')
plt.axline(xy1=(0, b6), slope=m6, color='black', label=f'$y = {m6:.2f}x {b6:+.2f}$')
plt.legend(fontsize='small')
plt.gca().set_title("F6_WS_TFCA")
plt.tick_params(axis='x', labelsize=5)  # Decrease font size for x-axis tick labels
plt.tick_params(axis='y', labelsize=0, length=0, width=0, labelleft=False)
plt.xlabel('WP80M')
plt.ylabel('WP-OBS')


###################################################################################
plt.subplot(2,4,7)
x7=wp30m[:,index_latitude1,index_longitude1]
y7=wp30_1.values

x7_clipped = np.clip(x7, 0, 1000)
y7_clipped = np.clip(y7, 0, 1000)

X7_design = np.vstack([np.ones(len(x7_clipped)), x7_clipped]).T
hat_matrix = X7_design @ np.linalg.inv(X7_design.T @ X7_design) @ X7_design.T
leverage = np.diag(hat_matrix)
mask = leverage < 0.5
x7_filtered = x7_clipped[mask]
y7_filtered = y7_clipped[mask]

x7_log = np.log(x7_filtered)
y7_log = np.log(y7_filtered)

m7_log, b7_log = np.polyfit(x7_log, y7_log, 1)

# Convert the slope and intercept back to the original scale
m7 = np.exp(m7_log)
b7 = np.exp(b7_log)

# Create scatter plot
plt.scatter(x7_filtered, y7_filtered, alpha=0.3, c=colors, cmap='rainbow')
plt.axline(xy1=(0, b7), slope=m7, color='black', label=f'$y = {m7:.2f}x {b7:+.2f}$')
plt.legend(fontsize='small')
plt.gca().set_title("F7_WS_TFCA")
plt.tick_params(axis='x', labelsize=5)  # Decrease font size for x-axis tick labels
plt.tick_params(axis='y', labelsize=0, length=0, width=0, labelleft=False)
plt.xlabel('WP30M')
plt.ylabel('WP-OBS')

#############################################################################3
im=plt.subplot(2,4,8)
x8=wp30m[:,index_latitude1,index_longitude1]
y8=wp30_2.values

x8_clipped = np.clip(x8, 0, 1000)
y8_clipped = np.clip(y8, 0, 1000)

X8_design = np.vstack([np.ones(len(x8_clipped)), x8_clipped]).T
hat_matrix = X8_design @ np.linalg.inv(X8_design.T @ X8_design) @ X8_design.T
leverage = np.diag(hat_matrix)
mask = leverage < 0.5
x8_filtered = x8_clipped[mask]
y8_filtered = y8_clipped[mask]

x8_log = np.log(x8_filtered)
y8_log = np.log(y8_filtered)

m8_log, b8_log = np.polyfit(x8_log, y8_log, 1)

# Convert the slope and intercept back to the original scale
m8 = np.exp(m8_log)
b8 = np.exp(b8_log)

# Create scatter plot
plt.scatter(x8_filtered, y8_filtered, alpha=0.3, c=colors, cmap='rainbow')

plt.axline(xy1=(0, b8), slope=m8, color='black', label=f'$y = {m8:.2f}x {b8:+.2f}$')
plt.gca().set_title("F8_WS_TFCA")
plt.tick_params(axis='x', labelsize=5)  # Decrease font size for x-axis tick labels
plt.legend(fontsize='small')
plt.tick_params(axis='y', labelsize=0, length=0, width=0, labelleft=False)
plt.xlabel('WP30M')
plt.ylabel('WP-OBS')


color_ax = fig.add_axes([0.92, 0.1, 0.02, 0.7])  # [left, bottom, width, height]
#color_ax.set_title('Color Bar')
plt.colorbar(cmap='rainbow',cax=color_ax)

#plt.colorbar(im, orientation='vertical', shrink=0.9, pad=0.03, extend='both')

fig.suptitle('AlBada_1', y=0.97, fontsize=25)


#fig.text(0.5, 0.04, 'Shared X Label', ha='center', fontsize=12)
#fig.text(0.04, 0.5, 'Shared Y Label', va='center', rotation='vertical', fontsize=12)

#plt.colorbar()
#plt.xlabel('Wp100m')
#plt.ylabel('WP-OBS')



plt.savefig('ScatterPlot_AlBada_REMOVED.png',format='png',dpi=300)
quit()

