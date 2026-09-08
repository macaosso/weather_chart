import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.interpolate import griddata

matplotlib.use('Agg')

# Ensure output directory exists
os.makedirs('ECwind', exist_ok=True)

# 1. Setup grid matching extent: (105, 130, 10, 30)
lons_grid = np.arange(105, 131, 2.0)
lats_grid = np.arange(10, 31, 2.0)
lon_mesh, lat_mesh = np.meshgrid(lons_grid, lats_grid)
flat_lons = lon_mesh.flatten()
flat_lats = lat_mesh.flatten()

lat_str = ','.join(map(str, flat_lats))
lon_str = ','.join(map(str, flat_lons))

# 2. Fetch ECMWF wind speed and direction forecast (10m) from Open-Meteo
url_grid = f'https://api.open-meteo.com/v1/forecast?latitude={lat_str}&longitude={lon_str}&hourly=wind_speed_10m,wind_direction_10m&models=ecmwf_ifs025&forecast_days=7'

res_grid = requests.get(url_grid).json()
if isinstance(res_grid, dict):
  if res_grid.get('error'):
    raise RuntimeError(f"Open-Meteo API Error: {res_grid.get('reason')}")
  res_grid = [res_grid]

# 3. Setup interpolation mesh and custom gradual colormap
interp_lon, interp_lat = np.meshgrid(
    np.linspace(105, 130, 200), np.linspace(10, 30, 200)
)

# Custom wind speed thresholds and corresponding gradual colors (7 colors for 7 levels)
levels = [0, 20, 40, 60, 90, 120, 300]
colors = [
    'white',  # 0 km/h
    '#98fb98',  # Light Green (20 km/h)
    '#87ceeb',  # Light Blue (40 km/h)
    '#ffdab9',  # Light Orange (60 km/h)
    '#f08080',  # Light Red (90 km/h)
    '#dda0dd',  # Light Purple (120 km/h)
    '#8b008b',  # Dark Purple (300 km/h - Max end point)
]

# Create a continuous gradual colormap matching the specified thresholds
norm_levels = (np.array(levels) - levels[0]) / (levels[-1] - levels[0])
color_list = list(zip(norm_levels, colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'gradual_wind', color_list
)
norm = matplotlib.colors.BoundaryNorm(levels, cmap.N, extend='max')

# 4. Loop from 0H to 144H every 6H
for step in range(0, 145, 6):
  wind_speeds = np.array(
      [
          loc.get('hourly', {}).get('wind_speed_10m', [0])[step] or 0.0
          for loc in res_grid
      ]
  )
  wind_dirs = np.array(
      [
          loc.get('hourly', {}).get('wind_direction_10m', [0])[step] or 0.0
          for loc in res_grid
      ]
  )

  # Convert wind speed and meteorological direction to U and V components
  rad = np.radians(wind_dirs)
  u_wind = -wind_speeds * np.sin(rad)
  v_wind = -wind_speeds * np.cos(rad)

  u_2d = u_wind.reshape(lon_mesh.shape)
  v_2d = v_wind.reshape(lon_mesh.shape)

  grid_wind_2d = griddata(
      (flat_lons, flat_lats),
      wind_speeds,
      (interp_lon, interp_lat),
      method='cubic',
  )
  grid_wind_2d = np.clip(grid_wind_2d, 0, 300)

  # Render Chart
  fig, ax = plt.subplots(
      figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()}
  )
  ax.set_extent([105, 130, 10, 30], crs=ccrs.PlateCarree())

  ax.add_feature(cfeature.LAND, facecolor='#f4f8f3')
  ax.add_feature(cfeature.OCEAN, facecolor='#e0f0ff')
  ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
  ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
  ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)

  cf = ax.contourf(
      interp_lon,
      interp_lat,
      grid_wind_2d,
      levels=levels,
      cmap=cmap,
      norm=norm,
      alpha=0.85,
      transform=ccrs.PlateCarree(),
      extend='max',
  )

  # Overlay Wind Barbs
  ax.barbs(
      lon_mesh,
      lat_mesh,
      u_2d,
      v_2d,
      transform=ccrs.PlateCarree(),
      length=5.5,
      color='k',
      linewidth=0.5,
      alpha=0.7,
  )

  cbar = plt.colorbar(
      cf, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7
  )
  cbar.set_label('10m Wind Speed (km/h)')
  cbar.set_ticks(levels[:-1])

  plt.title(
      'ECMWF 10m Wind Speed & Barbs (Open-Meteo)',
      fontsize=11,
      weight='bold',
      loc='left',
  )
  plt.title(f'Forecast Step: +{step}H', fontsize=10, loc='right')

  filename = os.path.join('ECwind', f'ECwind{step}.png')
  plt.savefig(filename, format='png', bbox_inches='tight', dpi=150)
  plt.close(fig)
  print(f'Saved: {filename}')
