import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.interpolate import griddata

matplotlib.use('Agg')
os.makedirs('temp_maps', exist_ok=True)

# Define grid covering China region (95°E to 125°E, 15°N to 38°N)
lons_grid = np.arange(95, 126, 1.0)
lats_grid = np.arange(15, 39, 1.0)
lon_mesh, lat_mesh = np.meshgrid(lons_grid, lats_grid)
flat_lons = lon_mesh.flatten()
flat_lats = lat_mesh.flatten()

lat_str = ','.join(map(str, flat_lats))
lon_str = ','.join(map(str, flat_lons))

url_grid = f'https://api.open-meteo.com/v1/forecast?latitude={lat_str}&longitude={lon_str}&hourly=temperature_2m&forecast_days=7'

res_grid = requests.get(url_grid).json()
if isinstance(res_grid, dict):
  if res_grid.get('error'):
    raise RuntimeError(f"Open-Meteo API Error: {res_grid.get('reason')}")
  res_grid = [res_grid]

interp_lon, interp_lat = np.meshgrid(
    np.linspace(95, 125, 250), np.linspace(15, 38, 250)
)

# Temperature color levels (°C)
levels = np.arange(-10, 42, 2)
cmap = plt.get_pylab_colorMap() if hasattr(plt, 'get_pylab_colorMap') else 'jet'

for step in range(0, 145, 6):
  temps = [
      loc.get('hourly', {}).get('temperature_2m', [20])[step] or 20.0
      for loc in res_grid
  ]
  temps = np.array(temps)

  grid_temp_2d = griddata(
      (flat_lons, flat_lats), temps, (interp_lon, interp_lat), method='cubic'
  )

  fig, ax = plt.subplots(
      figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()}
  )
  ax.set_extent([95, 125, 15, 38], crs=ccrs.PlateCarree())

  ax.add_feature(cfeature.LAND, facecolor='#f4f8f3')
  ax.add_feature(cfeature.OCEAN, facecolor='#e0f0ff')
  ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
  ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
  ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)

  cf = ax.contourf(
      interp_lon,
      interp_lat,
      grid_temp_2d,
      levels=levels,
      cmap='turbo',
      extend='both',
      transform=ccrs.PlateCarree(),
  )
  cs = ax.contour(
      interp_lon,
      interp_lat,
      grid_temp_2d,
      levels=levels[::2],
      colors='black',
      linewidths=0.4,
      alpha=0.5,
  )
  ax.clabel(cs, inline=True, fontsize=7, fmt='%d')

  cbar = plt.colorbar(
      cf, ax=ax, orientation='vertical', pad=0.03, shrink=0.85
  )
  cbar.set_label('2m Temperature (°C)')

  plt.title(
      'AIFS (0.25°) 2m Temperature (°C)', fontsize=11, weight='bold', loc='left'
  )
  plt.title(f'Forecast Step: +{step:03d}h', fontsize=10, loc='right')

  filename = os.path.join('temp_maps', f'temp_{step}.png')
  plt.savefig(filename, format='png', bbox_inches='tight', dpi=150)
  plt.close(fig)
