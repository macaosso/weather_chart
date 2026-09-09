import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.interpolate import griddata

matplotlib.use('Agg')
os.makedirs('rain_wind_maps', exist_ok=True)

# Grid covering Southern China & surrounding seas (100°E to 123°E, 8°N to 32°N)
lons_grid = np.arange(100, 124, 1.0)
lats_grid = np.arange(8, 33, 1.0)
lon_mesh, lat_mesh = np.meshgrid(lons_grid, lats_grid)
flat_lons = lon_mesh.flatten()
flat_lats = lat_mesh.flatten()

lat_str = ','.join(map(str, flat_lats))
lon_str = ','.join(map(str, flat_lons))

# 16 days forecast covers up to 384 hours (supports up to 360H)
url = f'https://api.open-meteo.com/v1/forecast?latitude={lat_str}&longitude={lon_str}&hourly=precipitation,wind_speed_10m,wind_direction_10m&forecast_days=16'
res = requests.get(url).json()
if isinstance(res, dict):
  if res.get('error'):
    raise RuntimeError(f"Open-Meteo API Error: {res.get('reason')}")
  res = [res]

interp_lon, interp_lat = np.meshgrid(
    np.linspace(100, 123, 220), np.linspace(8, 32, 160)
)

# Generate frames from 0H to 144H every 6 hours (adjust upper range up to 361 for full 360h)
for step in range(0, 145, 6):
  precips, wspeeds, wdirs = [], [], []
  for loc in res:
    hourly = loc.get('hourly', {})
    p_list = hourly.get('precipitation', [0] * 385)
    s_list = hourly.get('wind_speed_10m', [5] * 385)
    d_list = hourly.get('wind_direction_10m', [180] * 385)

    precips.append(p_list[step] if step < len(p_list) else 0.0)
    wspeeds.append(s_list[step] if step < len(s_list) else 5.0)
    wdirs.append(d_list[step] if step < len(d_list) else 180.0)

  precips = np.array(precips)
  wspeeds = np.array(wspeeds)
  wdirs = np.array(wdirs)

  grid_p = griddata(
      (flat_lons, flat_lats), precips, (interp_lon, interp_lat), method='cubic'
  )
  grid_p = np.nan_to_num(grid_p, nan=0.0)
  grid_p = np.clip(grid_p, 0, 300)

  # Subgrid for wind vectors
  sub_lons = np.arange(101, 123, 1.5)
  sub_lats = np.arange(9, 32, 1.5)
  sub_lon_mesh, sub_lat_mesh = np.meshgrid(sub_lons, sub_lats)

  sub_ws = griddata(
      (flat_lons, flat_lats),
      wspeeds,
      (sub_lon_mesh, sub_lat_mesh),
      method='nearest',
  )
  sub_wd = griddata(
      (flat_lons, flat_lats),
      wdirs,
      (sub_lon_mesh, sub_lat_mesh),
      method='nearest',
  )

  rad = np.radians(270 - sub_wd)
  u = sub_ws * np.cos(rad)
  v = sub_ws * np.sin(rad)

  fig, ax = plt.subplots(
      figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()}
  )
  ax.set_extent([100, 123, 8, 32], crs=ccrs.PlateCarree())

  ax.add_feature(cfeature.LAND, facecolor='#1e293b')
  ax.add_feature(cfeature.OCEAN, facecolor='#0f172a')
  ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#38bdf8')
  ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.4, edgecolor='#38bdf8')

  levels = [0.1, 4, 10, 20, 30, 50, 80, 120, 200]
  cf = ax.contourf(
      interp_lon,
      interp_lat,
      grid_p,
      levels=levels,
      cmap='YlGnBu',
      alpha=0.75,
      extend='max',
      transform=ccrs.PlateCarree(),
  )

  ax.quiver(
      sub_lon_mesh,
      sub_lat_mesh,
      u,
      v,
      sub_ws,
      cmap='cool',
      scale=450,
      width=0.003,
      transform=ccrs.PlateCarree(),
      alpha=0.9,
  )

  cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.03, shrink=0.7)
  cbar.set_label('Precipitation (mm/6h)', color='white')
  cbar.ax.yaxis.set_tick_params(color='white')
  plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

  ax.set_title(
      'Wind & Rain Forecast',
      fontsize=12,
      weight='bold',
      color='white',
      loc='left',
  )
  ax.set_title(
      f'Forecast Step: +{step:03d}h', fontsize=11, color='white', loc='right'
  )

  fig.patch.set_facecolor('#0b0f19')
  ax.set_facecolor('#0b0f19')

  filename = os.path.join('rain_wind_maps', f'rain_wind_{step}.png')
  plt.savefig(
      filename, format='png', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor()
  )
  plt.close(fig)
