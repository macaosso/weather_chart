import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter, minimum_filter

matplotlib.use('Agg')

# 1. Setup a high-density grid matching the requested extent: (90, 146, 8, 42)
lons_grid = np.arange(90, 148, 2.5)
lats_grid = np.arange(8, 44, 2.5)
lon_mesh, lat_mesh = np.meshgrid(lons_grid, lats_grid)
flat_lons = lon_mesh.flatten()
flat_lats = lat_mesh.flatten()

lat_str = ','.join(map(str, flat_lats))
lon_str = ','.join(map(str, flat_lons))

# Include past_days and forecast_days to safely cover -18H to +24H window
url_grid = f'https://api.open-meteo.com/v1/forecast?latitude={lat_str}&longitude={lon_str}&hourly=pressure_msl&past_days=2&forecast_days=2'

res_grid = requests.get(url_grid).json()
if isinstance(res_grid, dict):
  if res_grid.get('error'):
    raise RuntimeError(f"Open-Meteo API Error: {res_grid.get('reason')}")
  res_grid = [res_grid]

# Extract the shared time array from the first location
times = res_grid[0].get('hourly', {}).get('time', [])

# Define offsets and corresponding output filenames
offsets = {
    'B18': (-18, 'B18map90146842.png'),
    'B12': (-12, 'B12map90146842.png'),
    'B6': (-6, 'B6map90146842.png'),
    '0': (0, '0map90146842.png'),
    'A6': (6, 'A6map90146842.png'),
    'A12': (12, 'A12map90146842.png'),
    'A18': (18, 'A18map90146842.png'),
    'A24': (24, 'A24map90146842.png'),
}

now_utc = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
interp_lon, interp_lat = np.meshgrid(
    np.linspace(90, 146, 250), np.linspace(8, 42, 250)
)


def find_extrema_coords(grid_z, grid_x, grid_y, mode='max', n=2):
  if mode == 'max':
    local_mask = grid_z == maximum_filter(
        grid_z, size=15, mode='constant', cval=-9999
    )
  else:
    local_mask = grid_z == minimum_filter(
        grid_z, size=15, mode='constant', cval=9999
    )

  y_indices, x_indices = np.where(local_mask)
  values = grid_z[y_indices, x_indices]
  sorted_idx = (
      np.argsort(values)[::-1] if mode == 'max' else np.argsort(values)
  )

  points = []
  for idx in sorted_idx:
    yi, xi = y_indices[idx], x_indices[idx]
    points.append((grid_x[yi, xi], grid_y[yi, xi], grid_z[yi, xi]))
    if len(points) >= n:
      break
  return points


for label, (offset_hrs, filename) in offsets.items():
  target_time = now_utc + datetime.timedelta(hours=offset_hrs)
  target_str = target_time.strftime('%Y-%m-%dT%H:00')

  try:
    time_idx = times.index(target_str)
  except ValueError:
    print(f'Warning: Target time {target_str} not found in API response.')
    continue

  # Extract pressures for the specific time index across all grid locations
  grid_pressures = [
      loc.get('hourly', {}).get('pressure_msl', [])[time_idx] or 1013.0
      for loc in res_grid
  ]
  grid_pressures = np.array(grid_pressures)

  # Interpolate pressure onto the 2D mesh
  grid_pressure_2d = griddata(
      (flat_lons, flat_lats),
      grid_pressures,
      (interp_lon, interp_lat),
      method='cubic',
  )

  highs = find_extrema_coords(
      grid_pressure_2d, interp_lon, interp_lat, mode='max', n=2
  )
  lows = find_extrema_coords(
      grid_pressure_2d, interp_lon, interp_lat, mode='min', n=2
  )

  # Render Weather Chart
  fig, ax = plt.subplots(
      figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()}
  )
  ax.set_extent([90, 146, 8, 42], crs=ccrs.PlateCarree())

  ax.add_feature(cfeature.LAND, facecolor='#f4f8f3')
  ax.add_feature(cfeature.OCEAN, facecolor='#e0f0ff')
  ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
  ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
  ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)

  min_p = np.floor(np.nanmin(grid_pressure_2d) / 2) * 2
  max_p = np.ceil(np.nanmax(grid_pressure_2d) / 2) * 2
  contour_levels = np.arange(min_p, max_p + 2, 2)

  cs = ax.contour(
      interp_lon,
      interp_lat,
      grid_pressure_2d,
      levels=contour_levels,
      colors='black',
      linewidths=0.7,
  )
  ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

  for h in highs:
    ax.text(
        h[0],
        h[1],
        'H',
        color='red',
        fontsize=14,
        weight='bold',
        ha='center',
        va='center',
        transform=ccrs.PlateCarree(),
    )
    ax.text(
        h[0],
        h[1] - 0.8,
        f'{h[2]:.0f}',
        color='red',
        fontsize=8,
        weight='bold',
        ha='center',
        transform=ccrs.PlateCarree(),
    )

  for l in lows:
    ax.text(
        l[0],
        l[1],
        'L',
        color='blue',
        fontsize=14,
        weight='bold',
        ha='center',
        va='center',
        transform=ccrs.PlateCarree(),
    )
    ax.text(
        l[0],
        l[1] - 0.8,
        f'{l[2]:.0f}',
        color='blue',
        fontsize=8,
        weight='bold',
        ha='center',
        transform=ccrs.PlateCarree(),
    )

  valid_time_str = target_time.strftime('%Y-%m-%d %H:00 UTC')
  plt.title(
      'Regional Synoptic Weather Chart (Open-Meteo)',
      fontsize=11,
      weight='bold',
      loc='left',
  )
  plt.title(f'VALID: {valid_time_str}', fontsize=10, loc='right')

  plt.savefig(filename, format='png', bbox_inches='tight', dpi=150)
  plt.close(fig)
  print(f'Successfully generated and saved: {filename} ({valid_time_str})')
