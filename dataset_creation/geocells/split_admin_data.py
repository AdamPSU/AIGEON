"""
Takes GADM package and splits into their respective admin boundaries. Following the
examples in the PIGEOTTO paper, the only admin boundaries necessary are at the country, 
state, and county level.
"""

import geopandas as gpd

FILE_PATH = 'data/geocells/gadm_410-levels.gpkg'

for i in range(3): 
    gdf = gpd.read_file(FILE_PATH, layer=i)
    gdf.to_file(f"data/geocells/adm_{i}_boundaries.gpkg", driver="GPKG")

print("Successfully created admin boundaries!")
