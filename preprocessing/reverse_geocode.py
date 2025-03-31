"""
Prepares location metadata for geocell creation.
Country information is missing from inaturalist datasets, so 
spatial join is performed using admin 0 data.
"""

import geopandas as gpd 
import pandas as pd 
import json 

from config import ADMIN_0_PATH

LOC_PATH = 'data/inaturalist_2017/locations/train2017_locations.json'
OUTPUT_PATH = 'data/training/locations/locations.csv'

with open(LOC_PATH, 'r') as f: 
    data = json.load(f)

loc_df = pd.DataFrame(data)
country_df = gpd.read_file(ADMIN_0_PATH)[['geometry', 'COUNTRY']]

# Convert loc_df to gdf
gdf = gpd.GeoDataFrame(
    loc_df, 
    geometry=gpd.points_from_xy(loc_df['lon'], loc_df['lat']), 
    crs="EPSG:4326"
)

# Spatial join (is this lat/lon point inside this polygon?)
gdf = gpd.sjoin(gdf, country_df, how="left", predicate="within")
df = gdf[['id', 'lat', 'lon', 'COUNTRY']]

# For consistency with PIGEON paper
df = df.rename(columns={
    'lon': 'lng', 
    'COUNTRY': 'country_name'
})

df.to_csv(OUTPUT_PATH)





