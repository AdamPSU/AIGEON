"""
Prepares location metadata for geocell creation.
Country information is missing from inaturalist datasets, so 
spatial join is performed using admin 0 data.

Outputs a dataset with columns: id, lat, lon, country_name
"""

import geopandas as gpd 
import pandas as pd 
import json 

from config import ADMIN_0_PATH, LOC_PATH, CRS


FINAL_COLS = ['id', 'lat', 'lon', 'GID_0']

def prepare_data(location_path, adm_0_path, output_path): 
    """
    Adds country code column to location metadata.
    """

    adm_0_gdf = gpd.read_file(adm_0_path)

    with open(location_path, 'r') as f: 
        data = json.load(f)

    loc_df = pd.DataFrame(data)
    loc_df = loc_df.dropna(subset=['lat', 'lon'])

    # Convert loc_df to gdf
    gdf = gpd.GeoDataFrame(
        loc_df, 
        geometry=gpd.points_from_xy(loc_df['lon'], loc_df['lat']),
        crs=CRS
    )

    # Spatial join to obtain GID_0 column
    gdf = gpd.sjoin(gdf, adm_0_gdf[['GID_0', 'geometry']], how='left', predicate='within')
    df = gdf[FINAL_COLS].rename(columns={'GID_0': 'gid_0'})

    df.to_csv(output_path , index=False)
    
if __name__ == '__main__': 
    raw_loc_path = 'data/inaturalist_2017/locations/train2017_locations.json'
    prepare_data(raw_loc_path, ADMIN_0_PATH, LOC_PATH)





