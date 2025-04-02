"""
Prepares location metadata for geocell creation.
Country information is missing from inaturalist datasets, so 
spatial join is performed using admin 0 data.

Outputs a dataset with columns: id, lat, lon, country_name
"""

import geopandas as gpd 
import pandas as pd 
import json 

from config import ADMIN_0_PATH

def save_data(location_path, adm_0_path, output_path): 
    adm_0_gdf = gpd.read_file(adm_0_path)

    with open(location_path, 'r') as f: 
        data = json.load(f)

    loc_df = pd.DataFrame(data)
    loc_df = loc_df.dropna(subset=['lat', 'lon'])

    # Convert loc_df to gdf
    gdf = gpd.GeoDataFrame(
        loc_df, 
        geometry=gpd.points_from_xy(loc_df['lon'], loc_df['lat']), 
        crs="EPSG:4326"
    )

    gdf = gpd.sjoin(gdf, adm_0_gdf[['COUNTRY', 'geometry']], how='left', predicate='within')
    df = gdf[['id', 'lat', 'lon', 'COUNTRY']].rename(columns={'lon': 'lng', 'COUNTRY': 'country_name'})

    df.to_csv(output_path , index=False)
    
if __name__ == '__main__': 
    location_path = 'data/inaturalist_2017/locations/train2017_locations.json'
    output_path = 'data/training/locations/locations.csv'

    save_data(location_path, ADMIN_0_PATH, output_path)





