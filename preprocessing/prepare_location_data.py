"""
Prepares location metadata for geocell creation.
Country information is missing from inaturalist datasets, so 
spatial join is performed using admin 0 data.

Outputs a dataset with columns: id, lat, lon, country_name
"""

import cudf
import cuspatial
import geopandas as gpd 
import pandas as pd 
import json 

from typing import Tuple
from tqdm import tqdm

from config import ADMIN_0_PATH, ADMIN_1_PATH, ADMIN_2_PATH, LOC_PATH, CRS

ADMIN_NAMES = ['GID_0', 'GID_1', 'GID_2']
FINAL_COLS = ['id', 'country_code', 'lat', 'lon', 'gid_0', 'gid_1', 'gid_2']

class GeoDataset(): 
    def __init__(self, file): 
        with open(file, 'r') as f: 
            data = json.load(f)
        df = pd.DataFrame(data)

        self.gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.lon, df.lat), 
            crs=CRS
        )  
    
    def prepare(self): 
        self.gdf = self.gdf.dropna(subset=['lat', 'lon'])
        self.gdf = self._load_boundary_ids(self.gdf)
        self.gdf = self.gdf.copy()[FINAL_COLS]
        self.gdf.to_csv(LOC_PATH)

    def _load_boundary_ids(self, gdf) -> Tuple[gpd.GeoDataFrame]:
        print("LOADING ADMIN 0 BOUNDARIES")
        id_name = "gid_0"

        admin_0 = gpd.read_file(ADMIN_0_PATH).set_crs(crs=CRS)
        admin_0['geometry'] = admin_0['geometry'].buffer(0)

        # Perform spatial join
        gdf = gpd.sjoin(gdf, admin_0[['GID_0', 'geometry']], how='left', predicate='within')
        gdf = gdf.rename(columns={'GID_0': 'country_code', 'index_right': id_name})
        
        # Fill by nearest known value
        gdf = self._apply_nearest_match(gdf, id_name)
        gdf[id_name] = gdf[id_name].astype(int)
        
        # Use gid_0 to fill missing country_code values using admin_0 as lookup
        gid_to_code = admin_0['GID_0'].to_dict()
        idx_mappings = gdf[id_name].map(gid_to_code)
        gdf['country_code'] = gdf['country_code'].fillna(idx_mappings)

        print("LOADING ADMIN 1 BOUNDARIES")
        id_name = "gid_1"

        admin_1 = gpd.read_file(ADMIN_1_PATH).set_crs(crs=CRS)
        admin_1['geometry'] = admin_1['geometry'].buffer(0)

        # Perform spatial join
        gdf = gpd.sjoin(gdf, admin_1[['geometry']], how='left', predicate='within')
        gdf = gdf.rename(columns={'index_right': id_name})
        
        # Fill by nearest known value
        gdf = self._apply_nearest_match(gdf, id_name)
        gdf[id_name] = gdf[id_name].astype(int)

        print("LOADING ADMIN 2 BOUNDARIES")
        id_name = "gid_2"

        admin_2 = gpd.read_file(ADMIN_2_PATH).set_crs(crs=CRS)
        admin_2['geometry'] = admin_2['geometry'].buffer(0)

        # Perform spatial join
        gdf = gpd.sjoin(gdf, admin_2[['geometry']], how='left', predicate='within')
        gdf = gdf.rename(columns={'index_right': id_name})

        # Fill by nearest known value
        gdf = self._apply_nearest_match(gdf, id_name)
        gdf[id_name] = gdf[id_name].astype(int)

        print("LOADED ALL BOUNDARY IDS") 

        return gdf 
    
    def _apply_nearest_match(self, gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
        """Fill NaN values in a column based on the closest geographic match.

        Args:
            df (gpd.GeoDataFrame): Dataframe to fill NaN values in.
            col (str): Column to substitute NaNs in.

        Returns:
            gpd.GeoDataFrame: Dataframe with all NaNs replaced.
        """

        missing = gdf[gdf[col].isnull()].copy()
        not_missing = gdf[gdf[col].notnull()].copy()

        # nearest is a tuple of (missing_idx, not_missing_idx)
        # We align these to get corresponding values
        nearest = not_missing.sindex.nearest(missing.geometry, return_all=False)

        _, match_indices = nearest
        values = not_missing.iloc[match_indices][col].values

        # Fill the missing values in the original DataFrame
        gdf.loc[missing.index, col] = values
        
        return gdf

if __name__ == '__main__': 
    raw_loc_path = 'data/inaturalist_2017/locations/train2017_locations.json'
    dataset = GeoDataset(raw_loc_path)
    dataset.prepare()




