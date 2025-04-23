import json
import pandas as pd
import geopandas as gpd
import warnings; warnings.filterwarnings(action='ignore')

from typing import Tuple
from config import ADMIN_0_PATH, ADMIN_1_PATH, ADMIN_2_PATH, LOC_PATH, CRS

ADMIN_NAMES = ['GID_0', 'GID_1', 'GID_2']

class GeoDataset:
    def __init__(self, location_file: str, image_metadata_file: str):
        # Load raw location JSON
        with open(location_file, 'r') as f:
            loc_data = json.load(f)

        self.location_df = pd.DataFrame(loc_data)
        self.gdf = gpd.GeoDataFrame(
            self.location_df,
            geometry=gpd.points_from_xy(self.location_df.lon, self.location_df.lat),
            crs=CRS
        )

        # Load image metadata JSON
        with open(image_metadata_file, 'r') as f:
            image_data = json.load(f)

        # Extract relevant components
        self.images_df = pd.DataFrame(image_data['images'])[['id', 'file_name', 'height', 'width']]
        self.annotations_df = pd.DataFrame(image_data['annotations'])[['image_id', 'category_id']]
        self.categories_df = pd.DataFrame(image_data['categories'])[['id', 'name', 'supercategory']]

        # Rename for clarity
        self.categories_df = self.categories_df.rename(columns={'id': 'category_id', 'name': 'species'})
        self.location_df = self.location_df.rename(columns={'id': 'image_id'})

    def prepare(self) -> pd.DataFrame:
        gdf_cols = ['id', 'country_code', 'lat', 'lon', 'gid_0', 'gid_1', 'gid_2']

        self.gdf = self.gdf.dropna(subset=['lat', 'lon'])
        self.gdf = self._load_boundary_ids(self.gdf)
        self.gdf = self.gdf.copy()[gdf_cols]

        return self._merge_metadata()

    def _merge_metadata(self) -> pd.DataFrame:
        # Merge image, annotation, category
        merged = self.images_df.merge(self.annotations_df, left_on='id', right_on='image_id')
        merged = merged.merge(self.categories_df, on='category_id', how='left')

        # Join with location + admin codes
        location_df = self.gdf.rename(columns={'id': 'image_id'})  # ensure match
        merged = merged.merge(location_df, on='image_id', how='left')

        # Drop extras
        merged = merged.drop(columns=['id', 'category_id'])

        # Reorder columns
        merged = merged[[
            'image_id', 'file_name', 'height', 'width',
            'species', 'supercategory', 'lat', 'lon',
            'country_code', 'gid_0', 'gid_1', 'gid_2'
        ]]

        return merged

    def _load_boundary_ids(self, gdf) -> Tuple[gpd.GeoDataFrame]:
        print("LOADING ADMIN 0 BOUNDARIES")
        id_name = "gid_0"

        admin_0 = gpd.read_file(ADMIN_0_PATH)
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

        admin_1 = gpd.read_file(ADMIN_1_PATH)
        admin_1['geometry'] = admin_1['geometry'].buffer(0)

        # Perform spatial join
        gdf = gpd.sjoin(gdf, admin_1[['geometry']], how='left', predicate='within')
        gdf = gdf.rename(columns={'index_right': id_name})
        
        # Fill by nearest known value
        gdf = self._apply_nearest_match(gdf, id_name)
        gdf[id_name] = gdf[id_name].astype(int)

        print("LOADING ADMIN 2 BOUNDARIES")
        id_name = "gid_2"

        admin_2 = gpd.read_file(ADMIN_2_PATH)
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
    location_file = 'data/inaturalist_2017/raw/locations/train2017_locations.json'
    image_file = 'data/inaturalist_2017/raw/image_metadata/train2017.json'

    dataset = GeoDataset(location_file, image_file)
    processed_df = dataset.prepare()
    processed_df.to_csv('data/inaturalist_2017/processed/metadata.csv', index=False)
