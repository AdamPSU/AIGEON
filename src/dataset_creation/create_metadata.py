import json
import rasterio
import pandas as pd
import numpy as np 
import geopandas as gpd
import warnings; warnings.filterwarnings(action='ignore')

from rasterio.transform import rowcol
from typing import Tuple

from config import ADMIN_0_PATH, ADMIN_1_PATH, ADMIN_2_PATH, LOC_PATH, CRS

CLIMATE = np.array([
    'Unknown', 
    "Tropical, rainforest",
    "Tropical, monsoon",
    "Tropical, savannah",
    "Arid, desert, hot",
    "Arid, desert, cold",
    "Arid, steppe, hot",
    "Arid, steppe, cold",
    "Temperate, dry summer, hot summer",
    "Temperate, dry summer, warm summer",
    "Temperate, dry summer, cold summer",
    "Temperate, dry winter, hot summer",
    "Temperate, dry winter, warm summer",
    "Temperate, dry winter, cold summer",
    "Temperate, no dry season, hot summer",
    "Temperate, no dry season, warm summer",
    "Temperate, no dry season, cold summer",
    "Cold, dry summer, hot summer",
    "Cold, dry summer, warm summer",
    "Cold, dry summer, cold summer",
    "Cold, dry summer, very cold winter",
    "Cold, dry winter, hot summer",
    "Cold, dry winter, warm summer",
    "Cold, dry winter, cold summer",
    "Cold, dry winter, very cold winter",
    "Cold, no dry season, hot summer",
    "Cold, no dry season, warm summer",
    "Cold, no dry season, cold summer",
    "Cold, no dry season, very cold winter",
    "Polar, tundra",
    "Polar, frost"
])

def month_to_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'fall'

class GeoDataset:
    def __init__(self, location_file: str, image_metadata_file: str):
        # Load raw location JSON
        with open(location_file, 'r') as f:
            loc_data = json.load(f)

        # Load image metadata JSON
        with open(image_metadata_file, 'r') as f:
            image_data = json.load(f)

        # Extract relevant components
        self.location_df = pd.DataFrame(loc_data)
        self.images_df = pd.DataFrame(image_data['images'])[['id', 'file_name', 'height', 'width']]
        self.annotations_df = pd.DataFrame(image_data['annotations'])[['image_id', 'category_id']]
        self.categories_df = pd.DataFrame(image_data['categories'])[['id', 'name', 'supercategory']]

        # Rename for clarity
        self.categories_df = self.categories_df.rename(columns={'id': 'category_id', 'name': 'species'})
        self.location_df = self.location_df.rename(columns={'id': 'image_id'})

        self.tif_path = 'data/climate/Beck_KG_V1_present_0p5.tif'

    def create(self) -> pd.DataFrame:
        self.gdf = self._merge_metadata()
        self.gdf = self._load_boundary_ids(
            self.gdf, 
            admin_paths=[ADMIN_0_PATH, ADMIN_1_PATH, ADMIN_2_PATH]
        )
        self.gdf  = self._load_climate(self.tif_path)

        # Extract season from date 
        self.gdf['date'] = pd.to_datetime(self.gdf['date'], errors='coerce')
        self.gdf['season'] = self.gdf['date'].dt.month.map(month_to_season)

        final_cols = [
            'image_id',
            'file_name',
            'height',
            'width',
            'species',
            'supercategory',
            'climate',
            'season', 
            'lat',
            'lon',
            'gid_0',
            'country',
            'gid_1',
            'state', 
            'gid_2',
            'district',  
        ]

        self.gdf = self.gdf[final_cols].copy() 
        print("COMPLETE.")
        
        return self.gdf

    def _merge_metadata(self) -> pd.DataFrame:
        # Merge image, annotation, category
        merged = self.images_df.merge(self.annotations_df, left_on='id', right_on='image_id')
        merged = merged.merge(self.categories_df, on='category_id', how='left')

        # Join with location + admin codes
        location_df = self.location_df.rename(columns={'id': 'image_id'})  # ensure match
        merged = merged.merge(location_df, on='image_id', how='left')

        # Drop extras
        merged = merged.drop(columns=['id', 'category_id'])
        merged = merged.dropna()

        self.gdf = gpd.GeoDataFrame(
            merged,
            geometry=gpd.points_from_xy(merged.lon, merged.lat),
            crs='EPSG:4326'
        )

        return self.gdf

    def _load_climate(self, tif_path) -> pd.DataFrame:  
        # Load the climate raster and affine transform

        with rasterio.open(tif_path) as src:
            climate_classes = src.read(1)
            transform = src.transform

        # Step 1: Convert all lat/lon to row/col
        rows, cols = rowcol(transform, self.gdf['lon'].values, self.gdf['lat'].values)

        # Step 2: Lookup class IDs from raster
        class_ids = climate_classes[rows, cols]

        self.gdf['climate'] = CLIMATE[class_ids]
        
        return self.gdf

    def _load_boundary_ids(
        self, 
        gdf: gpd.GeoDataFrame,
        admin_paths: list,
        gid_cols: list = ['gid_0', 'gid_1', 'gid_2'],
        admin_keys: list = ['GID_0', 'GID_1', 'GID_2']
    ) -> gpd.GeoDataFrame:
        
        """Perform spatial join with admin boundaries and fill missing IDs."""
        print("LOADING ADMINISTRATIVE BOUNDARIES...")

        for i, (path, key_col, new_col) in enumerate(zip(admin_paths, admin_keys, gid_cols)):
            print(f"Processing ADMIN {i}...")

            admin = gpd.read_file(path)
            gdf = gpd.sjoin(gdf, admin[['geometry', key_col]], how='left', predicate='within')
            gdf[new_col] = gdf[key_col]
            gdf = gdf.drop(columns=['index_right', key_col])

            gdf = self._apply_nearest_match(gdf, new_col)
            gdf[new_col] = gdf[new_col].astype(str)

            if i == 2:
                name_cols = admin[[key_col, 'COUNTRY', 'NAME_1', 'NAME_2']].copy()
                name_cols[key_col] = name_cols[key_col].astype(str)
                gdf[new_col] = gdf[new_col].astype(str)

                gdf = gdf.merge(name_cols, left_on=new_col, right_on=key_col, how='left')
                gdf = gdf.rename(columns={
                    'COUNTRY': 'country',
                    'NAME_1': 'state',
                    'NAME_2': 'district'
                }).drop(columns=[key_col])

        for col in gid_cols:
            gdf[col] = pd.factorize(gdf[col])[0]

        print("LOADED ALL BOUNDARY IDS.")
        return gdf

    def _apply_nearest_match(self, gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
        """Fill NaN values in a column based on the closest geographic match."""
        missing = gdf[gdf[col].isnull()]
        if missing.empty:
            return gdf

        not_missing = gdf[gdf[col].notnull()].reset_index(drop=True)
        nearest = not_missing.sindex.nearest(missing.geometry, return_all=False)[1]
        gdf.loc[missing.index, col] = not_missing[col].values.take(nearest)
        
        return gdf
    
if __name__ == '__main__':
    location_file = 'data/inaturalist_2017/raw/locations/train2017_locations.json'
    image_file = 'data/inaturalist_2017/raw/image_metadata/train2017.json'

    dataset = GeoDataset(location_file, image_file)
    processed_df = dataset.create()
    processed_df.to_csv('data/inaturalist_2017/processed/metadata.csv', index=False)
