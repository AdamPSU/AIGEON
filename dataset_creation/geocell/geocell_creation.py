# FILE STARTS HERE
import warnings; warnings.filterwarnings("ignore")
import functools
import numpy as np
import pandas as pd
import geopandas as gpd
import s3fs 

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Tuple, List
from tqdm import tqdm

from .cell import Cell
from .cell_collection import CellCollection
from config import LOC_PATH, ADMIN_2_PATH, MIN_CELL_SIZE, MAX_CELL_SIZE, CRS

# Constants
NEEDED_COLS = {'id', 'country_code', 'lat', 'lon'}
ADMIN_NAMES = ['gid_0', 'gid_1', 'gid_2']

class GeocellCreator:
    def __init__(
        self,
        df: pd.DataFrame,
        admin_2_path: str,
        output_file: str,
    ) -> None:
        """
        Creates geocells based on a supplied dataframe.

        Args:
            df (pd.DataFrame): Pandas dataframe used during training.
            output_file (str): Where the geocells should be saved to.
            admin_2_path (str): Path to the admin 2 boundaries file.
            admin_names (List[str]): Column names for [admin_0, admin_1, admin_2].
        """
        self.output = output_file
        self.admin_2_path = admin_2_path
        self.cells = None

        # Convert the dataframe to a GeoDataFrame
        self.gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')
        self.gdf = self.gdf.to_crs(crs=CRS)

    def initialize_cells(self, min_cell_size: int) -> CellCollection:
        granular_boundaries = self._load_granular_boundaries()

        initialize_cell_fnc = functools.partial(self._load_granular_cells, granular_boundaries, min_cell_size)
        tqdm.pandas(desc='Loading admin 2 cells')
        cells = self.gdf.groupby(ADMIN_NAMES[2]).progress_apply(initialize_cell_fnc)
        cells = [item for sublist in cells for item in sublist]

        self._assign_unassigned_areas(cells, granular_boundaries)
        return CellCollection(cells)

    def _load_granular_boundaries(self):
        print('Loading admin 2 geometries...')
        if self.admin_2_path.startswith("s3://"):
            fs = s3fs.S3FileSystem()
            with fs.open(self.admin_2_path, 'rb') as f:
                admin_2 = gpd.read_file(f).to_crs(crs=CRS)
        else:
            admin_2 = gpd.read_file(self.admin_2_path).to_crs(crs=CRS)

        admin_2['geometry'] = admin_2['geometry'].buffer(0)
        return admin_2

    def _load_granular_cells(self, adm_2_gdf, min_cell_size: int, gdf: gpd.GeoDataFrame) -> List[Cell]:
        row = gdf.iloc[0]
        admin_2, admin_1, admin_0 = row[ADMIN_NAMES[2]], row[ADMIN_NAMES[1]], row[ADMIN_NAMES[0]]

        polygon_ids = np.array([int(x) for x in gdf[ADMIN_NAMES[2]].unique()])
        points = gdf.geometry.tolist()
        polygons = adm_2_gdf.iloc[polygon_ids].geometry.unique().tolist()

        return [Cell(admin_2, admin_1, admin_0, points, polygons)]

    def _assign_unassigned_areas(self, cells: List[Cell], admin_2: gpd.GeoDataFrame):
        cell_map = {int(cell.admin_2): cell for cell in cells}
        assigned_idx = set(cell_map.keys())

        is_assigned = admin_2.index.isin(assigned_idx)

        assigned = admin_2.loc[is_assigned].copy()
        unassigned = admin_2.loc[~is_assigned].copy()

        assigned['centroid'] = assigned.geometry.centroid
        unassigned['centroid'] = unassigned.geometry.centroid

        assigned_centroids = assigned.set_geometry('centroid')
        unassigned_centroids = unassigned.set_geometry('centroid')

        _, nearest_idx = assigned_centroids.sindex.nearest(unassigned_centroids.geometry, return_distance=False)

        assigned_original_idx = assigned.index.to_numpy()
        target_cell_ids = assigned_original_idx[nearest_idx]

        for poly, target_id in zip(unassigned.geometry, target_cell_ids):
            if target_id in cell_map:
                cell_map[target_id].add_polygons([poly])

def parallelize_fusing(granular_cells: CellCollection, num_workers):
    all_cells = []

    # Prepare the per-country cell groups
    grouped_country_cells = [
        CellCollection([
            cell for cell in granular_cells
            if cell.admin_0 == country
        ])
        for country in tqdm(granular_cells.countries, desc="Dividing into countries")
    ]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(fuse_cells, group) for group in grouped_country_cells]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fusing admin 2 cells within countries."):
            result = future.result()  # Raises if an exception occurred in worker
            all_cells.extend(result)

    return CellCollection(all_cells)  

def get_candidates(center_row, potential_df, admin_filter=True, small_filter=False):
    candidates = potential_df
    if admin_filter:
        candidates = candidates[candidates['admin_1'] == center_row['admin_1']]
    if small_filter:
        candidates = candidates[candidates['size'] < MIN_CELL_SIZE]
    return candidates

def fuse_cells(cells: CellCollection) -> CellCollection:
    excluded_ids = set()
    cell_df = cells.to_geopandas().set_index('admin_2')

    while True:
        consider_df = cell_df[~cell_df.index.isin(excluded_ids)]
        df_small = consider_df[consider_df['size'] < MIN_CELL_SIZE]

        if df_small.empty:
            break

        center_row = df_small.sample().iloc[0]
        center_id = center_row.name
        center_cell = cells.find(center_id)
        potential_neighbors = consider_df.drop(center_id)

        buffer_radii = [500, 1000, 4000, 8000]
        found_indices = np.array([])

        for radius in buffer_radii:
            center_buffer = center_row.geometry.buffer(radius)

            for admin_filter, small_filter in [
                (True, True),
                (True, False),
                (False, True),
                (False, False)
            ]:
                
                if found_indices.size > 0:
                    break

                candidates = get_candidates(center_row, potential_neighbors, admin_filter, small_filter)
                hits = candidates.sindex.query(center_buffer, predicate='intersects')
                found_indices = candidates.iloc[hits].index.values

            if found_indices.size > 0:
                break

        if found_indices.size == 0:
            excluded_ids.add(center_id)
            continue

        neighbors = potential_neighbors.loc[found_indices]
        neighbors = neighbors.sort_values(by='size', ascending=False)

        merged_cells = []
        total_size = center_cell.size

        for neighbor_id, row in neighbors.iterrows():
            if total_size >= MIN_CELL_SIZE:
                break
            neighbor_cell = cells.find(neighbor_id)
            merged_cells.append(neighbor_cell)
            total_size += neighbor_cell.size

        if merged_cells:
            center_cell.combine(merged_cells)

            # Update center info
            cell_df.loc[center_id, 'size'] = center_cell.size
            cell_df.loc[center_id, 'geometry'] = center_cell.shape
            cell_df.loc[center_id, 'num_polygons'] = len(center_cell.polygons)

            for neighbor in merged_cells:
                neighbor_id = neighbor.admin_2
                cell_df = cell_df.drop(neighbor_id, errors='ignore')
        else:
            excluded_ids.add(center_id)

    return list(cells)

def main(): 
    print("Beginning geocell creation algorithm...")
    df = pd.read_csv(LOC_PATH)
    geocell_creator = GeocellCreator(df, 'data/geocells/cells/inat2017_cells.csv')
    
    cells = geocell_creator.initialize_cells(MIN_CELL_SIZE)
    cells = parallelize_fusing(cells, num_workers=6)

if __name__ == '__main__':
    main()