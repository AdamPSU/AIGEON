# FILE STARTS HERE
import numpy as np
import pandas as pd
import geopandas as gpd
import logging 

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Tuple, List
from tqdm import tqdm

from .cell import Cell
from .cell_collection import CellCollection
from config import ADMIN_2_PATH, LOC_PATH, MIN_CELL_SIZE, MAX_CELL_SIZE, CRS

logger = logging.getLogger(__name__)

# Constants
NEEDED_COLS = {'id', 'country_code', 'lat', 'lon'}
ADMIN_NAMES = ['gid_0', 'gid_1', 'gid_2']

class GeocellCreator:
    def __init__(self, df: pd.DataFrame, output_file: str) -> None:
        """
        Creates geocells based on a supplied dataframe.

        Args:
            df (pd.DataFrame): Pandas dataframe used during training.
            output_file (str): Where the geocells should be saved to.
        """

        self.output = output_file
        self.cells = None
        
        # Must first read in ESPG:4326 to recognize lon/lat pairs, then convert to ESPG:4087
        # To get the same representation in meters. 
        self.gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')
        self.gdf.to_crs(crs=CRS)

    def initialize_cells(self, min_cell_size: int) -> CellCollection:
        """
        Assigns IDs to each location in the dataset based on geographic boundaries.

        Args:
            min_cell_size (int): Suggested minimum cell size.

        Returns:
            CellCollection: Initial geocells on admin 2 hierarchy level.
        """

        admin_2 = self._load_granular_boundaries()

        # Build a quick lookup dictionary from admin_2_boundary
        # NOTE: Not sure if the {gid_value: geometry} pairs actually match up. 
        # Make sure to double-check this soon.
        admin_2_lookup = admin_2.geometry.to_dict()

        # Initialize geocells by group
        cells = [
            self._load_granular_cells(group, admin_2_lookup)
            for _, group in self.gdf.groupby(ADMIN_NAMES[2])
        ]
        self._assign_unassigned_areas(cells, admin_2)
        print("Loaded admin 2 cells.")

        return CellCollection(cells)

    def _load_granular_cells(self, gdf: gpd.GeoDataFrame, admin_2_lookup: dict) -> Cell:
        """
        Initializes a geocell based on an admin 2 boundary level.

        Args:
            gdf (gpd.GeoDataFrame): Coordinates for a given admin 2 unit.
            admin_2_lookup (dict): Maps admin_2 names to geometries.

        Returns:
            Cell: A single geocell.
        """

        # Extract metadata from the first row
        row = gdf.iloc[0]
        admin_2, admin_1, admin_0 = row[ADMIN_NAMES[2]], row[ADMIN_NAMES[1]], row[ADMIN_NAMES[0]]

        # Get point geometries
        points = gdf.geometry.tolist()

        # Get polygon geometry from lookup
        polygon = admin_2_lookup.get(admin_0)
        polygons = [polygon] if polygon is not None else []

        return Cell(admin_2, admin_1, admin_0, points, polygons)

    def _load_granular_boundaries(self):
        """Loads geographic boundaries at the admin 2 level."""

        # Load smaller administrative areas
        admin_2 = gpd.read_file(ADMIN_2_PATH).to_crs(crs=CRS)
        admin_2['geometry'] = admin_2['geometry'].buffer(0)
        
        print('Loaded admin 2 geometries.')

        return admin_2

    def _assign_unassigned_areas(self, cells: List[Cell], admin_2: gpd.GeoDataFrame):
        """Adds unassigned admin 2 areas to the existing cells.

        Args:
            cells (List[Cell]): Existing geocells.
            admin_2 (gpd.GeoDataFrame): Admin 2 boundary GeoDataFrame.
        """

        # Build quick access to cell objects
        cell_map = {int(cell.admin_2): cell for cell in cells}
        assigned_idx = set(cell_map.keys())

        # Identify assigned and unassigned polygons
        is_assigned = admin_2.index.isin(assigned_idx)

        assigned = admin_2.loc[is_assigned].copy()
        unassigned = admin_2.loc[~is_assigned].copy()
        
        # Compute centroids
        assigned['centroid'] = assigned.geometry.centroid
        unassigned['centroid'] = unassigned.geometry.centroid

        # Use spatial index to find nearest assigned centroid for each unassigned one
        assigned_centroids = assigned.set_geometry('centroid')
        unassigned_centroids = unassigned.set_geometry('centroid')

        # Efficient nearest neighbor lookup using GeoPandas spatial index
        _, nearest_idx = assigned_centroids.sindex.nearest(unassigned_centroids.geometry, return_distance=False)

        # Get back original admin_2 indices from assigned
        assigned_original_idx = assigned.index.to_numpy()
        target_cell_ids = assigned_original_idx[nearest_idx]  # Map to actual cell IDs

        # Add unassigned polygons to their nearest assigned cells
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
        for country in granular_cells.countries
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(fuse_cells, group) for group in grouped_country_cells]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fusing countries"):
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