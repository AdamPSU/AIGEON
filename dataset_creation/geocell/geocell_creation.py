# FILE STARTS HERE
import warnings
import functools
import numpy as np
import pandas as pd
import geopandas as gpd

from typing import Tuple, List
from tqdm import tqdm

from .cell import Cell
from .cell_collection import CellCollection
from config import ADMIN_2_PATH, LOC_PATH, MIN_CELL_SIZE, MAX_CELL_SIZE

# Constants
CRS = 'EPSG:4326'
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
        self.gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS)
        
    def generate(self, min_cell_size: int=MIN_CELL_SIZE, max_cell_size: int=MAX_CELL_SIZE):
        """Generate geocells.

        Args:
            min_cell_size (int, optional): Minimum number of training examples per geocell.
                Defaults to MIN_CELL_SIZE.
            max_cell_size (int, optional): Maximum number of training examples per geocell.
                Defaults to MAX_CELL_SIZE.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.cells = self._initialize_cells(min_cell_size)
            self.cells.balance(min_cell_size, max_cell_size)
            self.geocell_df = self.cells.to_pandas()
            self.geocell_df.to_csv(self.output, index=False)

        return self.cells

    def _initialize_cells(self, min_cell_size: int) -> CellCollection:
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
        print("INITIALIZING ADMIN 2 CELLS")
        cells = [
            self.__initialize_cell(group, admin_2_lookup)
            for _, group in self.gdf.groupby(ADMIN_NAMES[2])
        ]

        self._assign_unassigned_areas(cells, admin_2)
        return CellCollection(cells)


    def __initialize_cell(self, gdf: gpd.GeoDataFrame, admin_2_lookup: dict) -> Cell:
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
        name, admin_1, country = row[ADMIN_NAMES[2]], row[ADMIN_NAMES[1]], row[ADMIN_NAMES[0]]

        # Get point geometries
        points = gdf.geometry.tolist()

        # Get polygon geometry from lookup
        polygon = admin_2_lookup.get(name)
        polygons = [polygon] if polygon is not None else []

        return Cell(name, admin_1, country, points, polygons)

    def _load_granular_boundaries(self):
        """Loads geographic boundaries at the admin 2 level."""

        print('LOADING GRANULAR BOUNDARIES')

        # Load smaller administrative areas
        admin_2 = gpd.read_file(ADMIN_2_PATH).set_crs(crs=CRS)
        admin_2['geometry'] = admin_2['geometry'].buffer(0)
        
        print('LOADED GRANULAR BOUNDARIES')

        return admin_2

    def _assign_unassigned_areas(self, cells: List[Cell], admin_2: gpd.GeoDataFrame):
        """Adds unassigned admin 2 areas to the existing cells.

        Args:
            cells (List[Cell]): Existing geocells.
            admin_2 (gpd.GeoDataFrame): Admin 2 boundary GeoDataFrame.
        """

        # Build quick access to cell objects
        cell_map = {int(cell.cell_id): cell for cell in cells}
        assigned_idx = set(cell_map.keys())

        # Identify assigned and unassigned polygons
        is_assigned = admin_2.index.isin(assigned_idx)
        if is_assigned.all():
            return  # Nothing to do — all polygons are assigned

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


if __name__ == '__main__':
    df = pd.read_csv(LOC_PATH)
    geocell_creator = GeocellCreator(df, 'data/geocells/cells/inat2017_cells.csv')
    geocells = geocell_creator.generate()