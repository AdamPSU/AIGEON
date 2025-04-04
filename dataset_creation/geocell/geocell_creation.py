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
        """Creates geocells based on a supplied dataframe.

        Args:
            df (pd.DataFrame): Pandas dataframe used during training.
            output_file (str): Where the geocells should be saved to.
        """

        self.output = output_file
        self.cells = None
        self.gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        

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
            self.geo_cell_df = self.cells.to_pandas()
            self.geo_cell_df.to_csv(self.output, index=False)

        return self.cells

    def _initialize_cells(self, min_cell_size: int) -> CellCollection:
        """Assigns IDs to each location in the dataset based on geographic boundaries.

        Args:
            min_cell_size (int): Suggested minimum cell size.

        Returns:
            CellCollection: Initial geocells on admin 2 hierarchy level.

        Note: This helper function is created such that the boundaries, which are huge
            files, can go out of scope as quickly as possible to free up memory.
        """

        admin_2 = self._load_granular_boundaries()

        # Initialize all geocells
        initialize_cell_fnc = functools.partial(self.__initialize_cell, admin_2)
        tqdm.pandas(desc='INITIALIZING ADMIN 2 GEOCELLS')
        cells = self.gdf.groupby(ADMIN_NAMES[2]).progress_apply(initialize_cell_fnc)
        cells = [item for sublist in cells for item in sublist]

        # Add unassigned areas to cells
        self._assign_unassigned_areas(cells, admin_2)
        return CellCollection(cells)
    
    def __initialize_cell(self, admin_2_boundary: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame) -> Cell:
        """Initializes a geocell based on an admin 2 boundary level.

        Args:
            admin_2_boundary (gpd.GeoDataFrame): file containing admin 2 polygons.
            min_cell_size (int): suggested minimum cell size.
            df (gpd.GeoDataFrame): Dataframe containing all coordinates of a given
                admin 2 level.

        Returns:
            Cell: Geocell.
        """
        
        # Get metadata
        name = gdf.iloc[0][ADMIN_NAMES[2]]
        admin_1 = gdf.iloc[0][ADMIN_NAMES[1]]
        country = gdf.iloc[0][ADMIN_NAMES[0]]

        # Get shapes
        polygon_ids = np.array([int(x) for x in gdf[ADMIN_NAMES[2]].unique()])
        points = gdf['geometry'].values.tolist()
        polygons = admin_2_boundary.iloc[polygon_ids].geometry.unique().tolist()

        return [Cell(name, admin_1, country, points, polygons)]

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
        # Determined assigned and unassigned polygons
        cell_map = {int(cell.cell_id): cell for cell in cells}
        cell_idx = list(cell_map.keys())

        assigned = admin_2.loc[admin_2.index.isin(cell_idx)].copy().reset_index()
        assigned['centroid'] = [row.geometry.centroid for _, row in assigned.iterrows()]
        assigned = gpd.GeoDataFrame(assigned, geometry='centroid')

        unassigned = admin_2.loc[admin_2.index.isin(cell_idx) == False].reset_index(drop=True)
        unassigned['centroid'] = [row.geometry.centroid for _, row in unassigned.iterrows()]
        unassigned = gpd.GeoDataFrame(unassigned, geometry='centroid')

        # Find assignments
        closest_match = assigned.sindex.nearest(unassigned.centroid)[1]
        assignments = assigned.iloc[closest_match]['index'].values

        # Add polygons to closest cells
        for i, row in unassigned.iterrows():
            closest_cell = assignments[i]
            cell_map[closest_cell].add_polygons([row['geometry']])


if __name__ == '__main__':
    df = pd.read_csv(LOC_PATH)
    geocell_creator = GeocellCreator(df, 'data/geocells/cells/inat2017_cells.csv')
    geocells = geocell_creator.generate()