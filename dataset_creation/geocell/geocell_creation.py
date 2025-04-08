# FILE STARTS HERE
import warnings; warnings.filterwarnings("ignore")
import functools
import numpy as np
import pandas as pd
import geopandas as gpd
import s3fs 

from typing import Tuple, List
from tqdm import tqdm

from .cell import Cell
from .cell_collection import CellCollection
from .cell_balancer import cell_fuser, cell_splitter 

from config import LOC_PATH, MIN_CELL_SIZE, CRS

# Constants
NEEDED_COLS = {'id', 'country_code', 'lat', 'lon'}
ADMIN_NAMES = ['gid_0', 'gid_1', 'gid_2']

class GeocellCreator:
    def __init__(
        self,
        df: pd.DataFrame,
        admin2_path: str,
        output_file: str,
    ) -> None:

        self.output_file = output_file
        self.admin_2_path = admin2_path
        self.cells = None

        self.fused_cell_path = 'data/geocells/cells/inat2017_fused_cells.npy'

        # Convert the dataframe to a GeoDataFrame
        self.gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.lon, df.lat), 
            crs='EPSG:4326'
        )
        self.gdf = self.gdf.to_crs(crs=CRS)

    def generate(self, min_cell_size: int, max_cell_size: int, num_workers: int, load_fused=True):
        print("Beginning geocell creation algorithm...") 
        granular_cells = self.initialize_cells(min_cell_size)

        if load_fused:
            print("Skipped cell fusing.")
            cell_list = np.load(self.fused_cell_path, allow_pickle=True)
            fused_cells = CellCollection(cell_list)
        else: 
            fused_cells = cell_fuser(granular_cells, min_cell_size, num_workers)
        
        divided_cells = cell_splitter(fused_cells, min_cell_size, max_cell_size, num_workers)
        cell_df = divided_cells.to_geopandas()
        cell_df.to_csv(self.output_file)

    def initialize_cells(self, min_cell_size: int) -> CellCollection:
        granular_boundaries = self.load_granular_boundaries()

        initialize_cell_fnc = functools.partial(self._load_granular_cells, granular_boundaries, min_cell_size)
        tqdm.pandas(desc='Loading admin 2 cells')
        cells = self.gdf.groupby(ADMIN_NAMES[2]).progress_apply(initialize_cell_fnc)
        cells = [item for sublist in cells for item in sublist]

        self._assign_unassigned_areas(cells, granular_boundaries)
        return CellCollection(cells)

    def load_granular_boundaries(self):
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

def main(): 
    print("Beginning geocell creation algorithm...")
    df = pd.read_csv(LOC_PATH)
    geocell_creator = GeocellCreator(df, 'data/geocells/cells/inat2017_cells.csv')
    
    cells = geocell_creator.initialize_cells(MIN_CELL_SIZE)
    cells = cell_fuser(cells, num_workers=6)

if __name__ == '__main__':
    main()