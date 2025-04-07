import numpy as np
import pandas as pd
import geopandas as gpd
import logging 
import networkx as nx
from rtree import index

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

def build_neighbor_graph(cells: CellCollection, tolerance=0):
    """
    Build a neighbor graph where each cell is a node and an edge exists 
    between cells if their shapes intersect (or are within a distance tolerance).
    
    Args:
        cells (CellCollection): A collection of Cell objects.
        tolerance (float): A tolerance distance in the same units as CRS.
        
    Returns:
        networkx.Graph: Graph with cell.admin_2 as nodes and edges for neighbors.
    """
    G = nx.Graph()
    cell_list = [(cell.admin_2, cell, cell.shape) for cell in cells]
    
    # Create spatial index for fast lookup
    spatial_idx = index.Index()
    for pos, (cell_id, cell, shape) in enumerate(cell_list):
        spatial_idx.insert(pos, shape.bounds, obj=cell_id)
        G.add_node(cell_id, cell=cell)
        
    # For each cell, use the spatial index to find neighbors
    for pos, (cell_id, cell, shape) in enumerate(cell_list):
        # Query the spatial index using the cell's bounding box
        for hit in spatial_idx.intersection(shape.bounds, objects=True):
            neighbor_id = hit.object
            if neighbor_id == cell_id:
                continue
            # Retrieve the neighbor cell and its shape
            neighbor = next(c for (cid, c, s) in cell_list if cid == neighbor_id)
            # Add an edge if the shapes intersect or are within tolerance distance
            if shape.intersects(neighbor.shape) or shape.distance(neighbor.shape) <= tolerance:
                G.add_edge(cell_id, neighbor_id)
                
    return G

def merge_cells_in_component(cell_ids, graph):
    """
    Given a set of cell IDs (nodes in the graph), merge all corresponding cells.
    
    Args:
        cell_ids (set): Set of cell identifiers (admin_2 values).
        graph (networkx.Graph): The neighbor graph.
        
    Returns:
        Cell: A merged Cell object.
    """
    # Retrieve the cell objects from the graph nodes
    cells_to_merge = [graph.nodes[cell_id]['cell'] for cell_id in cell_ids]
    # Choose one cell as the base; merge the rest into it
    base_cell = cells_to_merge[0]
    if len(cells_to_merge) > 1:
        base_cell.combine(cells_to_merge[1:])
    return base_cell

def neighbor_graph_fuse(cells: CellCollection, tolerance=0, min_size=MIN_CELL_SIZE):
    """
    Fuses geocells by building a neighbor graph and merging connected cells.
    
    Args:
        cells (CellCollection): The original collection of cells.
        tolerance (float): Distance tolerance to consider cells as neighbors.
        min_size (int): The minimum cell size required.
        
    Returns:
        CellCollection: New collection with merged cells.
    """
    # Build the neighbor graph
    G = build_neighbor_graph(cells, tolerance=tolerance)
    
    # Identify connected components in the graph. Each component is a set of cell IDs.
    components = list(nx.connected_components(G))
    merged_cells = []
    
    for comp in components:
        # Merge all cells in the connected component
        merged_cell = merge_cells_in_component(comp, G)
        # Optionally, if a merged cell does not reach min_size, you could further process it.
        # For now, we add the merged cell as-is.
        merged_cells.append(merged_cell)
        
    return CellCollection(merged_cells)

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
    
    # Parallelize neighbor fusing algorithm
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(neighbor_graph_fuse, group) for group in grouped_country_cells]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fusing countries"):
            result = future.result() 
            all_cells.extend(result)

    return CellCollection(all_cells)  

def main(): 
    print("Beginning geocell creation algorithm...")
    df = pd.read_csv(LOC_PATH)
    geocell_creator = GeocellCreator(df, 'data/geocells/cells/inat2017_cells.csv')
    
    cells = geocell_creator.initialize_cells(MIN_CELL_SIZE)
    cells = parallelize_fusing(cells, num_workers=8)

if __name__ == '__main__':
    main()
