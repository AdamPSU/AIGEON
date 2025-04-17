from time import time 
import numpy as np
import shapely.wkt
import shapely.ops
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.errors import TopologicalError
from typing import Dict, List, Any, Tuple
from hdbscan import HDBSCAN
from scipy.spatial import Voronoi
from .voronoi import voronoi_finite_polygons
import logging
import time as time_module

from config import CRS 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("geocell")

CELL_COLUMNS = ['id', 'lon', 'lat']
GEOCELL_COLUMNS = ['admin_2', 'admin_1', 'admin_0', 'size', 'num_polygons', 'geometry']

def timer(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time_module.time()
        result = func(*args, **kwargs)
        end_time = time_module.time()
        logger.info(f"{func.__name__} completed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class Cell:
    """Abstraction of a geocell.
    """
    def __init__(self, admin_2: str, admin_1: str, admin_0: str,
                 points: List[Point], polygons: List[Polygon]):
        """Initializes a geocell.

        Args:
            admin_2 (str): name
            admin_1 (str): name of Admin 1 area
            admin_0 (str): name of country
            points (List[Point]): collection of coordinates
            polygons (List[Polygon]): collection of polygons
        """
        start_time = time_module.time()
        self.admin_2 = str(admin_2)
        self.admin_1 = str(admin_1)
        self.admin_0 = str(admin_0)
        self._points = points

        if isinstance(polygons, Polygon):
            self._polygons = [polygons]
        else:
            self._polygons = list(polygons)
        
        logger.debug(f"Cell initialized: {self.admin_2} with {len(points)} points and {len(self._polygons)} polygons")
        logger.debug(f"Cell initialization took {time_module.time() - start_time:.4f} seconds")
    
    @property
    def size(self) -> int:
        """Returns the number of coordinates in cell.

        Returns:
            int: coordinates in cell
        """
        return len(self.points)
        
    @property
    @timer
    def shape(self) -> Polygon:
        """Combines cell's collection of polygons to a geocell shape.

        Returns:
            Polygon: geocell shape
        """
        logger.debug(f"Computing shape for cell {self.admin_2} with {len(self.polygons)} polygons")
        union = shapely.ops.unary_union(self.polygons)
        union = union.buffer(0)
        return union

    @property
    @timer
    def points(self) -> List[Point]:
        """Safely loads points.

        Returns:
            List[Point]: all points.
        """
        try:
            p = [shapely.wkt.loads(x) for x in self._points]
        except TypeError:
            p = self._points

        logger.debug(f"Loaded {len(p)} points for cell {self.admin_2}")
        return p

    @property
    @timer
    def polygons(self) -> List[Polygon]:
        """Safely loads polygons.

        Returns:
            List[Polygon]: all polygons.
        """
        try:
            p = [shapely.wkt.loads(x) for x in self._polygons]
        except TypeError:
            p = self._polygons

        logger.debug(f"Loaded {len(p)} polygons for cell {self.admin_2}")
        return p

    @property
    def multi_point(self) -> MultiPoint:
        """Generates a multi-point shape from points

        Returns:
            MultiPoint: multi-point shape
        """
        return MultiPoint(self.points)

    @property
    @timer
    def coords(self) -> np.ndarray:
        """Generate coordinates from points in the cell.

        Returns:
            np.ndarray: coordinates (lon, lat)
        """
        coords = np.array([[x.x, x.y] for x in self.points])
        logger.debug(f"Generated {len(coords)} coordinates for cell {self.admin_2}")
        return coords

    @property
    def centroid(self) -> np.ndarray:
        """Computes the centroid of the geocell.

        Returns:
            np.ndarray: coordinates of centroid (lon,lat)
        """
        # COMPUTATION BASED ON POINTS:
        return np.mean(self.coords, axis=0)

        # NEW COMPUTATION BASED ON SHAPE
        # return self.shape.centroid
    
    @property
    def empty(self) -> bool:
        """Whether the geocell is empty.

        Returns:
            bool: whether the geocell is empty.
        """
        return len(self.points) == 0

    @timer
    def subtract(self, other: Any):
        """Subtracts other cell from current cell.

        Args:
            other (Any): other cell
        """
        logger.info(f"Subtracting cell {other.admin_2} from {self.admin_2}")
        try:
            diff_shape = self.shape.difference(other.shape)
            
        except TopologicalError as e:
            error_msg = f'Error occurred during subtracting in cell: {self.admin_2}'
            logger.error(error_msg)
            print(error_msg)
            raise TopologicalError(e)
        
        self._polygons = [diff_shape.buffer(0)]

        # Convert Point objects to tuples
        A_tuples = {(point.x, point.y) for point in self.points}
        B_tuples = {(point.x, point.y) for point in other.points}

        # Find tuples in A that are not in B
        difference_tuples = A_tuples - B_tuples

        # Convert tuples back to Point objects if needed
        self._points = [x for x in self.points if (x.x, x.y) in difference_tuples]
        logger.debug(f"After subtraction, cell {self.admin_2} has {len(self._points)} points remaining")

    @timer
    def combine(self, others: List):
        """Combines cell with other cells and deletes other cells' shapes and points.

        Args:
            others (List): list of other geocells.
        """
        logger.info(f"Combining cell {self.admin_2} with {len(others)} other cells")
        for other in others:
            if other is self:
                logger.warning('Tried to combine cell with itself')
                print('Tried to combine cell with itself')
                continue 

            self.add_points(other.points)
            self.add_polygons(other.polygons)
            other._points = []
            other._polygons = []
        
        logger.debug(f"After combination, cell {self.admin_2} has {len(self._points)} points and {len(self._polygons)} polygons")
    
    def add_polygons(self, polygons: List[Polygon]):
        """Adds list of polygons to current cell.

        Args:
            polygons (List[Polygon]): polygons
        """
        logger.debug(f"Adding {len(polygons)} polygons to cell {self.admin_2}")
        self._polygons += polygons
        
    def add_points(self, points: List[Point]):
        """Adds list of points to current cell.

        Args:
            points (List[Point]): points
        """
        logger.debug(f"Adding {len(points)} points to cell {self.admin_2}")
        try:
            self._points += points
        except TypeError:
            self._points += points.tolist()
        
    def tolist(self) -> List:
        """Converts cell to a list.

        Returns:
            List: output
        """
        return [self.admin_2, self.admin_1, self.admin_0, len(self.points), len(self.polygons), self.shape]

    @timer
    def to_pandas(self) -> gpd.GeoDataFrame:
        """Converts a cell to a geopandas DataFrame.

        Returns:
            gpd.GeoDataFrame: geopandas DataFrame.
        """
        data = [[self.admin_2, p.x, p.y] for p in self.points]
        df = pd.DataFrame(data=data, columns=CELL_COLUMNS)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS)
        logger.debug(f"Converted cell {self.admin_2} to pandas DataFrame with {len(df)} rows")
        return df

    @timer
    def __separate_points(self, points: List[Point], polygons: List[Polygon],
                          contain_points: bool) -> Any:
        """Separates the given points and polygons from the current cell.

        Args:
            points (List[Point]): points in the new cell.
            polygons (List[Polygon]): polygons in the new cell.
            contain_points (bool): whether to smooth a cell to contain all inner points

        Returns:
            Any: New cell
        """
        logger.debug(f"Separating {len(points)} points and {len(polygons)} polygons from cell {self.admin_2}")

        coords = tuple((p.x, p.y) for p in points)
        new_name = str(hash(coords))[:12] 

        # Create new shape           
        new_shape = shapely.ops.unary_union(polygons)       
        new_shape = new_shape.buffer(0)
        if contain_points and isinstance(new_shape, MultiPolygon) == False:
            new_shape = Polygon(new_shape.exterior)

        # Create new cell
        new_cell = Cell(new_name, self.admin_1, self.admin_0, points, [new_shape])
        logger.info(f"Created new cell {new_name} with {len(points)} points")
        
        return new_cell

    @timer
    def voronoi_polygons(self, coords: np.ndarray=None) -> List[Polygon]:
        """Generates voronoi shapes that fill out the cell shape.

        Args:
            coords (np.ndarray): Coordinates to be tesselated.
                Defaults to None.

        Returns:
            List[Polygon]: List of polygons.

        Note:
            If coords is none, cell's own points will be tesselated.
        """
        logger.info(f"Generating Voronoi polygons for cell {self.admin_2}")
        # Get Voronoi Regions
        if coords is None:
            v_coords = np.unique(self.coords, axis=0)
        else:
            v_coords = np.unique(coords, axis=0)

        logger.debug(f"Using {len(v_coords)} unique coordinates for Voronoi tessellation")
        vor = Voronoi(v_coords)
        regions, vertices = voronoi_finite_polygons(vor)
        
        # Create Polygons
        polys = []
        for region in regions:
            polygon = Polygon(vertices[region])
            polys.append(polygon)
        
        # Intersect with original cell shape
        try:
            polys = [x.intersection(self.shape) for x in polys]
        except TopologicalError as e:
            error_msg = f'Error occurred in cell: {self.admin_2}'
            logger.error(error_msg)
            print(error_msg)
            raise TopologicalError(e)

        # Return area belonging to each Point
        df = pd.DataFrame({'geometry': polys})
        df = gpd.GeoDataFrame(df, geometry='geometry')
        points = [Point(p[0], p[1]) for p in coords] if coords is not None else self.points
        indices = df.sindex.nearest(points, return_all=False)[1]
        result = [polys[i] for i in indices]
        logger.debug(f"Generated {len(result)} Voronoi polygons")
        return result

    @timer
    def _separate_single_cluster(self, df: pd.DataFrame, cluster: int=0) -> Tuple[List[Any]]:
        """Separates a single cluster from a geocell.

        Args:
            df (pd.DataFrame): Dataframe of points.
            cluster (int): Cluster to seperate out. Defaults to 0.

        Returns:
            Tuple[List[Any]]: New cells.
        """
        logger.info(f"Separating single cluster {cluster} from cell {self.admin_2}")

        # Create polygon map
        polygons = self.voronoi_polygons()

        # Separate out points
        cluster_df = df[df['cluster'] == cluster][['lon', 'lat']]
        assert len(cluster_df.index) > 0, 'Dataframe does not contain a cluster'
        cluster_points = [self.points[i] for i in cluster_df.index]
        cluster_polys = [polygons[i] for i in cluster_df.index]

        # Create new cell
        new_cell = self.__separate_points(cluster_points, cluster_polys, contain_points=True)
        logger.info(f"Created new cell from cluster {cluster} with {len(cluster_points)} points")
        return [new_cell], []

    @timer
    def _separate_multi_cluster(self, df: pd.DataFrame, non_null_large_clusters: List[int]) -> List[Any]:
        """Separates multiple cluster from a geocell.

        Args:
            df (pd.DataFrame): Dataframe of points.
            non_null_large_clusters (pd.Series): Large clusters that are not unassigned.

        Returns:
            List[Any]: New cells.
        """
        logger.info(f"Separating multiple clusters from cell {self.admin_2}: {non_null_large_clusters}")
        # Assign unassigned points based on cluster centroids
        assigned_df = df[df['cluster'].isin(non_null_large_clusters)]
        unassigned_df = df[df['cluster'].isin(non_null_large_clusters) == False]
        cc = assigned_df.groupby(['cluster'])[['lon', 'lat']].mean().reset_index()
        cc = gpd.GeoDataFrame(cc, geometry=gpd.points_from_xy(cc.lon, cc.lat), crs=CRS)

        # Assign unassigned points
        nearest_index = cc.sindex.nearest(unassigned_df.geometry, return_all=False)[1]
        df.loc[df['cluster'].isin(non_null_large_clusters) == False, 'cluster'] = cc.iloc[nearest_index]['cluster'].values   

        # Get polygons
        if len(cc.index) == 2:
            logger.debug("Only 2 clusters found, using single cluster separation")
            return self._separate_single_cluster(df, cluster=cc.iloc[0]['cluster'])
        
        else:
            polygons = self.voronoi_polygons(coords=cc[['lon', 'lat']].values)

            # Separate out clusters
            new_cells = []
            for cluster, polygon in zip(cc['cluster'].unique(), polygons):
                cluster_coords = df[df['cluster'] == cluster][['lon', 'lat']]
                cluster_points = [Point(row.lon, row.lat) for _, row in cluster_coords.iterrows()]
                new_cell = self.__separate_points(cluster_points, [polygon], contain_points=True)
                new_cells.append(new_cell)
                logger.debug(f"Created new cell from cluster {cluster} with {len(cluster_points)} points")

            logger.info(f"Created {len(new_cells)} new cells from multiple clusters")
            return new_cells, [self]

    @timer
    def _split_cell(self, cell_collection: Any, min_samples: int, min_cell_size: int,
                    max_cell_size: int) -> List[Any]:
        """
        The method can split a single Cell into two or more new cells,
        depending on how many valid clusters are found by HDBSCAN.
        """
        logger.info(f"Attempting to split cell {self.admin_2} with {self.size} points")

        # No need to split, already below threshold
        if self.size < max_cell_size:
            logger.debug(f"Cell {self.admin_2} size {self.size} is less than max size {max_cell_size}, no need to split")
            return []

        df = pd.DataFrame(data=self.coords, columns=['lon', 'lat'])
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS)

        # Creates clusters within the geocell
        clustering_start = time_module.time()
        clusterer = HDBSCAN(min_cluster_size=min_cell_size, min_samples=min_samples)
        df['cluster'] = clusterer.fit_predict(df[['lon', 'lat']].values)
        logger.debug(f"HDBSCAN clustering completed in {time_module.time() - clustering_start:.4f} seconds")

        unique_clusters = df['cluster'].nunique()
        logger.debug(f"Found {unique_clusters} unique clusters")
        if unique_clusters < 2:
            logger.info(f"Not enough clusters ({unique_clusters}) found to split cell {self.admin_2}")
            return []

        # If the cluster resulted in a size that's less than the minimum, it's an invalid cluster
        cluster_counts = df['cluster'].value_counts()
        small_clusters = cluster_counts[cluster_counts < min_cell_size].index.tolist()
        df.loc[df['cluster'].isin(small_clusters), 'cluster'] = -1
        logger.debug(f"Marked {len(small_clusters)} small clusters as noise")
        
        # Identify all valid large clusters
        cluster_counts = df['cluster'].value_counts()
        large_clusters = cluster_counts[cluster_counts >= min_cell_size].index
        non_null_large_clusters = [
            c for c in large_clusters 
            if c != -1
        ]
        logger.debug(f"Found {len(non_null_large_clusters)} valid large clusters")

        # There must be at least 2 large clusters
        if len(large_clusters) < 2:
            logger.info(f"Not enough large clusters ({len(large_clusters)}) found to split cell {self.admin_2}")
            return []

        # Special case: 1 real large cluster, and 1 cluster that's just noise. 
        # We're okay with splitting further if the "noise" cluster isn't too large.
        if len(large_clusters) == 2 and len(non_null_large_clusters) == 1:
            null_df = df[df['cluster'] == -1]
            if len(null_df) > max_cell_size:
                logger.info(f"Noise cluster is too large ({len(null_df)} > {max_cell_size}) to split cell {self.admin_2}")
                return []

            logger.info(f"Performing donut split on cell {self.admin_2}")
            # Donut split: one dense cluster, and some noise surrounding it
            # Extracts the real cluster and gives it a voronoi polygon
            new_cells, remove_cells = self._separate_single_cluster(df, non_null_large_clusters[0])
        else:
            logger.info(f"Performing multi-cluster split on cell {self.admin_2}")
            # Normal case: 2 real clusters, and each cluster gets its own voronoi polygon
            new_cells, remove_cells = self._separate_multi_cluster(df, non_null_large_clusters)

        """
        Cells have been split into 1 or more new cells. 
        
        Goals: 
            1. Remove the new pieces from the original cell 
            2. Add those new pieces to CellCollection
            3. Remove "dead" cells that should no longer exist
        """

        # Add cell to CellCollection
        for cell in new_cells:
            self.subtract(cell)
            cell_collection.add(cell)
            logger.debug(f"Added new cell {cell.admin_2} to collection")

        # Remove dead cells from CellCollection
        for cell in remove_cells:
            cell_collection.remove(cell)
            logger.debug(f"Removed cell {cell.admin_2} from collection")

        """
        Newly-created cells may be "dirty". This means
        - MultiPolygons 
        - Ovelapping fragments
        - Disconnected regions 

        This next step cleans them.
        """

        clean_cells = new_cells

        # Special case: current cell still exists and was not removed earlier
        if not len(remove_cells):
            clean_cells += [self]
            logger.debug(f"Added original cell {self.admin_2} to clean cells list")

        cleaning_start = time_module.time()
        self.__clean_dirty_splits(clean_cells)
        logger.debug(f"Cleaned dirty splits in {time_module.time() - cleaning_start:.4f} seconds")

        """
        Are there any more cells that need to be split up again?
        If so, add them to cell_to_split. 
        """

        cells_to_split = []

        # First, check original cell. If it's too big, split again
        if self.size > max_cell_size and self not in remove_cells:
            cells_to_split.append(self)
            logger.debug(f"Original cell {self.admin_2} still needs splitting")

        # Next, check new cells. If they're too big, split again.
        for cell in new_cells:
            if cell.size > max_cell_size:
                cells_to_split.append(cell)
                logger.debug(f"New cell {cell.admin_2} needs further splitting")

        logger.info(f"Split operation complete. {len(cells_to_split)} cells need further splitting")
        return cells_to_split

    @timer
    def __clean_dirty_splits(self, cells: List[Any]):
        """Cleans messy splits that split polygons into multiple parts.
        """
        logger.info(f"Cleaning dirty splits for {len(cells)} cells")
        df = pd.DataFrame(data=[x.tolist() for x in cells], columns=GEOCELL_COLUMNS)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=CRS)

        # Identifying Multipolygons
        multi_polys = df[df['geometry'].type == 'MultiPolygon']
        logger.debug(f"Found {len(multi_polys)} MultiPolygons to clean")

        # Iterate through rows with Multipolygons
        for index, row in multi_polys.iterrows():

            # Find points
            points = cells[index].to_pandas()['geometry'] # .to_crs('EPSG:3857')
            
            # Splitting Multipolygons
            all_polygons = list(row['geometry'].geoms)
            logger.debug(f"Cell {cells[index].admin_2} has {len(all_polygons)} sub-polygons")
            
            # Finding the Largest Sub-Polygon
            largest_poly = max(all_polygons, key=lambda polygon: polygon.area)
            
            # Flag
            did_assign = False
            
            # Assigning Smaller Polygons
            for small_poly in all_polygons:
                if small_poly != largest_poly:
                    
                    # Creating a GeoSeries with the same index and CRS as 'test'
                    small_poly_gseries = gpd.GeoSeries([small_poly], index=[index], crs=CRS)
                    
                    # Exclude the original polygon during the intersection calculation
                    other_polys = df.drop(index)
                    
                    # Create a small buffer around the small polygon to account for mismatched borders
                    buffered_poly = small_poly_gseries.buffer(0.01)
                    
                    # Identify polygons that intersect with the buffered small polygon
                    intersecting_polys = other_polys[other_polys.intersects(buffered_poly.unary_union)]

                    if len(intersecting_polys) == 0:
                        logger.debug(f"No intersecting polygons found for a sub-polygon in cell {cells[index].admin_2}")
                        continue

                    did_assign = True
                    
                    # Find the polygon that has the largest intersection area
                    largest_intersect_index = intersecting_polys.geometry.apply(
                        lambda poly: poly.intersection(buffered_poly.unary_union).area
                    ).idxmax()

                    # Checking which points fall into 'small_poly'
                    mask = points.within(small_poly)
                    points_in_small_poly = points[mask]
                    cells[index]._points = [x for x in cells[index].points if x not in points_in_small_poly]
                    cells[largest_intersect_index].add_points(points_in_small_poly)
                    logger.debug(f"Transferred {len(points_in_small_poly)} points from cell {cells[index].admin_2} to cell {cells[largest_intersect_index].admin_2}")
                    
                    # Union the small polygon with the polygon having largest common border
                    cells[largest_intersect_index]._polygons = [cells[largest_intersect_index].shape.union(small_poly)]
                    logger.debug(f"Merged sub-polygon from {cells[index].admin_2} into {cells[largest_intersect_index].admin_2}")

            if did_assign:
                # Keeping the largest polygon in the original GeoDataFrame
                cells[index]._polygons = [largest_poly]
                logger.debug(f"Updated cell {cells[index].admin_2} to keep only largest polygon")

    def __eq__(self, other):
        return self.admin_2 == other.admin_2
    
    def __ne__(self, other):
        return self.admin_2 != other.admin_2

    def __hash__(self):
        return hash(self.admin_2)
        
    def __repr__(self):
        rep = f'Cell(id={self.admin_2}, admin_1={self.admin_1}, admin_0={self.admin_0}, size={len(self.points)}, num_polys={len(self.polygons)})'
        return rep
        
    def __str__(self):
        return self.__repr__()