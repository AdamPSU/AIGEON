import numpy as np
import shapely.wkt
import shapely.ops
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.errors import TopologicalError
from typing import List, Any, Tuple
from hdbscan import HDBSCAN
from scipy.spatial import Voronoi
from .voronoi import voronoi_finite_polygons

from config import CRS 

CELL_COLUMNS = ['id', 'lon', 'lat']
GEOCELL_COLUMNS = ['admin_2', 'admin_1', 'admin_0', 'size', 'num_polygons', 'geometry']

class Cell:
    """Abstraction of a geocell."""
    def __init__(self, admin_2: str, admin_1: str, admin_0: str,
                 points: List[Point], polygons: List[Polygon]):
        """Initializes a geocell.

        Args:
            admin_2 (str): admin 2 identifier
            admin_1 (str): admin 1 identifier
            admin_0 (str): admin 0 identifier 
            points (List[Point]): collection of coordinates
            polygons (List[Polygon]): collection of polygons
        """

        self.admin_2 = str(admin_2)
        self.admin_1 = str(admin_1)
        self.admin_0 = str(admin_0)
        self._points = points

        if isinstance(polygons, Polygon):
            self._polygons = [polygons]
        else:
            self._polygons = list(polygons)
        
    @property
    def size(self) -> int:
        """Returns the number of coordinates in cell.

        Returns:
            int: coordinates in cell
        """
        return len(self.points)
        
    @property
    def shape(self) -> Polygon:
        """Combines cell's collection of polygons to a geocell shape.

        Returns:
            Polygon: geocell shape
        """
        union = shapely.ops.unary_union(self.polygons)
        if not union.is_valid:
            print(f"[WARNING] Invalid geometry in unary_union for cell: {self.admin_2}")
        
        union = union.buffer(0)  # Buffer(0) fixes many invalid geometries
        return union


    @property
    def points(self) -> List[Point]:
        """Safely loads points.

        Returns:
            List[Point]: all points.
        """
        try:
            p = [shapely.wkt.loads(x) for x in self._points]
        except TypeError:
            p = self._points

        return p

    @property
    def polygons(self) -> List[Polygon]:
        """Safely loads polygons.

        Returns:
            List[Polygon]: all polygons.
        """
        try:
            p = [shapely.wkt.loads(x) for x in self._polygons]
        except TypeError:
            p = self._polygons

        return p

    @property
    def multi_point(self) -> MultiPoint:
        """Generates a multi-point shape from points

        Returns:
            MultiPoint: multi-point shape
        """
        return MultiPoint(self.points)

    @property
    def coords(self) -> np.ndarray:
        """Generate coordinates from points in the cell.

        Returns:
            np.ndarray: coordinates (lon, lat)
        """
        return np.array([[x.x, x.y] for x in self.points])

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

    def subtract(self, other: Any):
        """Subtracts other cell from current cell."""
        try:
            diff_shape = self.shape.difference(other.shape)
            if not diff_shape.is_valid:
                print(f"[WARNING] Invalid geometry during difference in cell: {self.admin_2}")
        except TopologicalError as e:
            print(f"[ERROR] TopologicalError in subtract: {self.admin_2}")
            raise TopologicalError(e)

        self._polygons = [diff_shape.buffer(0)]

        # Update points
        A_tuples = {(point.x, point.y) for point in self.points}
        B_tuples = {(point.x, point.y) for point in other.points}
        difference_tuples = A_tuples - B_tuples
        self._points = [x for x in self.points if (x.x, x.y) in difference_tuples]

    def combine(self, others: List):
        """Combines cell with other cells and deletes other cells' shapes and points.

        Args:
            others (List): list of other geocells.
        """
        for other in others:
            if other is self:
                continue 

            self.add_points(other.points)
            self.add_polygons(other.polygons)
            other._points = []
            other._polygons = []
        
    def add_polygons(self, polygons: List[Polygon]):
        """Adds list of polygons to current cell.

        Args:
            polygons (List[Polygon]): polygons
        """
        self._polygons += polygons
        
    def add_points(self, points: List[Point]):
        """Adds list of points to current cell.

        Args:
            points (List[Point]): points
        """
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

    def to_geopandas(self) -> gpd.GeoDataFrame:
        """Converts a cell to a geopandas DataFrame.

        Returns:
            gpd.GeoDataFrame: geopandas DataFrame.
        """
        data = [[self.admin_2, p.x, p.y] for p in self.points]
        df = pd.DataFrame(data=data, columns=CELL_COLUMNS)
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS)
        return df

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
        coords = tuple((p.x, p.y) for p in points)
        new_name = str(hash(coords))[:12] 

        # Create new shape           
        new_shape = shapely.ops.unary_union(polygons)       
        new_shape = new_shape.buffer(0)
        if contain_points and isinstance(new_shape, MultiPolygon) == False:
            new_shape = Polygon(new_shape.exterior)

        # Create new cell
        new_cell = Cell(new_name, self.admin_1, self.admin_0, points, [new_shape])
        return new_cell

    def voronoi_polygons(self, coords: np.ndarray = None) -> List[Polygon]:
        """Generates Voronoi shapes that fill out the cell shape."""
        
        if coords is None:
            coords = self.coords

        # Remove duplicate coordinates and keep their original indices
        unique_coords, unique_indices = np.unique(coords, axis=0, return_index=True)

        if len(unique_coords) < 3:
            print(f"[ERROR] Voronoi failed: insufficient unique points in cell {self.admin_2}")
            return []

        try:
            vor = Voronoi(unique_coords)
            regions, vertices = voronoi_finite_polygons(vor)
        except Exception as e:
            print(f"[ERROR] Voronoi generation failed for {self.admin_2}: {e}")
            return []

        # Create polygons corresponding to unique points
        polys = []
        for region in regions:
            try:
                polygon = Polygon(vertices[region])
                if not polygon.is_valid:
                    print(f"[WARNING] Invalid Voronoi polygon in {self.admin_2}")
                polys.append(polygon)
            except Exception as e:
                print(f"[ERROR] Failed to create polygon in cell {self.admin_2}: {e}")

        try:
            # Clip polygons to the cell shape
            polys = [p.intersection(self.shape) for p in polys]
        except TopologicalError as e:
            print(f"[ERROR] TopologicalError during intersection in {self.admin_2}")
            raise TopologicalError(e)

        # Re-expand to match the original point count
        full_polygons = [None] * len(coords)
        for i, idx in enumerate(unique_indices):
            full_polygons[idx] = polys[i]

        # Replace missing polygons with fallbacks if needed
        fallback = Polygon()
        full_polygons = [p if p is not None else fallback for p in full_polygons]

        return full_polygons

    def _separate_single_cluster(self, df: pd.DataFrame, cluster: int=0) -> Tuple[List[Any]]:
        """Separates a single cluster from a geocell."""
        # Create polygon map
        polygons = self.voronoi_polygons()

        # Reindex df to match point order
        df = df.reset_index(drop=True)  # Ensure index starts from 0
        if len(df) != len(polygons):
            raise ValueError(f"Length mismatch: {len(df)} points vs {len(polygons)} polygons in cell {self.admin_2}")

        # Get only the points in the specified cluster
        cluster_df = df[df['cluster'] == cluster][['lon', 'lat']]
        if cluster_df.empty:
            raise ValueError(f'Dataframe does not contain cluster {cluster} in cell {self.admin_2}')

        # Get positions of these rows (row numbers after reset)
        cluster_indices = cluster_df.index.to_list()
        
        # Select points and polygons based on positional index
        cluster_points = [self.points[i] for i in cluster_indices]
        cluster_polys = [polygons[i] for i in cluster_indices]

        # Create new cell
        new_cell = self.__separate_points(cluster_points, cluster_polys, contain_points=True)
        return [new_cell], []

    def _separate_multi_cluster(self, df: pd.DataFrame, non_null_large_clusters: List[int]) -> List[Any]:
        """Separates multiple cluster from a geocell.

        Args:
            df (pd.DataFrame): Dataframe of points.
            non_null_large_clusters (pd.Series): Large clusters that are not unassigned.

        Returns:
            List[Any]: New cells.
        """
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

            return new_cells, [self]

    def _split_cell(self, add_to: Any, min_samples: int, min_cell_size: int,
                    max_cell_size: int) -> List[Any]:
        """
        The method can split a single Cell into two or more new cells,
        depending on how many valid clusters are found by HDBSCAN.
        """

        # No need to split, already below threshold
        if self.size < max_cell_size:
            return []

        df = pd.DataFrame(data=self.coords, columns=['lon', 'lat'])
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS)

        # Creates clusters within the geocell
        clusterer = HDBSCAN(min_cluster_size=min_cell_size, min_samples=min_samples)
        df['cluster'] = clusterer.fit_predict(df[['lon', 'lat']].values)
        
        unique_clusters = df['cluster'].nunique()
        if unique_clusters < 2:
            return []

        # If the cluster resulted in a size that's less than the minimum, it's an invalid cluster
        cluster_counts = df['cluster'].value_counts()
        small_clusters = cluster_counts[cluster_counts < min_cell_size].index.tolist()
        df.loc[df['cluster'].isin(small_clusters), 'cluster'] = -1
        
        # Identify all valid large clusters
        cluster_counts = df['cluster'].value_counts()
        large_clusters = cluster_counts[cluster_counts >= min_cell_size].index
        non_null_large_clusters = [
            c for c in large_clusters 
            if c != -1
        ]

        # There must be at least 2 large clusters
        if len(large_clusters) < 2:
            return []

        # Special case: 1 real large cluster, and 1 cluster that's just noise. 
        # We're okay with splitting further if the "noise" cluster isn't too large.
        if len(large_clusters) == 2 and len(non_null_large_clusters) == 1:
            null_df = df[df['cluster'] == -1]
            if len(null_df) > max_cell_size:
                return []

            # Donut split: one dense cluster, and some noise surrounding it
            # Extracts the real cluster and gives it a voronoi polygon
            new_cells, remove_cells = self._separate_single_cluster(df, non_null_large_clusters[0])
        else:
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
            add_to.add(cell)

        # Remove dead cells from CellCollection
        for cell in remove_cells:
            add_to.remove(cell)

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

        self.__clean_dirty_splits(clean_cells)

        """
        Are there any more cells that need to be split up again?
        If so, add them to cell_to_split. 
        """

        cells_to_split = []

        # First, check original cell. If it's too big, split again
        if self.size > max_cell_size and self not in remove_cells:
            cells_to_split.append(self)

        # Next, check new cells. If they're too big, split again.
        for cell in new_cells:
            if cell.size > max_cell_size:
                cells_to_split.append(cell)

        return cells_to_split

    def __clean_dirty_splits(self, cells: List[Any]):
        """Cleans messy splits that split polygons into multiple parts.
        """
        df = pd.DataFrame(data=[x.tolist() for x in cells], columns=GEOCELL_COLUMNS)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=CRS)

        # Identifying Multipolygons
        multi_polys = df[df['geometry'].type == 'MultiPolygon']

        # Iterate through rows with Multipolygons
        for index, row in multi_polys.iterrows():

            # Find points
            points = cells[index].to_geopandas()['geometry'] # .to_crs('EPSG:3857')
            
            # Splitting Multipolygons
            all_polygons = list(row['geometry'].geoms)
            
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
                    
                    # Union the small polygon with the polygon having largest common border
                    cells[largest_intersect_index]._polygons = [cells[largest_intersect_index].shape.union(small_poly)]

            if did_assign:
                # Keeping the largest polygon in the original GeoDataFrame
                cells[index]._polygons = [largest_poly]

    def __eq__(self, other):
        return self.admin_2 == other.admin_2
    
    def __ne__(self, other):
        return self.admin_2 != other.admin_2

    def __hash__(self):
        return hash(self.admin_2)
        
    def __repr__(self):
        rep = f'Cell(admin_2={self.admin_2}, admin_1={self.admin_1}, admin_0={self.admin_0}, size={len(self.points)}, num_polys={len(self.polygons)})'
        return rep
        
    def __str__(self):
        return self.__repr__()