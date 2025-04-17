import logging
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
from config import CRS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} took {time() - start:.2f}s")
        return result
    return wrapper

CELL_COLUMNS = ['id', 'lon', 'lat']
GEOCELL_COLUMNS = ['admin_2', 'admin_1', 'admin_0', 'size', 'num_polygons', 'geometry']

class Cell:
    """Abstraction of a geocell.
    """
    def __init__(self, admin_2: str, admin_1: str, admin_0: str,
                 points: List[Point], polygons: List[Polygon]):
        self.admin_2 = str(admin_2)
        self.admin_1 = str(admin_1)
        self.admin_0 = str(admin_0)
        self._points = points
        self._polygons = [polygons] if isinstance(polygons, Polygon) else list(polygons)

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def shape(self) -> Polygon:
        union = shapely.ops.unary_union(self.polygons)
        return union.buffer(0)

    @property
    def points(self) -> List[Point]:
        try:
            return [shapely.wkt.loads(x) for x in self._points]
        except TypeError:
            return self._points

    @property
    def polygons(self) -> List[Polygon]:
        try:
            return [shapely.wkt.loads(x) for x in self._polygons]
        except TypeError:
            return self._polygons

    @property
    def multi_point(self) -> MultiPoint:
        return MultiPoint(self.points)

    @property
    def coords(self) -> np.ndarray:
        return np.array([[x.x, x.y] for x in self.points])

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.coords, axis=0)

    @property
    def empty(self) -> bool:
        return len(self.points) == 0

    def subtract(self, other: Any):
        try:
            diff_shape = self.shape.difference(other.shape)
        except TopologicalError as e:
            print(f'Error occurred during subtracting in cell: {self.admin_2}')
            raise TopologicalError(e)

        self._polygons = [diff_shape.buffer(0)]
        A_tuples = {(p.x, p.y) for p in self.points}
        B_tuples = {(p.x, p.y) for p in other.points}
        difference_tuples = A_tuples - B_tuples
        self._points = [x for x in self.points if (x.x, x.y) in difference_tuples]

    def combine(self, others: List):
        for other in others:
            if other is self:
                print('Tried to combine cell with itself')
                continue
            self.add_points(other.points)
            self.add_polygons(other.polygons)
            other._points = []
            other._polygons = []

    def add_polygons(self, polygons: List[Polygon]):
        self._polygons += polygons

    def add_points(self, points: List[Point]):
        try:
            self._points += points
        except TypeError:
            self._points += points.tolist()

    def tolist(self) -> List:
        return [self.admin_2, self.admin_1, self.admin_0, len(self.points), len(self.polygons), self.shape]

    def to_pandas(self) -> gpd.GeoDataFrame:
        data = [[self.admin_2, p.x, p.y] for p in self.points]
        df = pd.DataFrame(data=data, columns=CELL_COLUMNS)
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS)

    @timeit
    def __separate_points(self, points: List[Point], polygons: List[Polygon], contain_points: bool) -> Any:
        logger.info(f"Creating new cell from {len(points)} points in {self.admin_2}")
        coords = tuple((p.x, p.y) for p in points)
        new_name = str(hash(coords))[:12]
        new_shape = shapely.ops.unary_union(polygons).buffer(0)
        if contain_points and not isinstance(new_shape, MultiPolygon):
            new_shape = Polygon(new_shape.exterior)
        return Cell(new_name, self.admin_1, self.admin_0, points, [new_shape])

    @timeit
    def voronoi_polygons(self, coords: np.ndarray = None) -> List[Polygon]:
        logger.info(f"Generating Voronoi polygons for cell {self.admin_2}")
        v_coords = np.unique(coords, axis=0) if coords is not None else np.unique(self.coords, axis=0)
        vor = Voronoi(v_coords)
        regions, vertices = voronoi_finite_polygons(vor)
        polys = [Polygon(vertices[region]) for region in regions]
        try:
            polys = [x.intersection(self.shape) for x in polys]
        except TopologicalError:
            print(f'Error occurred in cell: {self.admin_2}')
            raise
        df = gpd.GeoDataFrame({'geometry': polys}, geometry='geometry')
        points = [Point(p[0], p[1]) for p in coords] if coords is not None else self.points
        indices = df.sindex.nearest(points, return_all=False)[1]
        return [polys[i] for i in indices]

    @timeit
    def _separate_single_cluster(self, df: pd.DataFrame, cluster: int = 0) -> Tuple[List[Any]]:
        logger.info(f"Separating single cluster {cluster} from cell {self.admin_2}")
        polygons = self.voronoi_polygons()
        cluster_df = df[df['cluster'] == cluster][['lon', 'lat']]
        assert len(cluster_df.index) > 0, 'Dataframe does not contain a cluster'
        cluster_points = [self.points[i] for i in cluster_df.index]
        cluster_polys = [polygons[i] for i in cluster_df.index]
        new_cell = self.__separate_points(cluster_points, cluster_polys, contain_points=True)
        return [new_cell], []

    @timeit
    def _separate_multi_cluster(self, df: pd.DataFrame, non_null_large_clusters: List[int]) -> List[Any]:
        logger.info(f"Separating multiple clusters from cell {self.admin_2}")
        assigned_df = df[df['cluster'].isin(non_null_large_clusters)]
        unassigned_df = df[~df['cluster'].isin(non_null_large_clusters)]
        cc = assigned_df.groupby(['cluster'])[['lon', 'lat']].mean().reset_index()
        cc = gpd.GeoDataFrame(cc, geometry=gpd.points_from_xy(cc.lon, cc.lat), crs=CRS)
        nearest_index = cc.sindex.nearest(unassigned_df.geometry, return_all=False)[1]
        df.loc[~df['cluster'].isin(non_null_large_clusters), 'cluster'] = cc.iloc[nearest_index]['cluster'].values
        if len(cc.index) == 2:
            return self._separate_single_cluster(df, cluster=cc.iloc[0]['cluster'])
        else:
            polygons = self.voronoi_polygons(coords=cc[['lon', 'lat']].values)
            new_cells = []
            for cluster, polygon in zip(cc['cluster'].unique(), polygons):
                cluster_coords = df[df['cluster'] == cluster][['lon', 'lat']]
                cluster_points = [Point(row.lon, row.lat) for _, row in cluster_coords.iterrows()]
                new_cell = self.__separate_points(cluster_points, [polygon], contain_points=True)
                new_cells.append(new_cell)
            return new_cells, [self]

    @timeit
    def _split_cell(self, cell_collection: Any, min_samples: int, min_cell_size: int, max_cell_size: int) -> List[Any]:
        if self.size < max_cell_size:
            return []
        logger.info(f"Splitting cell {self.admin_2} with size {self.size}")
        df = pd.DataFrame(data=self.coords, columns=['lon', 'lat'])
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS)
        clusterer = HDBSCAN(min_cluster_size=min_cell_size, min_samples=min_samples)
        df['cluster'] = clusterer.fit_predict(df[['lon', 'lat']].values)
        unique_clusters = df['cluster'].nunique()
        if unique_clusters < 2:
            return []
        cluster_counts = df['cluster'].value_counts()
        small_clusters = cluster_counts[cluster_counts < min_cell_size].index.tolist()
        df.loc[df['cluster'].isin(small_clusters), 'cluster'] = -1
        cluster_counts = df['cluster'].value_counts()
        large_clusters = cluster_counts[cluster_counts >= min_cell_size].index
        non_null_large_clusters = [c for c in large_clusters if c != -1]
        if len(large_clusters) < 2:
            return []
        if len(large_clusters) == 2 and len(non_null_large_clusters) == 1:
            null_df = df[df['cluster'] == -1]
            if len(null_df) > max_cell_size:
                return []
            new_cells, remove_cells = self._separate_single_cluster(df, non_null_large_clusters[0])
        else:
            new_cells, remove_cells = self._separate_multi_cluster(df, non_null_large_clusters)
        for cell in new_cells:
            self.subtract(cell)
            cell_collection.add(cell)
        for cell in remove_cells:
            cell_collection.remove(cell)
        clean_cells = new_cells
        if not len(remove_cells):
            clean_cells += [self]
        self.__clean_dirty_splits(clean_cells)
        cells_to_split = []
        if self.size > max_cell_size and self not in remove_cells:
            cells_to_split.append(self)
        for cell in new_cells:
            if cell.size > max_cell_size:
                cells_to_split.append(cell)
        return cells_to_split

    @timeit
    def __clean_dirty_splits(self, cells: List[Any]):
        logger.info(f"Cleaning dirty splits for {len(cells)} cells")
        df = pd.DataFrame(data=[x.tolist() for x in cells], columns=GEOCELL_COLUMNS)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs=CRS)
        multi_polys = df[df['geometry'].type == 'MultiPolygon']
        for index, row in multi_polys.iterrows():
            points = cells[index].to_pandas()['geometry']
            all_polygons = list(row['geometry'].geoms)
            largest_poly = max(all_polygons, key=lambda polygon: polygon.area)
            did_assign = False
            for small_poly in all_polygons:
                if small_poly != largest_poly:
                    small_poly_gseries = gpd.GeoSeries([small_poly], index=[index], crs=CRS)
                    other_polys = df.drop(index)
                    buffered_poly = small_poly_gseries.buffer(0.01)
                    intersecting_polys = other_polys[other_polys.intersects(buffered_poly.unary_union)]
                    if len(intersecting_polys) == 0:
                        continue
                    did_assign = True
                    largest_intersect_index = intersecting_polys.geometry.apply(
                        lambda poly: poly.intersection(buffered_poly.unary_union).area
                    ).idxmax()
                    mask = points.within(small_poly)
                    points_in_small_poly = points[mask]
                    cells[index]._points = [x for x in cells[index].points if x not in points_in_small_poly]
                    cells[largest_intersect_index].add_points(points_in_small_poly)
                    cells[largest_intersect_index]._polygons = [cells[largest_intersect_index].shape.union(small_poly)]
            if did_assign:
                cells[index]._polygons = [largest_poly]

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