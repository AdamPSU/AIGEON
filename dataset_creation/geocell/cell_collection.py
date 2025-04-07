import concurrent.futures
import numpy as np
import pandas as pd
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Iterable, Any, List
from shapely.affinity import scale
from .cell import Cell

import time

CRS = 'EPSG:4326'
GEOCELL_COLUMNS = ['admin_2', 'admin_1', 'admin_0', 'size', 'num_polygons', 'geometry']
OPTIC_PARAMS = [(300, 0.05), (400, 0.005), (1000, 0.0001)]

class CellCollection(set):
    def __init__(self, cells: Iterable[Cell]):
        """A collection of geocells and a wrapper around a set of geocells.

        Args:
            cells (Iterable[Cell]): cells to create the CellCollection from.
        """

        cells = [cell for cell in cells if not cell.empty]
        super(CellCollection, self).__init__(set(cells))
        
    @property
    def countries(self):
        return sorted(list(set([cell.admin_0 for cell in self])))

    def clean(self) -> Any:
        """Removes empty cells.

        Returns:
            Any: Cleaned CellCollection
        """
        return CellCollection([cell for cell in self if not cell.empty])
    
    def find(self, admin_2: str) -> Cell:
        """Finds the geocells with the given id in a collection.

        Args:
            admin_2 (str): admin id to search for.

        Returns:
            Cell: geocell
        """
        if type(admin_2) == int:
            admin_2 = str(admin_2)

        for cell in self:
            if admin_2 == cell.admin_2:
                return cell

        raise KeyError(f'Cell {admin_2} is not in collection.')

    def copy(self) -> Any:
        """Creates copy of cells.

        Args:
            cell_list (List): list of geocells

        Returns:
            List: copy of list of geocells
        """
        return CellCollection([Cell(x.admin_2, x.admin_1, x.admin_0, x.points, x.polygons) \
                               for x in self])

    def to_geopandas(self) -> gpd.GeoDataFrame:
        """Converts a list of cells to a geopandas DataFrame.

        Args:
            admin_0 (str, optional): admin_0 to filter by. Defaults to None.

        Returns:
            gpd.GeoDataFrame: geopandas DataFrame.
        """

        cells = [cell for cell in self if not cell.empty]

        # Convert to DataFrame and then to GeoDataFrame
        df = pd.DataFrame([cell.tolist() for cell in cells], columns=GEOCELL_COLUMNS)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=CRS)

        return gdf

    def save(self, output_file: str):
        """Saves the cell collection to file.

        Args:
            output_file (str): Output filename.
        """
        np.save(output_file, list(self))

    def overwrite(self, file: str):
        """Overwrites current CellCollection with file.

        Args:
            file (str): Filename to load.
        """

        collection = np.load(file, allow_pickle=True)
        cs = [x for x in collection if not x.empty]
        super(CellCollection, self).__init__(set(cs))
        print('Overwrote with contents of:', file)

    @classmethod
    def load(cls, file: str):
        """Load a CellCollection from file.

        Args:
            file (str): Filename to load.
        """

        return cls(set(np.load(file, allow_pickle=True)))

    def balance(self, min_cell_size: int, max_cell_size: int):
        """Balances all contained cells such that most cells are not smaller than min_cell_size.

        Args:
            min_cell_size: (int): Minimum cell size.
            max_cell_size (int): Minimum cell size.
        """
        countries = self.countries[::-1]

        self._split_geocells(min_cell_size, max_cell_size)
        self.save('data/geocells/cells/inat2017_gcell_collection.npy')

    def _split_geocells(self, min_cell_size: int, max_cell_size: int):
        """Split large geocells into cells smaller or equal to max_cell_size.

        Args:
            min_cell_size: (int): Minimum cell size.
            max_cell_size (int): Maximum cell size.
        """

        for args in OPTIC_PARAMS:
            print('||| NEW OPTICS PARAMS ||| ', args)
            new_cells = []

            large_cells = [x for x in self if x.size > max_cell_size]
            round = 1
            while len(large_cells) > 0:

                # Progress bar
                desc = f'Round {round}: splitting large cells'
                pbar = tqdm(total=len(large_cells), desc=desc, dynamic_ncols=True, unit='cell')

                # Parallalize the splitting of cells across cores
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(cell._split_cell, self, args, min_cell_size, max_cell_size) for cell in large_cells]
                    for future in concurrent.futures.as_completed(futures):
                        nc = future.result()
                        new_cells.extend(nc)
                        pbar.update(1)

                    concurrent.futures.wait(futures)

                # Update variables
                large_cells = new_cells
                new_cells = []
                round += 1

                # Close the progress bar
                pbar.close()

    def __sub__(self, other):
        return CellCollection(super().__sub__(other))
    
    def __add__(self, other):
        return CellCollection(self.union(other))