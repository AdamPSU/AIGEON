from typing import Any, Dict, List, Tuple
import logging; logging.basicConfig(level=logging.INFO, format='%(message)s')
import time 

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.wkt
import shapely.ops
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.errors import TopologicalError
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from hdbscan import HDBSCAN
from scipy.spatial import Voronoi

from config import (
    CRS,
    ADMIN_2_PATH,
    LOC_PATH,
    GEOCELL_PATH,
    MIN_CELL_SIZE,
    MAX_CELL_SIZE
)

from dataset_creation.geocell.geocell_creation import GeocellCreator
from dataset_creation.geocell.cell_collection import CellCollection
from dataset_creation.geocell.voronoi import voronoi_finite_polygons

HDBSCAN_PARAMS = [100, 250, 500]

fused_cells = np.load('data/geocells/cells/inat2017_fused_cells.npy', allow_pickle=True)

cells = [
    cell for cell in fused_cells 
    if cell.admin_2 == '43285'
]

cell = CellCollection(cells)

def cell_splitter(cells:CellCollection, min_cell_size: int, max_cell_size: int, num_workers: int):
    """Split large geocells into cells smaller or equal to max_cell_size.

    Args:
        min_cell_size: (int): Minimum cell size.
        max_cell_size (int): Maximum cell size.
    """
    
    for params in HDBSCAN_PARAMS:
        new_cells = []

        large_cells = [
            cell for cell in cells 
            if cell.size > max_cell_size
        ]
        round = 1

        while len(large_cells) > 0:
            # Progress bar
            desc = f'Round {round} of splitting large cells, trying min_sample_size = {params}'
            # pbar = tqdm(total=len(large_cells), desc=desc, dynamic_ncols=True, unit='cell')
            
            for cell in large_cells:
                t0 = time.time() 
                logging.info(f"Processing {cell}")

                nc = cell._split_cell(cells, params, min_cell_size, max_cell_size)
                new_cells.extend(nc)

                logging.info(f"Processed cell. Time: {(time.time() - t0):.3f}s")
       
            # # Parallelize the splitting of cells across cores
            # with ThreadPoolExecutor(max_workers=num_workers) as executor:
            #     futures = [
            #         executor.submit(cell._split_cell, cells, params, min_cell_size, max_cell_size) 
            #         for cell in large_cells
            #     ]
                
            #     for future in as_completed(futures):
            #         nc = future.result()
            #         new_cells.extend(nc)
            #         # pbar.update(1)

            # # Update variables
            # large_cells = new_cells
            # new_cells = []
            # round += 1

            # pbar.close()

    return cells

split_cells = cell_splitter(cell, min_cell_size=MIN_CELL_SIZE, max_cell_size=MAX_CELL_SIZE, num_workers=1)