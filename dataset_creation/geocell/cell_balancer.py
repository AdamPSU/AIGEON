import warnings; warnings.filterwarnings("ignore")
import numpy as np

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait
from tqdm import tqdm

from .cell_collection import CellCollection

# Constants
HDBSCAN_PARAMS = [100, 250, 500]

def cell_fuser(granular_cells: CellCollection, min_cell_size: int, num_workers: int):
    all_cells = []

    # Prepare the per-country cell groups
    grouped_country_cells = [
        CellCollection([
            cell for cell in granular_cells
            if cell.admin_0 == country
        ])
        for country in tqdm(granular_cells.countries, desc="Dividing into countries")
    ]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_fuse_within_country, group, min_cell_size) 
            for group in grouped_country_cells
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fusing admin 2 cells within countries."):
            result = future.result()  # Raises if an exception occurred in worker
            all_cells.extend(result)

    return CellCollection(all_cells)  

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
            pbar = tqdm(total=len(large_cells), desc=desc, dynamic_ncols=True, unit='cell')

            # Parallelize the splitting of cells across cores
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(cell._split_cell, cells, params, min_cell_size, max_cell_size) 
                    for cell in large_cells
                ]
                
                for future in as_completed(futures):
                    nc = future.result()
                    new_cells.extend(nc)
                    pbar.update(1)

            wait(futures)

            # Update variables
            large_cells = new_cells
            new_cells = []
            round += 1

            pbar.close()

        return cells
    
def _get_candidates(center_row, potential_df, min_cell_size, admin_filter=True, small_filter=False):
    candidates = potential_df
    if admin_filter:
        candidates = candidates[candidates['admin_1'] == center_row['admin_1']]
    if small_filter:
        candidates = candidates[candidates['size'] < min_cell_size]
    return candidates

def _fuse_within_country(cells: CellCollection, min_cell_size: int) -> CellCollection:
    excluded_ids = set()
    cell_df = cells.to_geopandas().set_index('admin_2')

    while True:
        consider_df = cell_df[~cell_df.index.isin(excluded_ids)]
        df_small = consider_df[consider_df['size'] < min_cell_size]

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

                candidates = _get_candidates(center_row, potential_neighbors, min_cell_size, admin_filter, small_filter)
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

        for neighbor_id, _ in neighbors.iterrows():
            if total_size >= min_cell_size:
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