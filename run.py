import argparse
import pandas as pd
import multiprocessing as mp 
from dataset_creation.geocell import GeocellCreator, parallelize_fusing
from config import LOC_PATH, GEOCELL_PATH, MIN_CELL_SIZE, MAX_CELL_SIZE 

import logging

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate geocells from a locations.csv file.")
    parser.add_argument('--csv', default=LOC_PATH, help='Path to the CSV file containing lat/lon columns.')
    parser.add_argument('--min_cell_size', type=int, default=MIN_CELL_SIZE, help='Minimum cell size.')
    parser.add_argument('--max_cell_size', type=int, default=MAX_CELL_SIZE, help='Maximum cell size.')
    return parser.parse_args()

def main():
    print("Beginning geocell creation algorithm...")
    df = pd.read_csv(LOC_PATH)
    geocell_creator = GeocellCreator(df, 'data/geocells/cells/inat2017_cells.csv')
    cells = geocell_creator.initialize_cells(MIN_CELL_SIZE)
    cells = parallelize_fusing(cells, num_workers=8)
    cells.save('data/geocells/cells/inat2017_fused_cells.npy')

if __name__ == '__main__':
    main()

