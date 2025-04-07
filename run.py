import argparse
import pandas as pd
import s3fs
import os

from dataset_creation.geocell import GeocellCreator, parallelize_fusing
from config import LOC_PATH, GEOCELL_PATH, MIN_CELL_SIZE, MAX_CELL_SIZE 

ADMINS_FOLDER_PATH = "data/geocells/admin_data/"
FUSED_GEOCELL_PATH = "data/geocells/cells/inat2017_fused_cells.npy"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate geocells from administrative boundaries.")
    parser.add_argument('--locations_path', default=LOC_PATH)
    parser.add_argument('--admins_folder_path', default=ADMINS_FOLDER_PATH)
    parser.add_argument('--fused_geocell_path', default=FUSED_GEOCELL_PATH)
    parser.add_argument('--min_cell_size', type=int, default=MIN_CELL_SIZE)
    parser.add_argument('--max_cell_size', type=int, default=MAX_CELL_SIZE)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())

    return parser.parse_args()

def main():
    args = parse_args()
    print("Beginning geocell creation algorithm...")

    # Load from S3 or local
    df = pd.read_csv(args.csv)

    geocell_creator = GeocellCreator(df, 'data/geocells/cells/inat2017_cells.csv')
    cells = geocell_creator.initialize_cells(args.min_cell_size)
    cells = parallelize_fusing(cells, num_workers=args.num_workers)
    cells.save(args.geocell_path)    


    
if __name__ == '__main__':
    main()

