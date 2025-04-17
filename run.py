import argparse
import pandas as pd
import os

from dataset_creation.geocell import GeocellCreator
from config import (LOC_PATH, FUSED_GEOCELL_PATH, GEOCELL_PATH, 
    ADMIN_2_PATH, MIN_CELL_SIZE, MAX_CELL_SIZE)

def parse_args():
    # args for GeocellCreator
    parser = argparse.ArgumentParser(description="Generate geocells from administrative boundaries.")
    parser.add_argument('--locations_path', default=LOC_PATH)
    parser.add_argument('--admin2_path', default=ADMIN_2_PATH)
    parser.add_argument('--fused_geocell_path', default=FUSED_GEOCELL_PATH)
    parser.add_argument('--output_path', type=str, default=GEOCELL_PATH)

    # args for GeocellCreator.generate()
    parser.add_argument('--min_cell_size', type=int, default=MIN_CELL_SIZE)
    parser.add_argument('--max_cell_size', type=int, default=MAX_CELL_SIZE)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--load_fused', action='store_true', help='Load fused geocells if available')
    parser.add_argument('--no_load_fused', dest='load_fused', action='store_false')
    parser.set_defaults(load_fused=True)

    return parser.parse_args()

def main():
    args = parse_args()

    loc_df = pd.read_csv(args.locations_path)
    geocell_creator = GeocellCreator(
        df=loc_df, 
        admin2_path=args.admin2_path, 
        output_file=args.output_path
    )
    geocell_creator.generate(
        min_cell_size=args.min_cell_size, 
        max_cell_size=args.max_cell_size, 
        num_workers=args.num_workers,
        load_fused=args.load_fused
    )
    
if __name__ == '__main__':
    main()

