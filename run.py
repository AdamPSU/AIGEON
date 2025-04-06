import argparse
import pandas as pd
from dataset_creation.geocell import GeocellCreator
from config import LOC_PATH, GEOCELL_PATH, MIN_CELL_SIZE, MAX_CELL_SIZE 

def parse_args():
    parser = argparse.ArgumentParser(description="Generate geocells from a locations.csv file.")
    parser.add_argument('--csv', default=LOC_PATH, help='Path to the CSV file containing lat/lon columns.')
    parser.add_argument('--min_cell_size', type=int, default=MIN_CELL_SIZE, help='Minimum cell size.')
    parser.add_argument('--max_cell_size', type=int, default=MAX_CELL_SIZE, help='Maximum cell size.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load location data
    df = pd.read_csv(args.csv)
    min_cell_size = args.min_cell_size 
    max_cell_size = args.max_cell_size

    # Run geocell creation
    creator = GeocellCreator(df, GEOCELL_PATH)
    creator.generate(min_cell_size, max_cell_size)

if __name__ == '__main__':
    main()

