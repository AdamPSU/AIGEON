import pandas as pd 
import json 

from pathlib import Path 

def create_geocaptions(df): 
    annotations = {}
    prefix = Path('data/inaturalist_2017/inaturalist_2017_subset/')

    for _, row in df.iterrows():
        # Join paths 
        relative_path = Path(row['file_name'].removeprefix("train_val_images/"))
        file_path = prefix / relative_path 

        if not file_path.exists():
            continue  # Skip if the file doesn't exist

        species = row['species']
        lat = row['lat']
        lon = row['lon']
        country = row['country_code']
        
        prompt = f"a photo of a {species} in {country}, located at latitude {lat:.3f} and longitude {lon:.3f}"
        annotations[str(file_path)] = prompt

    with open('data/inaturalist_2017/processed/geocaptions.json', 'w') as f: 
        json.dump(annotations, f, indent=2)

if __name__ == '__main__': 
    df = pd.read_csv('data/inaturalist_2017/processed/metadata.csv') 
    create_geocaptions(df)
    

