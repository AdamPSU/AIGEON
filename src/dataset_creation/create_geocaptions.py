import pandas as pd 
import json 

def create_geocaptions(df): 
    annotations = {}
    prefix = 'data/inaturalist_2017/inaturalist_2017_subset/'

    for _, row in df.iterrows():
        file_name = prefix + row['file_name'].removeprefix("train_val_images/")
        species = row['species']
        lat = row['lat']
        lon = row['lon']
        country = row['country_code']

        prompt = f"a photo of a {species} in {country}, located at latitude {lat:.3f} and longitude {lon:.3f}"
        annotations[file_name] = prompt

    with open('data/inaturalist_2017/processed/geocaptions.json', 'w') as f: 
        json.dump(annotations, f, indent=2)

if __name__ == '__main__': 
    df = pd.read_csv('data/inaturalist_2017/processed/metadata.csv') 
    create_geocaptions(df)
    

