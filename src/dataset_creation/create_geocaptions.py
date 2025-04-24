import pandas as pd
import json
import boto3
from pathlib import Path
from tqdm import tqdm

def create_geocaptions(metadata_path, local_image_prefix, s3_image_prefix, output_path):
    # Load metadata CSV (locally)
    df = pd.read_csv(metadata_path)
    df['file_name'] = df['file_name'].str.removeprefix('train_val_images/')

    annotations = {}
    s3 = boto3.client("s3")

    # tqdm wraps the iterator to show progress
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating geocaptions"):
        relative_path = row['file_name']
        local_path = Path(local_image_prefix) / relative_path

        if not local_path.exists():
            continue  # Skip if local file doesn't exist

        # Compose final S3 path for the annotation key
        s3_image_path = f"{s3_image_prefix}/{relative_path}"

        species = row["species"]
        lat = row["lat"]
        lon = row["lon"]
        country = row["country_code"]

        prompt = f"a photo of a {species} in {country}, located at latitude {lat:.3f} and longitude {lon:.3f}"
        annotations[s3_image_path] = prompt

    # Write annotations JSON to S3
    annotations_json = json.dumps(annotations, indent=2)
    bucket, key = output_path.replace("s3://", "").split("/", 1)
    s3.put_object(Bucket=bucket, Key=key, Body=annotations_json.encode("utf-8"))

    print(f"Geocaptions uploaded to {output_path}")

if __name__ == '__main__':
    create_geocaptions(
        metadata_path="data/inaturalist_2017/processed/metadata.csv",  # local file
        local_image_prefix="data/inaturalist_2017/inaturalist_2017_subset",  # check here
        s3_image_prefix="s3://animaldex/inaturalist_2017/inaturalist_2017_subset",  # write this
        output_path="s3://animaldex/inaturalist_2017/processed/geocaptions.json"
    )
