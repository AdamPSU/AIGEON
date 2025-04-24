import pandas as pd
import json
import boto3
from io import BytesIO
from PIL import Image
from smart_open import open as s3_open  # smart_open handles S3 URLs

def create_geocaptions(metadata_s3_path, image_prefix_s3, output_s3_path):
    # Load metadata from S3
    df = pd.read_csv(metadata_s3_path)

    annotations = {}
    s3 = boto3.client("s3")

    for _, row in df.iterrows():
        relative_path = row["file_name"].removeprefix("train_val_images/")
        s3_image_path = f"{image_prefix_s3}/{relative_path}"

        try:
            with s3_open(s3_image_path, 'rb') as f:
                # This verifies the file exists and can be opened
                Image.open(f).convert("RGB")
        except Exception as e:
            continue  # skip files that can't be accessed or opened

        species = row["species"]
        lat = row["lat"]
        lon = row["lon"]
        country = row["country_code"]

        prompt = f"a photo of a {species} in {country}, located at latitude {lat:.3f} and longitude {lon:.3f}"
        annotations[s3_image_path] = prompt

    # Write annotations back to S3
    annotations_json = json.dumps(annotations, indent=2)
    bucket, key = output_s3_path.replace("s3://", "").split("/", 1)
    s3.put_object(Bucket=bucket, Key=key, Body=annotations_json.encode("utf-8"))

    print(f"Geocaptions uploaded to {output_s3_path}")

if __name__ == '__main__':
    create_geocaptions(
        metadata_s3_path="s3://animaldex/inaturalist_2017/processed/metadata.csv",
        image_prefix_s3="s3://animaldex/inaturalist_2017/inaturalist_2017_subset",
        output_s3_path="s3://animaldex/inaturalist_2017/processed/geocaptions.json"
    )
