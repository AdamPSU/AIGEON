import pandas as pd
import json
import boto3
from tqdm import tqdm


def create_geocaptions(metadata_path: str, s3_image_prefix: str, output_path: str):
    """
    Generate geocaptions for images based on metadata and save them to S3.

    Args:
        metadata_path: Path to the local CSV file containing image metadata.
        s3_image_prefix: Prefix for image paths in S3 (e.g. s3://bucket/folder).
        output_path: Full S3 path where the geocaptions JSON should be saved.
    """

    # Load metadata
    df = pd.read_csv(metadata_path)
    df['file_name'] = df['file_name'].str.removeprefix('train_val_images/')

    # Setup S3
    s3 = boto3.client("s3")
    annotations = {}

    # Generate prompts
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating geocaptions"):
        relative_path = row['file_name']
        s3_image_path = f"{s3_image_prefix}/{relative_path}"

        prompt = (
            f"This is a photo I took of a {row['species']} in {row['state']}, {row['country']} during {row['season']}. "
            f"This location has {row['climate']} climate."
        )

        annotations[s3_image_path] = prompt

    # Save annotations to S3
    annotations_json = json.dumps(annotations, indent=2)
    bucket, key = output_path.replace("s3://", "").split("/", 1)
    s3.put_object(Bucket=bucket, Key=key, Body=annotations_json.encode("utf-8"))

    print(f"✅ Geocaptions uploaded to {output_path}")


if __name__ == '__main__':
    create_geocaptions(
        metadata_path="data/annotations/metadata.csv",
        s3_image_prefix="s3://animaldex/inaturalist_2017/train_val_images",
        output_path="s3://animaldex/annotations/geocaptions.json"
    )
