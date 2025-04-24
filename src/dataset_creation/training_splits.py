import boto3
import random
import argparse
from urllib.parse import urlparse
from tqdm import tqdm


def list_all_images(bucket: str, prefix: str, suffix: str = ".jpg") -> list:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    all_images = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(suffix):
                all_images.append(f"s3://{bucket}/{key}")
    return all_images

def split_dataset(image_paths: list, pretrain_size: int = 1000, total_sample: int = 2000) -> tuple:
    if total_sample < len(image_paths):
        image_paths = random.sample(image_paths, total_sample)

    random.shuffle(image_paths)
    return image_paths[:pretrain_size], image_paths[pretrain_size:]


def copy_to_s3(image_paths: list, dest_root_prefix: str, base_prefix: str, bucket: str):
    s3 = boto3.client("s3")

    for path in tqdm(image_paths, desc=f"Uploading to {dest_root_prefix}"):
        parsed = urlparse(path)
        src_key = parsed.path.lstrip("/")

        if not src_key.startswith(base_prefix):
            raise ValueError(f"Key '{src_key}' does not start with expected base prefix '{base_prefix}'")

        # Derive relative path and form destination key
        rel_path = src_key[len(base_prefix):]  # e.g. Insecta/Danaus/image.jpg
        dest_key = f"{dest_root_prefix}/{rel_path}"

        s3.copy_object(
            Bucket=bucket,
            CopySource={'Bucket': bucket, 'Key': src_key},
            Key=dest_key
        )


def main(full_dataset: bool = False):
    bucket = "animaldex"
    base_prefix = "inaturalist_2017/train_val_images/"
    pretrain_prefix = "subset/pretrain"
    finetune_prefix = "subset/finetune"

    print("Listing image paths from S3...")
    image_paths = list_all_images(bucket, base_prefix)

    if full_dataset:
        print("Splitting entire dataset...")
        pretrain, finetune = split_dataset(image_paths, pretrain_size=len(image_paths)//2, total_sample=len(image_paths))
    else:
        print("Sampling 2,000 images...")
        pretrain, finetune = split_dataset(image_paths)

    print("Copying pretraining images...")
    copy_to_s3(pretrain, pretrain_prefix, base_prefix, bucket)

    print("Copying finetuning images...")
    copy_to_s3(finetune, finetune_prefix, base_prefix, bucket)

    print("✅ Upload complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use the full dataset (instead of a 2,000 image sample)")
    args = parser.parse_args()

    main(full_dataset=args.full)
