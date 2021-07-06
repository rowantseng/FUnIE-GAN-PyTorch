import argparse
import glob
import json
import os

import numpy as np


def split_data(image_path, prefix=""):
    paths = glob.glob(f"{image_path}/*.*")
    base_names = list(map(os.path.basename, paths))
    base_names = np.array(base_names)

    # Set data length for valid splits
    total_len = len(base_names)
    print(f"Total {total_len} data")
    valid_len = int(total_len * 0.1)

    # Create data splits
    indices = np.random.permutation(total_len)
    train_indices = indices[valid_len:]
    valid_indices = indices[:valid_len]

    train_names = base_names[train_indices].tolist()
    valid_names = base_names[valid_indices].tolist()
    train_names = list(map(lambda x: f"{prefix}{x}", train_names))
    valid_names = list(map(lambda x: f"{prefix}{x}", valid_names))
    print(f"Train has {len(train_names)} data from {image_path}")
    print(f"Valid has {len(valid_names)} data from {image_path}")
    return {"train": train_names, "valid": valid_names}


def paired_data(data_path):
    # Load distorted images and split
    splits = split_data(f"{data_path}/trainA")

    # Check if every image exists in folder 'trainB'
    for key in ["train", "valid"]:
        paths = map(lambda x: f"{data_path}/trainB/{x}", splits[key])
        assert all(map(lambda x: os.path.isfile(x), paths))
    return splits


def unpaired_data(data_path):
    # Load distorted and enhanced images and split
    dt_splits = split_data(f"{data_path}/trainA", prefix="trainA/")
    eh_splits = split_data(f"{data_path}/trainB", prefix="trainB/")

    # Collect train/valid images from folders
    splits = dict()
    for key in ["train", "valid"]:
        splits[key] = dt_splits[key] + eh_splits[key]
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch FUnIE-GAN Data Splitting")
    parser.add_argument("-d", "--data", default="", type=str, metavar="PATH",
                        help="path to data (default: none)")
    parser.add_argument("-p", "--pair", action="store_true",
                        help="Set if data in pairs")

    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    if args.pair:
        print("Create paired dataset...")
        splits = paired_data(args.data)
    else:
        print("Create unpaired dataset...")
        splits = unpaired_data(args.data)

    # Write out
    output_path = f"{args.data}/splits.json"
    with open(output_path, "w") as f:
        json.dump(splits, f)
    print(f"Write splits to JSON {output_path}")
