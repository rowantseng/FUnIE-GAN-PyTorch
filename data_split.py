import argparse
import glob
import json
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch FUnIE-GAN Data Splitting")
    parser.add_argument("-d", "--data", default="", type=str, metavar="PATH",
                        help="path to data (default: none)")

    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)

    # Load images and make sure distorted and enhanced images have same order
    dt_images = list(glob.glob(f"{args.data}/trainA/*.*"))
    eh_images = list(glob.glob(f"{args.data}/trainB/*.*"))
    base_names = list(map(os.path.basename, dt_images))
    assert base_names == list(map(os.path.basename, eh_images))

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
    print(f"Train has {len(train_names)} data")
    print(f"Valid has {len(valid_names)} data")

    # Write out
    output_path = f"{args.data}/splits.json"
    json.dump({"train": train_names, "valid": valid_names}, open(output_path, "w"))
    print(f"Write splits to JSON {output_path}")
