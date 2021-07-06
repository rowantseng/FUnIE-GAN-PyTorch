"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
# python libs
import numpy as np
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from ntpath import basename
# local libs
from uqim_utils import getUIQM
import argparse


def measure_UIQMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PyTorch FUnIE-GAN UIQM Metric Runner")
    parser.add_argument("--data", default="", type=str, metavar="PATH",
                        help="path to images (default: none)")

    args = parser.parse_args()

    uqims = measure_UIQMs(args.data)
    print(f"[UIQM] Mean={np.mean(uqims):.3f}, Std={np.std(uqims):.3f}")
