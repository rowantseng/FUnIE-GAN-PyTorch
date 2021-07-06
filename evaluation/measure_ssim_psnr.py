"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
# python libs
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
# local libs
from imqual_utils import getSSIM, getPSNR
import argparse


# compares avg ssim and psnr
def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images 
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims, psnrs = [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        if (gtr_f == gen_f):
            # assumes same filenames
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)
            # get ssim on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            # get psnt on L channel (SOTA norm)
            r_im = r_im.convert("L")
            g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PyTorch FUnIE-GAN SSIM and PSNR Metric Runner")
    parser.add_argument("--image-data", default="", type=str, metavar="PATH",
                        help="path to images (default: none)")
    parser.add_argument("--label-data", default="", type=str, metavar="PATH",
                        help="path to ground truths (default: none)")

    args = parser.parse_args()

    # Compute SSIM and PSNR
    SSIM_measures, PSNR_measures = SSIMs_PSNRs(
        args.label_data, args.image_data)
    print("SSIM on {0} samples".format(len(SSIM_measures)))
    print(f"Mean={np.mean(SSIM_measures):.3f}, Std={np.std(SSIM_measures):.3f}")
    print("PSNR on {0} samples".format(len(PSNR_measures)))
    print(f"Mean={np.mean(PSNR_measures):.3f}, Std={np.std(PSNR_measures):.3f}")
