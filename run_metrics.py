import argparse
import glob
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity


def _compute_mse(label: np.ndarray, image: np.ndarray) -> float:
    """Compute MSE(Mean Squared Error)
    """
    return ((label - image) ** 2).mean()


def compute_psnr(label: np.ndarray, image: np.ndarray) -> float:
    """Compute PSNR
    """
    mse = _compute_mse(label, image)
    if mse == 0:
        return 100
    return 10 * np.log10(255 * 255 / mse)


def compute_mssim(label: np.ndarray, image: np.ndarray) -> float:
    """Compute MSSIM
    """
    return structural_similarity(label, image, data_range=255, win_size=5, multichannel=True, use_sample_covariance=True)


def get_results(label_paths: list, image_paths: list, metric: str = "psnr", im_size: tuple = (256, 256)):
    """Get mean and std results of PSNR or MSSIM
    """
    if metric == "psnr":
        metric_fn = compute_psnr
    elif metric == "ssim":
        metric_fn = compute_mssim
    else:
        raise ValueError(f"Metric '{metric}' is not supported!")
    results = [None] * len(label_paths)
    for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        image = Image.open(image_path).convert("RGB").resize(im_size)
        label = Image.open(label_path).convert("RGB").resize(im_size)
        image = np.array(image).astype(np.float64)
        label = np.array(label).astype(np.float64)
        results[i] = metric_fn(label, image)
    results = np.array(results)
    return results.mean(), results.std()


def _mu_alpha_trim(x: np.ndarray, alpha_low: float = 0.1, alpha_high: float = 0.1):
    """Calculate asymetric alpha-trimmed mu
    """
    assert x.ndim == 1, "Input must be flattened!"
    x = np.sort(x)
    num = len(x)
    low = math.ceil(alpha_low * num)
    high = math.floor(alpha_high * num)
    # Author: mu = sum(x) / (len(x) + 1)
    x = x[low+1:num-high]
    return x.mean()


def _compute_uicm(x: np.ndarray, c1: float = -0.0268, c2: float = 0.1586):
    """Underwater image colourfulness measure
    """
    R = x[..., 0].flatten()
    G = x[..., 1].flatten()
    B = x[..., 2].flatten()
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_RG = _mu_alpha_trim(RG)
    mu_YB = _mu_alpha_trim(YB)
    l = (mu_RG ** 2 + mu_YB ** 2) ** (1 / 2)
    r = (_compute_mse(RG, mu_RG) + _compute_mse(YB, mu_YB)) ** (1 / 2)
    return c1 * l + c2 * r


def _sobel(x: np.ndarray):
    """Perform sobel edge detector on x and y axis
    """
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = (dx ** 2 + dy ** 2) ** (1 / 2)
    mag /= mag.max()
    mag *= 255.0
    return mag


def _eme(x: np.ndarray, win_size: int):
    """Enhancement measure estimation
    """
    # Calculate width and height of the window
    w = int(x.shape[0] / win_size)
    h = int(x.shape[1] / win_size)
    # Cut out pixels outside the multiples of windows
    x = x[:win_size*w, :win_size*h]
    # Apply sliding window
    val = 0
    for i in range(w):
        for j in range(h):
            window = x[i*win_size:win_size*(i+1), j*win_size:win_size*(j+1)]
            max_val = window.max()
            min_val = window.min()
            if min_val == 0.0:
                val += 0
            else:
                val += math.log(max_val/min_val)
    return 2 * val / (w * h)


def _compute_uism(x: np.ndarray, win_size: int, lambda_r: float = 0.299, lambda_g: float = 0.587, lambda_b: float = 0.144):
    """Underwater image sharpness measure
    """
    R = x[..., 0]
    G = x[..., 1]
    B = x[..., 2]
    # Apply Sobel edge detector to each channel
    r_sobel = _sobel(R)
    g_sobel = _sobel(G)
    b_sobel = _sobel(B)
    # Get enhancement measure estimation for each edge enhanced image
    r_eme = _eme(R * r_sobel, win_size=win_size)
    g_eme = _eme(G * g_sobel, win_size=win_size)
    b_eme = _eme(B * b_sobel, win_size=win_size)
    return lambda_r * r_eme + lambda_g * g_eme + lambda_b * b_eme


def _compute_uiconm(x: np.ndarray, win_size: int):
    """Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/abstract/document/5609219
    """
    # Calculate width and height of the window
    w = int(x.shape[0] / win_size)
    h = int(x.shape[1] / win_size)
    # Cut out pixels outside the multiples of windows
    x = x[:win_size*w, :win_size*h]
    # Apply sliding window
    val = 0
    for i in range(w):
        for j in range(h):
            window = x[i*win_size:win_size*(i+1), j*win_size:win_size*(j+1), :]
            max_val = window.max()
            min_val = window.min()
            top = max_val - min_val
            bot = max_val + min_val
            if bot == 0 or top == 0:
                val += 0
            else:
                val += (top / bot) * math.log(top / bot)
    return (-1) * val / (w * h)


def compute_uiqm(image_paths: list, im_size: tuple = (256, 256), win_size: int = 10, c1: float = 0.0282, c2: float = 0.2953, c3: float = 3.5753):
    """Compute UIQM
    """
    uiqms = [None] * len(image_paths)
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB").resize(im_size)
        image = np.array(image)
        image = image.astype(np.float64)
        uicm = _compute_uicm(image)
        uism = _compute_uism(image, win_size=win_size)
        uiconm = _compute_uiconm(image, win_size=win_size)
        uiqms[i] = c1 * uicm + c2 * uism + c3 * uiconm
    uiqms = np.array(uiqms)
    return uiqms.mean(), uiqms.std()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch FUnIE-GAN Metric Runner")
    parser.add_argument("--image-data", default="", type=str, metavar="PATH",
                        help="path to images (default: none)")
    parser.add_argument("--label-data", default="", type=str, metavar="PATH",
                        help="path to ground truths (default: none)")

    args = parser.parse_args()

    # Define metrics
    metrics = ["psnr", "ssim"]

    # Load ground truths
    print(f"Load ground truths from {args.label_data}")
    labels = sorted(glob.glob(f"{args.label_data}/*.*"))
    blabels = [os.path.splitext(os.path.basename(i))[0] for i in labels]

    # Load images
    print(f"Load images from {args.image_data}")
    images = sorted(glob.glob(f"{args.image_data}/*.*"))
    bimages = [os.path.splitext(os.path.basename(i))[0] for i in images]
    assert bimages == blabels

    # Compute metric over test samples
    for metric in metrics:
        mu, std = get_results(labels, images, metric=metric)
        print(f"[{metric.upper()}] Mean={mu:.3f}, Std={std:.3f}")

    mu, std = compute_uiqm(images)
    print(f"[UIQM] Mean={mu:.3f}, Std={std:.3f}")
