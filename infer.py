import os
from datasets import TestDataset

import argparse
import os
import time
from shutil import copyfile

import numpy as np

import torch
import torch.optim as optim
from datasets import TestDataset, denorm
from models import FUnIEDiscriminator, FUnIEGeneratorV1, TotalGenLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from utils import AverageMeter, ProgressMeter
from torchvision import transforms

class Predictor(object):
    def __init__(self, test_loader, gen_model, save_path, is_cuda):

        self.test_loader = test_loader
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.is_cuda = is_cuda
        self.print_freq = 20

        self.gen = FUnIEGeneratorV1()
        if gen_model:
            self.load(gen_model)

        if self.is_cuda:
            self.gen.cuda()

    def predict(self):
        self.gen.eval()

        batch_time = AverageMeter("Time", "3.3f")
        progress = ProgressMeter(len(self.test_loader), [batch_time], prefix="Test: ")

        with torch.no_grad():
            end = time.time()
            for batch_idx, (paths, images) in enumerate(self.test_loader):
                bs = images.size(0)
                if self.is_cuda:
                    images = images.cuda()
                fake_images = self.gen(images)
                
                fake_images = denorm(fake_images.data)
                fake_images = torch.clamp(fake_images, min=0., max=255.)
                fake_images = fake_images.type(torch.uint8)

                for idx in range(bs):
                    name = os.path.splitext(os.path.basename(paths[idx]))[0]
                    fake_image = fake_images[idx]
                    fake_image = transforms.ToPILImage()(fake_image).convert("RGB")
                    fake_image.save(f"{self.save_path}/{name}.png")

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % self.print_freq == 0:
                    progress.display(batch_idx)
        return

    def load(self, gen_model):
        if self.is_cuda:
            gen_ckpt = torch.load(gen_model)
        else:
            gen_ckpt = torch.load(gen_model, map_location="cpu")

        self.gen.load_state_dict(gen_ckpt["state_dict"])
        self.best_gen_loss = gen_ckpt["best_loss"]
        self.start_epoch = gen_ckpt["epoch"] + 1

        print(
            f">>> Load generator from {gen_model} at epoch={gen_ckpt['epoch']}")


if __name__ == "__main__":

    # Set seed
    np.random.seed(77)
    torch.manual_seed(77)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.manual_seed(77)

    model_names = ["v1", "v2"]

    parser = argparse.ArgumentParser(description="PyTorch FUnIE-GAN Inference")
    parser.add_argument("-d", "--data", default="", type=str, metavar="PATH",
                        help="path to data (default: none)")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="v1",
                        choices=model_names,
                        help="model architecture: " +
                        " | ".join(model_names) +
                        " (default: v1)")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers (default: 4)")
    parser.add_argument("-b", "--batch-size", default=256, type=int,
                        metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                        "batch size of all GPUs on the current node when "
                        "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("-p", "--print-freq", default=10, type=int,
                        metavar="N", help="print frequency (default: 10)")
    parser.add_argument("--gen-model", default="", type=str, metavar="PATH",
                        help="path to latest generator checkpoint (default: none)")
    parser.add_argument("--save-path", default="", type=str, metavar="PATH",
                        help="path to save results (default: none)")

    args = parser.parse_args()

    # Build data loader
    test_set = TestDataset(args.data, (256, 256))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Create predictor
    predictor = Predictor(test_loader, args.gen_model, args.save_path, is_cuda)
    predictor.predict()