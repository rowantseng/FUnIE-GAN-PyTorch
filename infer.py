import argparse
import os
import time

import numpy as np

import torch
from datasets import TestDataset, denorm
from models import FUnIEGeneratorV1, FUnIEGeneratorV2, FUnIEUpGenerator
from torchvision import transforms
from utils import AverageMeter, ProgressMeter


class Predictor(object):
    def __init__(self, model, test_loader, model_path, save_path, is_cuda):

        self.test_loader = test_loader
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.is_cuda = is_cuda
        self.print_freq = 20

        # Load model weights
        self.model = model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found!")
        self.load(model_path)
        if self.is_cuda:
            self.model.cuda()

    def predict(self):
        self.model.eval()

        batch_time = AverageMeter("Time", "3.3f")
        progress = ProgressMeter(len(self.test_loader), [
                                 batch_time], prefix="Test: ")

        with torch.no_grad():
            end = time.time()
            for batch_idx, (paths, images) in enumerate(self.test_loader):
                bs = images.size(0)
                if self.is_cuda:
                    images = images.cuda()
                fake_images = self.model(images)

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

    def load(self, model):
        device = "cuda:0" if self.is_cuda else "cpu"
        ckpt = torch.load(model, map_location=device)
        self.model.load_state_dict(ckpt["state_dict"])
        print(f"At epoch: {ckpt['epoch']} (loss={ckpt['best_loss']:.3f})")
        print(f">>> Load generator from {model}")


if __name__ == "__main__":

    # Set seed
    np.random.seed(77)
    torch.manual_seed(77)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.manual_seed(77)

    model_names = ["v1", "v2", "unpair"]
    model_archs = [FUnIEGeneratorV1, FUnIEGeneratorV2, FUnIEUpGenerator]
    model_mapper = {m: net for m, net in zip(model_names, model_archs)}

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
    parser.add_argument("-m", "--model", default="", type=str, metavar="PATH",
                        help="path to generator checkpoint (default: none)")
    parser.add_argument("--save-path", default="", type=str, metavar="PATH",
                        help="path to save results (default: none)")

    args = parser.parse_args()

    # Build data loader
    test_set = TestDataset(args.data, (256, 256))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Create predictor
    net = model_mapper[args.arch]()
    predictor = Predictor(net, test_loader, args.model,
                          args.save_path, is_cuda)
    predictor.predict()
