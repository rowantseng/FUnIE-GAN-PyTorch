import argparse
import os
import time
from shutil import copyfile

import numpy as np

import torch
import torch.optim as optim
from datasets import PairDataset, denorm
from models import FUnIEDiscriminator, FUnIEGeneratorV1, TotalGenLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from utils import AverageMeter, ProgressMeter


class Trainer(object):
    def __init__(self, train_loader, valid_loader, lr, epochs, gen_resume, dis_resume, save_path, is_cuda):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.start_epoch = 0
        self.epochs = epochs
        self.save_path = save_path
        os.makedirs(f"{self.save_path}/viz", exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_path)

        self.is_cuda = is_cuda
        self.print_freq = 20
        self.best_gen_loss = 1e6

        self.gen = FUnIEGeneratorV1()
        self.dis = FUnIEDiscriminator()
        if gen_resume and dis_resume:
            self.load(gen_resume, dis_resume)

        if self.is_cuda:
            self.gen.cuda()
            self.dis.cuda()

        self.dis_criterion = torch.nn.MSELoss()
        self.gen_criterion = TotalGenLoss(self.is_cuda)

        self.dis_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.dis.parameters()), lr)
        self.gen_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.gen.parameters()), lr)

    def train(self):
        for e in range(self.start_epoch, self.epochs):
            self.epoch = e
            train_gen_loss, train_dis_loss = self.train_epoch()
            valid_gen_loss, valid_dis_loss = self.validate()

            # Save models
            is_best = valid_gen_loss < self.best_gen_loss
            self.best_gen_loss = min(self.best_gen_loss, valid_gen_loss)
            self.save(is_best)

        self.writer.close()

    def train_epoch(self):
        self.gen.train()
        self.dis.train()

        batch_time = AverageMeter("Time", "3.3f")
        gen_losses = AverageMeter("Generator Loss")
        dis_losses = AverageMeter("Discriminator Loss")
        progress = ProgressMeter(len(self.train_loader), [
                                 batch_time, gen_losses, dis_losses], prefix="Train: ")

        end = time.time()
        for batch_idx, (dstd_images, ehcd_images) in enumerate(self.train_loader):
            bs = dstd_images.size(0)
            valid = torch.ones((bs, 16, 16))
            fake = torch.zeros((bs, 16, 16))
            if self.is_cuda:
                dstd_images = dstd_images.cuda()
                ehcd_images = ehcd_images.cuda()
                valid = valid.cuda()
                fake = fake.cuda()

            # -----
            # Train the discriminator
            # -----
            fake_images = self.gen(dstd_images)

            real_outputs = self.dis(ehcd_images, dstd_images)
            real_d_loss = self.dis_criterion(real_outputs, valid)

            fake_outputs = self.dis(fake_images, dstd_images)
            fake_d_loss = self.dis_criterion(fake_outputs, fake)

            d_loss = (real_d_loss + fake_d_loss) / 2

            self.dis_optimizer.zero_grad()
            d_loss.backward()
            self.dis_optimizer.step()

            # -----
            # Train the generator
            # -----
            fake_images = self.gen(dstd_images)
            g_loss = self.dis_criterion(fake_images, ehcd_images)

            # Total loss
            total_gen_loss = 0.2 * g_loss + 0.8 * \
                self.gen_criterion(ehcd_images, fake_images)

            self.gen_optimizer.zero_grad()
            total_gen_loss.backward()
            self.gen_optimizer.step()

            # Update
            dis_losses.update(d_loss.item(), bs)
            gen_losses.update(total_gen_loss.item(), bs)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.print_freq == 0:
                progress.display(batch_idx)

        # Write stats to tensorboard
        self.writer.add_scalar("Generator Loss/Train",
                               gen_losses.avg, self.epoch)
        self.writer.add_scalar("Discriminator Loss/Train",
                               dis_losses.avg, self.epoch)

        return gen_losses.avg, dis_losses.avg

    def validate(self):
        self.gen.eval()
        self.dis.eval()

        batch_time = AverageMeter("Time", "3.3f")
        gen_losses = AverageMeter("Generator Loss")
        dis_losses = AverageMeter("Discriminator Loss")
        progress = ProgressMeter(len(self.valid_loader), [
                                 batch_time, gen_losses, dis_losses], prefix="Valid: ")

        with torch.no_grad():
            end = time.time()
            for batch_idx, (dstd_images, ehcd_images) in enumerate(self.valid_loader):
                bs = dstd_images.size(0)
                valid = torch.ones((bs, 16, 16))
                fake = torch.zeros((bs, 16, 16))
                if self.is_cuda:
                    dstd_images = dstd_images.cuda()
                    ehcd_images = ehcd_images.cuda()
                    valid = valid.cuda()
                    fake = fake.cuda()

                # -----
                # Train the discriminator
                # -----
                fake_images = self.gen(dstd_images)

                real_outputs = self.dis(ehcd_images, dstd_images)
                real_d_loss = self.dis_criterion(real_outputs, valid)

                fake_outputs = self.dis(fake_images, dstd_images)
                fake_d_loss = self.dis_criterion(fake_outputs, fake)

                d_loss = (real_d_loss + fake_d_loss) / 2

                # -----
                # Train the generator
                # -----
                fake_images = self.gen(dstd_images)
                g_loss = self.dis_criterion(fake_images, ehcd_images)

                # Total loss
                total_gen_loss = 0.2 * g_loss + 0.8 * \
                    self.gen_criterion(ehcd_images, fake_images)

                # Update
                dis_losses.update(d_loss.item(), bs)
                gen_losses.update(total_gen_loss.item(), bs)

                batch_time.update(time.time() - end)
                end = time.time()

                # Vis
                if batch_idx == 0:
                    fake_grid = denorm(make_grid(fake_images.data)).div_(255.)
                    ehcd_grid = denorm(make_grid(ehcd_images.data)).div_(255.)
                    dstd_grid = denorm(make_grid(dstd_images.data)).div_(255.)
                    save_image(
                        fake_grid, f"{self.save_path}/viz/fake_{self.epoch}.png")
                    save_image(
                        ehcd_grid, f"{self.save_path}/viz/ehcd_{self.epoch}.png")
                    save_image(
                        dstd_grid, f"{self.save_path}/viz/dstd_{self.epoch}.png")
                    self.writer.add_image("Viz/Fake", fake_grid, self.epoch)
                    self.writer.add_image("Viz/Enhance", ehcd_grid, self.epoch)
                    self.writer.add_image("Viz/Distort", dstd_grid, self.epoch)

                if batch_idx % self.print_freq == 0:
                    progress.display(batch_idx)

        # Write stats to tensorboard
        self.writer.add_scalar("Generator Loss/Validation",
                               gen_losses.avg, self.epoch)
        self.writer.add_scalar("Discriminator Loss/Validation",
                               dis_losses.avg, self.epoch)

        return gen_losses.avg, dis_losses.avg

    def predict(self, image_tensors):
        self.gen.eval()
        return self.gen(image_tensors)

    def save(self, is_best):
        gen_path = f"{self.save_path}/{self.epoch}_gen.pth.tar"
        torch.save({"state_dict": self.gen.state_dict(),
                    "best_loss": self.best_gen_loss,
                    "epoch": self.epoch}, gen_path)
        print(f">>> Save generator to {gen_path}")

        dis_path = f"{self.save_path}/{self.epoch}_dis.pth.tar"
        torch.save({"state_dict": self.dis.state_dict(),
                    "epoch": self.epoch}, dis_path)
        print(f">>> Save discriminator to {dis_path}")

        if is_best:
            copyfile(gen_path, f"{self.save_path}/best_gen.pth.tar")
            copyfile(dis_path, f"{self.save_path}/best_dis.pth.tar")

    def load(self, gen_resume, dis_resume):
        if self.is_cuda:
            gen_ckpt = torch.load(gen_resume)
            dis_ckpt = torch.load(dis_resume)
        else:
            gen_ckpt = torch.load(gen_resume, map_location="cpu")
            dis_ckpt = torch.load(dis_resume, map_location="cpu")

        assert gen_ckpt["epoch"] == dis_ckpt["epoch"]
        self.gen.load_state_dict(gen_ckpt["state_dict"])
        self.dis.load_state_dict(dis_ckpt["state_dict"])
        self.best_gen_loss = gen_ckpt["best_loss"]
        self.start_epoch = gen_ckpt["epoch"] + 1

        print(
            f">>> Load generator from {gen_resume} at epoch={gen_ckpt['epoch']}")
        print(
            f">>> Load discriminator from {dis_resume} at epoch={gen_ckpt['epoch']}")


if __name__ == "__main__":

    # Set seed
    np.random.seed(77)
    torch.manual_seed(77)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.manual_seed(77)

    model_names = ["v1", "v2"]

    parser = argparse.ArgumentParser(description="PyTorch FUnIE-GAN Training")
    parser.add_argument("-d", "--data", default="", type=str, metavar="PATH",
                        help="path to data (default: none)")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="v1",
                        choices=model_names,
                        help="model architecture: " +
                        " | ".join(model_names) +
                        " (default: v1)")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers (default: 4)")
    parser.add_argument("--epochs", default=90, type=int, metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("-b", "--batch-size", default=256, type=int,
                        metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                        "batch size of all GPUs on the current node when "
                        "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float,
                        metavar="LR", help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                        help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float,
                        metavar="W", help="weight decay (default: 1e-4)")
    parser.add_argument("-p", "--print-freq", default=10, type=int,
                        metavar="N", help="print frequency (default: 10)")
    parser.add_argument("--gen-resume", default="", type=str, metavar="PATH",
                        help="path to latest generator checkpoint (default: none)")
    parser.add_argument("--dis-resume", default="", type=str, metavar="PATH",
                        help="path to latest discriminator checkpoint (default: none)")
    parser.add_argument("--save-path", default="", type=str, metavar="PATH",
                        help="path to save results (default: none)")
    parser.add_argument("-e", "--evaluate", action="store_true",
                        help="evaluate model on validation set")

    args = parser.parse_args()

    # Build data loaders
    train_set = PairDataset(args.data, (256, 256), "train")
    valid_set = PairDataset(args.data, (256, 256), "valid")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Create trainer
    trainer = Trainer(train_loader, valid_loader, args.lr, args.epochs,
                      args.gen_resume, args.dis_resume, args.save_path, is_cuda)

    # Train or evaluate
    if args.evaluate:
        trainer.validate()
    else:
        trainer.train()
