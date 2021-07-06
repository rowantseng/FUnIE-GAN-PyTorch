import argparse
import os
import time
from shutil import copyfile

import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from datasets import UnpairDataset, denorm
from models import FUnIEUpGenerator, FUnIEUpDiscriminator
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter, ProgressMeter


class Trainer(object):
    def __init__(self, train_loader, valid_loader, lr, epochs, gen_dstd2ehcd_resume, gen_ehcd2dstd_resume, dis_dstd_resume, dis_ehcd_resume, save_path, is_cuda):

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

        self.gen_dstd2ehcd = FUnIEUpGenerator()
        self.gen_ehcd2dstd = FUnIEUpGenerator()
        self.dis_dstd = FUnIEUpDiscriminator()
        self.dis_ehcd = FUnIEUpDiscriminator()

        if gen_dstd2ehcd_resume and gen_ehcd2dstd_resume and dis_dstd_resume and dis_ehcd_resume:
            self.load(gen_dstd2ehcd_resume, gen_ehcd2dstd_resume,
                      dis_dstd_resume, dis_ehcd_resume)

        if self.is_cuda:
            self.gen_dstd2ehcd.cuda()
            self.gen_ehcd2dstd.cuda()
            self.dis_dstd.cuda()
            self.dis_ehcd.cuda()

        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()

        dis_params = list(self.dis_dstd.parameters()) + \
            list(self.dis_ehcd.parameters())
        gen_params = list(self.gen_dstd2ehcd.parameters()) + \
            list(self.gen_ehcd2dstd.parameters())
        self.dis_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, dis_params), lr)
        self.gen_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, gen_params), lr)

    def train(self):
        for e in range(self.start_epoch, self.epochs):
            self.epoch = e
            _, _ = self.train_epoch()
            valid_gen_loss, _ = self.validate()

            # Save models
            self.save(valid_gen_loss)

        self.writer.close()

    def train_epoch(self):
        self.gen_dstd2ehcd.train()
        self.gen_ehcd2dstd.train()
        self.dis_dstd.train()
        self.dis_ehcd.train()

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

            # Train the discriminator using real samples
            valid_dstd = self.dis_dstd(dstd_images)
            valid_ehcd = self.dis_ehcd(ehcd_images)
            d_loss_real = self.mse(valid, valid_dstd) + self.mse(valid, valid_ehcd)
            self.dis_optimizer.zero_grad()
            d_loss_real.backward()
            self.dis_optimizer.step()

            # Train the discriminator using fake samples
            valid_dstd = self.dis_dstd(self.gen_ehcd2dstd(ehcd_images))
            valid_ehcd = self.dis_ehcd(self.gen_dstd2ehcd(dstd_images))
            d_loss_fake = self.mse(fake, valid_dstd) + self.mse(fake, valid_ehcd)
            self.dis_optimizer.zero_grad()
            d_loss_fake.backward()
            self.dis_optimizer.step()

            # Train the generator using dstd->ehcd->dstd cycle
            fake_ehcd = self.gen_dstd2ehcd(dstd_images)
            valid_ehcd = self.dis_ehcd(fake_ehcd)
            recn_dstd = self.gen_ehcd2dstd(fake_ehcd)
            g_loss_dstd = self.mae(valid, valid_ehcd) + \
                10 * self.mae(dstd_images, recn_dstd)
            self.gen_optimizer.zero_grad()
            g_loss_dstd.backward()
            self.gen_optimizer.step()

            # Train the generator using ehcd->dstd->ehcd cycle
            fake_dstd = self.gen_ehcd2dstd(ehcd_images)
            valid_dstd = self.dis_dstd(fake_dstd)
            recn_ehcd = self.gen_dstd2ehcd(fake_dstd)
            g_loss_ehcd = self.mae(valid, valid_dstd) + \
                10 * self.mae(ehcd_images, recn_ehcd)
            self.gen_optimizer.zero_grad()
            g_loss_ehcd.backward()
            self.gen_optimizer.step()

            # Total loss
            d_loss = d_loss_real + d_loss_fake
            g_loss = g_loss_dstd + g_loss_ehcd

            # Update
            dis_losses.update(d_loss.item(), bs)
            gen_losses.update(g_loss.item(), bs)

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
        self.gen_dstd2ehcd.eval()
        self.gen_ehcd2dstd.eval()
        self.dis_dstd.eval()
        self.dis_ehcd.eval()

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

                # Train the discriminator using real samples
                valid_dstd = self.dis_dstd(dstd_images)
                valid_ehcd = self.dis_ehcd(ehcd_images)
                d_loss_real = self.mse(valid, valid_dstd) + \
                    self.mse(valid, valid_ehcd)

                # Train the discriminator using fake samples
                fake_dstd = self.gen_ehcd2dstd(ehcd_images)
                fake_ehcd = self.gen_dstd2ehcd(dstd_images)
                valid_dstd = self.dis_dstd(fake_dstd)
                valid_ehcd = self.dis_ehcd(fake_ehcd)
                d_loss_fake = self.mse(fake, valid_dstd) + \
                    self.mse(fake, valid_ehcd)

                # Train the generator using dstd->ehcd->dstd cycle
                valid_ehcd = self.dis_ehcd(fake_ehcd)
                recn_dstd = self.gen_ehcd2dstd(fake_ehcd)
                g_loss_dstd = self.mse(valid, valid_ehcd) + \
                    self.mse(dstd_images, recn_dstd)

                # Train the generator using ehcd->dstd->ehcd cycle
                valid_dstd = self.dis_dstd(fake_dstd)
                recn_ehcd = self.gen_dstd2ehcd(fake_dstd)
                g_loss_ehcd = self.mse(valid, valid_dstd) + \
                    self.mse(ehcd_images, recn_ehcd)

                # Total loss
                d_loss = d_loss_real + d_loss_fake
                g_loss = g_loss_dstd + g_loss_ehcd

                # Update
                dis_losses.update(d_loss.item(), bs)
                gen_losses.update(g_loss.item(), bs)

                batch_time.update(time.time() - end)
                end = time.time()

                # Vis
                if batch_idx == 0:
                    fake_ehcd_grid = denorm(
                        make_grid(fake_ehcd.data)).div_(255.)
                    fake_dstd_grid = denorm(
                        make_grid(fake_dstd.data)).div_(255.)
                    recn_ehcd_grid = denorm(
                        make_grid(recn_ehcd.data)).div_(255.)
                    recn_dstd_grid = denorm(
                        make_grid(recn_dstd.data)).div_(255.)
                    save_image(
                        fake_ehcd_grid, f"{self.save_path}/viz/fake_ehcd_{self.epoch}.png")
                    save_image(
                        fake_dstd_grid, f"{self.save_path}/viz/fake_dstd_{self.epoch}.png")
                    save_image(
                        recn_ehcd_grid, f"{self.save_path}/viz/recn_ehcd_{self.epoch}.png")
                    save_image(
                        recn_dstd_grid, f"{self.save_path}/viz/recn_dstd_{self.epoch}.png")
                    self.writer.add_image(
                        "Viz/Fake Distort", fake_ehcd_grid, self.epoch)
                    self.writer.add_image(
                        "Viz/Fake Enhance", fake_dstd_grid, self.epoch)
                    self.writer.add_image(
                        "Viz/Recn Distort", recn_ehcd_grid, self.epoch)
                    self.writer.add_image(
                        "Viz/Recn Enhance", recn_dstd_grid, self.epoch)

                if batch_idx % self.print_freq == 0:
                    progress.display(batch_idx)

        # Write stats to tensorboard
        self.writer.add_scalar("Generator Loss/Validation",
                               gen_losses.avg, self.epoch)
        self.writer.add_scalar("Discriminator Loss/Validation",
                               dis_losses.avg, self.epoch)

        return gen_losses.avg, dis_losses.avg

    def save_model(self, model_type, model, model_content, is_best):
        model_path = f"{self.save_path}/{self.epoch}_{model_type}.pth.tar"
        model_content["state_dict"] = model.state_dict()
        torch.save(model_content, model_path)
        print(f">>> Save '{model_type}' model to {model_path}")
        if is_best:
            best_path = f"{self.save_path}/best_{model_type}.pth.tar"
            copyfile(model_path, best_path)

    def load_model(self, model_type, model, model_path, device):
        ckpt = torch.load(model_path, map_location=device)
        epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["state_dict"])
        print(
            f">>> Load '{model_type}' model at epoch {epoch} from {model_path}")
        return epoch, ckpt["best_loss"]

    def save(self, loss):
        # Check if the current model is the best
        is_best = loss < self.best_gen_loss
        self.best_gen_loss = min(self.best_gen_loss, loss)

        # Prepare model info to be saved
        model_content = {"best_loss": loss, "epoch": self.epoch}

        # Save generator and discriminator
        self.save_model("gen_dstd2ehcd", self.gen_dstd2ehcd, model_content, is_best)
        self.save_model("gen_ehcd2dstd", self.gen_ehcd2dstd, model_content, is_best)
        self.save_model("dis_dstd", self.dis_dstd, model_content, is_best)
        self.save_model("dis_ehcd", self.dis_ehcd, model_content, is_best)

    def load(self, gen_dstd2ehcd_resume, gen_ehcd2dstd_resume, dis_dstd_resume, dis_ehcd_resume):
        device = "cuda:0" if self.is_cuda else "cpu"
        gen_dstd2ehcd_epoch, best_loss = self.load_model(
            "gen_dstd2ehcd", self.gen_dstd2ehcd, gen_dstd2ehcd_resume, device)
        gen_ehcd2dstd_epoch, _ = self.load_model(
            "gen_ehcd2dstd", self.gen_ehcd2dstd, gen_ehcd2dstd_resume, device)
        dis_dstd_epoch, _ = self.load_model(
            "dis_dstd", self.dis_dstd, dis_dstd_resume, device)
        dis_dst_epoch, _ = self.load_model(
            "dis_ehcd", self.dis_ehcd, dis_ehcd_resume, device)

        assert gen_dstd2ehcd_epoch == gen_ehcd2dstd_epoch == dis_dstd_epoch == dis_dst_epoch
        self.start_epoch = gen_dstd2ehcd_epoch + 1


if __name__ == "__main__":

    # Set seed
    np.random.seed(77)
    torch.manual_seed(77)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.manual_seed(77)

    parser = argparse.ArgumentParser(description="PyTorch FUnIE-GAN Training")
    parser.add_argument("-d", "--data", default="", type=str, metavar="PATH",
                        help="path to data (default: none)")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers (default: 4)")
    parser.add_argument("--epochs", default=90, type=int, metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("-b", "--batch-size", default=256, type=int,
                        metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                        "batch size of all GPUs on the current node when "
                        "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float,
                        metavar="LR", help="initial learning rate")
    parser.add_argument("--gen-dstd2ehcd-resume", default="", type=str, metavar="PATH",
                        help="path to latest dstd2ehcd generator checkpoint (default: none)")
    parser.add_argument("--gen-ehcd2dstd-resume", default="", type=str, metavar="PATH",
                        help="path to latest ehcd2dstd generator checkpoint (default: none)")
    parser.add_argument("--dis-dstd-resume", default="", type=str, metavar="PATH",
                        help="path to latest dstd discriminator checkpoint (default: none)")
    parser.add_argument("--dis-ehcd-resume", default="", type=str, metavar="PATH",
                        help="path to latest ehcd discriminator checkpoint (default: none)")
    parser.add_argument("--save-path", default="", type=str, metavar="PATH",
                        help="path to save results (default: none)")

    args = parser.parse_args()

    # Build data loaders
    train_set = UnpairDataset(args.data, (256, 256), "train")
    valid_set = UnpairDataset(args.data, (256, 256), "valid")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Create trainer
    trainer = Trainer(train_loader, valid_loader, args.lr, args.epochs, args.gen_dstd2ehcd_resume,
                      args.gen_ehcd2dstd_resume, args.dis_dstd_resume, args.dis_ehcd_resume, args.save_path, is_cuda)
    trainer.train()
