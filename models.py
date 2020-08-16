import torch.nn as nn
import torch
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def forward(self, targets, outputs):
        """
           Calculating perceptual distance
           Thanks to github.com/wandb/superres
        """
        targets = (targets + 1.0) * 127.5  # [-1,1] -> [0, 255]
        outputs = (outputs + 1.0) * 127.5  # [-1,1] -> [0, 255]
        rmean = (targets[:, :, :, 0] + outputs[:, :, :, 0]) / 2
        r = targets[:, :, :, 0] - outputs[:, :, :, 0]
        g = targets[:, :, :, 1] - outputs[:, :, :, 1]
        b = targets[:, :, :, 2] - outputs[:, :, :, 2]
        return torch.sqrt((((512 + rmean) * r * r) / 256) + 4 * g * g + (((767 - rmean) * b * b) / 256))


class TotalGenLoss(nn.Module):
    def __init__(self, is_cuda, gen_version="v1"):
        super(TotalGenLoss, self).__init__()
        self.vgg = VGGContent()
        if is_cuda:
            self.vgg = self.vgg.cuda()
        self.compute_perceptual_loss = PerceptualLoss()
        self.gen_version = gen_version

    def forward(self, org_image, gen_image):
        vgg_org_image = self.vgg(org_image)
        vgg_gen_image = self.vgg(gen_image)
        
        bs = org_image.size(0)
        content_loss = ((vgg_org_image - vgg_gen_image) ** 2).mean(1)
        mae_gen_loss = (torch.abs(org_image - gen_image)).view(bs, -1).mean(1)
        perceptual_loss = self.compute_perceptual_loss(org_image, gen_image).view(bs, -1).mean(1)

        if self.gen_version == "v1":
            return (0.7 * mae_gen_loss + 0.3 * content_loss).mean()
        else:
            return (0.6 * mae_gen_loss + 0.3 * content_loss + 0.1 * perceptual_loss).mean()


class VGGContent(nn.Module):
    def __init__(self):
        super(VGGContent, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True).features

    def forward(self, x):
        bs = x.size(0)
        return self.vgg(x).view(bs, -1)


class FUnIEGeneratorV1(nn.Module):
    def __init__(self, num_features=32):
        super(FUnIEGeneratorV1, self).__init__()
        self.conv1 = self._conv_block(3, num_features, 5, 2, False)
        self.conv2 = self._conv_block(num_features, num_features*4, 4, 1)
        self.conv3 = self._conv_block(num_features*4, num_features*8, 4, 1)
        self.conv4 = self._conv_block(num_features*8, num_features*8, 3, 1)
        self.conv5 = self._conv_block(num_features*8, num_features*8, 3, 1)

        self.deconv1 = self._deconv_block(num_features*8, num_features*8)
        self.deconv2 = self._deconv_block(num_features*16, num_features*8)
        self.deconv3 = self._deconv_block(num_features*16, num_features*4)
        self.deconv4 = self._deconv_block(num_features*8, num_features*1)
        self.deconv5 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        # kernel_size from 4 to 3
        self.final = nn.Conv2d(num_features*2, 3, 3, 1, 1)
        self.act = nn.Tanh()

    def _conv_block(self, in_channels, out_channels, kernel_size, pad, use_bn=True):
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, 2, pad))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.8))
        return nn.Sequential(*layers)

    def _deconv_block(self, in_channels, out_channels, use_bn=True):
        layers = []
        layers.append(nn.Upsample(scale_factor=2,
                                  mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.8))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Downsample
        d1 = self.conv1(x)   # (B, 32, 128, 128)
        d2 = self.conv2(d1)  # (B, 128, 64, 64)
        d3 = self.conv3(d2)  # (B, 256, 32, 32)
        d4 = self.conv4(d3)  # (B, 256, 16, 16)
        d5 = self.conv5(d4)  # (B, 256, 8, 8)

        # Upsample
        u1 = torch.cat([self.deconv1(d5), d4], dim=1)  # (B, 512, 16, 16)
        u2 = torch.cat([self.deconv2(u1), d3], dim=1)  # (B, 512, 32, 32)
        u3 = torch.cat([self.deconv3(u2), d2], dim=1)  # (B, 256, 64, 64)
        u4 = torch.cat([self.deconv4(u3), d1], dim=1)  # (B, 64, 128, 128)
        u5 = self.deconv5(u4)                          # (B, 64, 256, 256)

        return self.act(self.final(u5))


class FUnIEGeneratorV2(nn.Module):
    def __init__(self, num_features=32):
        super(FUnIEGeneratorV2, self).__init__()
        self.conv1 = self._conv_block(3, num_features, 5, 2, False)
        self.conv2 = self._conv_block(
            num_features, num_features*2, 3, 1)    # kernel_size from 4 to 3
        self.conv3 = self._conv_block(
            num_features*2, num_features*2, 3, 1)  # kernel_size from 4 to 3
        self.conv4 = self._conv_block(num_features*2, num_features*4, 3, 1)
        self.conv5 = self._conv_block(num_features*4, num_features*4, 3, 1)
        self.conv6 = self._conv_block(num_features*4, num_features*8, 3, 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.deconv1 = self._deconv_block(num_features*8, num_features*8)
        self.deconv2 = self._deconv_block(num_features*12, num_features*8)
        self.deconv3 = self._deconv_block(num_features*10, num_features*4)

        self.out1 = self._conv_block(num_features*5, num_features*4, 3, 1)
        self.out2 = self._conv_block(num_features*4, num_features*8, 3, 1)
        # kernel_size from 4 to 3
        self.final = nn.Conv2d(num_features*8, 3, 3, 1, 1)
        self.act = nn.Tanh()

    def _conv_block(self, in_channels, out_channels, kernel_size, pad, use_bn=True):
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, pad))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.75))
        return nn.Sequential(*layers)

    def _deconv_block(self, in_channels, out_channels, use_bn=True):
        layers = []
        layers.append(nn.Upsample(scale_factor=2,
                                  mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.8))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Downsample
        d1 = self.conv1(x)
        d1a = self.pool(d1)   # (B, 32, 128, 128)
        d2 = self.conv2(d1a)
        d3 = self.conv3(d2)
        d3a = self.pool(d3)   # (B, 64, 64, 64)
        d4 = self.conv4(d3a)
        d5 = self.conv5(d4)
        d5a = self.pool(d5)   # (B, 128, 32, 32)
        d6 = self.conv6(d5a)  # (B, 256, 32, 32)

        # Upsample
        u1 = torch.cat([self.deconv1(d6), d5], dim=1)  # (B, 384, 64, 64)
        u2 = torch.cat([self.deconv2(u1), d3], dim=1)  # (B, 320, 128, 128)
        u3 = torch.cat([self.deconv3(u2), d1], dim=1)  # (B, 160, 256, 256)

        return self.act(self.final(self.out2(self.out1(u3))))


class FUnIEDiscriminator(nn.Module):
    def __init__(self, num_features=32):
        super(FUnIEDiscriminator, self).__init__()

        # Build discriminator blocks
        self.block1 = self._block(3*2, num_features, False)
        self.block2 = self._block(num_features, num_features*2)
        self.block3 = self._block(num_features*2, num_features*4)
        self.block4 = self._block(num_features*4, num_features*8)

        # Validility block
        # kernel_size from 4 to 3
        self.validility = nn.Conv2d(num_features*8, 1, 3, 1, 1)

    def _block(self, in_channels, out_channels, use_bn=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 2, 1))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.8))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # (B, 6, 256, 256)
        x = self.block1(x)              # (B, 32, 128, 128)
        x = self.block2(x)              # (B, 64, 64, 64)
        x = self.block3(x)              # (B, 128, 32, 32)
        x = self.block4(x)              # (B, 256, 16, 16)
        valid = self.validility(x)      # (B, 1, 16, 16)
        return valid.squeeze(1)


if __name__ == "__main__":
    model = FUnIEGeneratorV1()
    x = torch.rand(1, 3, 256, 256)
    print(model(x).size())

    model = FUnIEGeneratorV2()
    x = torch.rand(1, 3, 256, 256)
    print(model(x).size())

    model = FUnIEDiscriminator()
    x1 = torch.rand(1, 3, 256, 256)
    x2 = torch.rand(1, 3, 256, 256)
    print(model(x1, x2).size())

    model = VGGContent()
    x = torch.rand(1, 3, 256, 256)
    print(model(x).size())
