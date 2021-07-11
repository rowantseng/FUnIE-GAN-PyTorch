import torch.nn as nn
import torch
import torchvision.models as models


class TotalGenLoss(nn.Module):
    def __init__(self, is_cuda):
        super(TotalGenLoss, self).__init__()
        self.vgg = VGGContent()
        if is_cuda:
            self.vgg = self.vgg.cuda()

    def forward(self, org_image, gen_image):
        vgg_org_image = self.vgg(org_image)
        vgg_gen_image = self.vgg(gen_image)

        bs = org_image.size(0)
        content_loss = ((vgg_org_image - vgg_gen_image) ** 2).mean(1)
        mae_gen_loss = (torch.abs(org_image - gen_image)).view(bs, -1).mean(1)
        return (0.7 * mae_gen_loss + 0.3 * content_loss).mean()


class VGGContent(nn.Module):
    def __init__(self):
        super(VGGContent, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True).features

    def forward(self, x):
        bs = x.size(0)
        return self.vgg(x).view(bs, -1)


def build_conv_block(in_chans, out_chans, kernel_size=3, stride=2, padding=1, use_bn=True, bn_momentum=0.8, use_leaky=False):
    layers = []
    layers.append(nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding))
    if use_leaky:
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    else:
        layers.append(nn.ReLU(inplace=True))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_chans, momentum=bn_momentum))
    return nn.Sequential(*layers)


def build_deconv_block(in_chans, out_chans, use_bn=True):
    layers = []
    layers.append(nn.Upsample(scale_factor=2,
                              mode="bilinear", align_corners=True))
    layers.append(nn.Conv2d(in_chans, out_chans, 3, 1, 1))
    layers.append(nn.ReLU(inplace=True))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_chans, momentum=0.8))
    return nn.Sequential(*layers)


class FUnIEGeneratorV1(nn.Module):
    def __init__(self, n_feats=32):
        super(FUnIEGeneratorV1, self).__init__()
        self.conv1 = build_conv_block(
            3, n_feats, 5, padding=2, use_bn=False)
        self.conv2 = build_conv_block(n_feats, n_feats*4, 4)
        self.conv3 = build_conv_block(n_feats*4, n_feats*8, 4)
        self.conv4 = build_conv_block(n_feats*8, n_feats*8)
        self.conv5 = build_conv_block(n_feats*8, n_feats*8)

        self.deconv1 = build_deconv_block(n_feats*8, n_feats*8)
        self.deconv2 = build_deconv_block(n_feats*16, n_feats*8)
        self.deconv3 = build_deconv_block(n_feats*16, n_feats*4)
        self.deconv4 = build_deconv_block(n_feats*8, n_feats*1)
        self.deconv5 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)

        # In this work, kernel size is 3 instead of 4
        self.final = nn.Conv2d(n_feats*2, 3, 3, 1, 1)
        self.act = nn.Tanh()

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
    def __init__(self, n_feats=32):
        super(FUnIEGeneratorV2, self).__init__()
        self.conv1 = build_conv_block(
            3, n_feats, 5, stride=1, padding=2, use_bn=False)
        # In this work, kernel size is 3 instead of 4
        self.conv2 = build_conv_block(
            n_feats, n_feats*2, stride=1, bn_momentum=0.75)
        # In this work, kernel size is 3 instead of 4
        self.conv3 = build_conv_block(
            n_feats*2, n_feats*2, stride=1, bn_momentum=0.75)
        self.conv4 = build_conv_block(
            n_feats*2, n_feats*4, stride=1, bn_momentum=0.75)
        self.conv5 = build_conv_block(
            n_feats*4, n_feats*4, stride=1, bn_momentum=0.75)
        self.conv6 = build_conv_block(
            n_feats*4, n_feats*8, stride=1, bn_momentum=0.75)

        self.pool = nn.MaxPool2d(2, 2)

        self.deconv1 = build_deconv_block(n_feats*8, n_feats*8)
        self.deconv2 = build_deconv_block(n_feats*12, n_feats*8)
        self.deconv3 = build_deconv_block(n_feats*10, n_feats*4)

        self.out1 = build_conv_block(
            n_feats*5, n_feats*4, stride=1, bn_momentum=0.75)
        self.out2 = build_conv_block(
            n_feats*4, n_feats*8, stride=1, bn_momentum=0.75)
        # In this work, kernel size is 3 instead of 4
        self.final = nn.Conv2d(n_feats*8, 3, 3, 1, 1)
        self.act = nn.Tanh()

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
    def __init__(self, n_feats=32):
        super(FUnIEDiscriminator, self).__init__()

        # Build discriminator blocks
        self.block1 = self._block(3*2, n_feats, False)
        self.block2 = self._block(n_feats, n_feats*2)
        self.block3 = self._block(n_feats*2, n_feats*4)
        self.block4 = self._block(n_feats*4, n_feats*8)

        # Validility block
        # In this work, kernel size is 3 instead of 4
        self.validility = nn.Conv2d(n_feats*8, 1, 3, 1, 1)

    def _block(self, in_chans, out_chans, use_bn=True):
        layers = []
        layers.append(nn.Conv2d(in_chans, out_chans, 3, 2, 1))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_chans, momentum=0.8))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # (B, 6, 256, 256)
        x = self.block1(x)              # (B, 32, 128, 128)
        x = self.block2(x)              # (B, 64, 64, 64)
        x = self.block3(x)              # (B, 128, 32, 32)
        x = self.block4(x)              # (B, 256, 16, 16)
        valid = self.validility(x)      # (B, 1, 16, 16)
        return valid.squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, n_feats=64):
        super(ResidualBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_feats, momentum=0.8))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_feats, momentum=0.8))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        x = self.block(x)
        return x + identity


class FUnIEUpGenerator(nn.Module):
    def __init__(self, n_feats=32):
        super(FUnIEUpGenerator, self).__init__()
        # Conv blocks
        self.conv1 = build_conv_block(
            3, n_feats, 5, padding=2, use_bn=False, use_leaky=True)
        self.conv2 = build_conv_block(n_feats, n_feats*4, 4, use_leaky=True)
        self.conv3 = build_conv_block(n_feats*4, n_feats*8, 4, use_leaky=True)
        self.conv4 = build_conv_block(n_feats*8, n_feats*8, use_leaky=True)
        self.conv5 = build_conv_block(n_feats*8, n_feats*8, use_leaky=True)

        # Three additional conv layers
        self.add_conv1 = nn.Conv2d(n_feats*8, 64, 3, stride=1, padding=1)
        self.add_conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.add_conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.res_block1 = ResidualBlock()
        self.res_block2 = ResidualBlock()
        self.res_block3 = ResidualBlock()
        self.res_block4 = ResidualBlock()
        self.res_block5 = ResidualBlock()

        # Deconv blocks
        self.deconv1 = self._deconv_block(n_feats*2, n_feats*8)
        self.deconv2 = self._deconv_block(n_feats*(8+8), n_feats*8)
        self.deconv3 = self._deconv_block(n_feats*(8+8), n_feats*4)
        self.deconv4 = self._deconv_block(n_feats*(4+4), n_feats*1)

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)

        # In this work, kernel size is 3 instead of 4
        self.final = nn.Conv2d(n_feats*2, 3, 3, stride=1, padding=1)
        self.act = nn.Tanh()

    def _deconv_block(self, in_chans, out_chans, use_bn=True):
        layers = []
        layers.append(nn.Upsample(scale_factor=2,
                                  mode="bilinear", align_corners=True))
        layers.append(nn.Conv2d(in_chans, out_chans, 3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_chans, momentum=0.8))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Downsample
        d1 = self.conv1(x)   # (B, 32, 128, 128)
        d2 = self.conv2(d1)  # (B, 128, 64, 64)
        d3 = self.conv3(d2)  # (B, 256, 32, 32)
        d4 = self.conv4(d3)  # (B, 256, 16, 16)
        d5 = self.conv5(d4)  # (B, 256, 8, 8)

        # Additional conv layers
        a1 = self.relu(self.add_conv1(d5))  # (B, 64, 8, 8)
        a2 = self.relu(self.add_conv2(a1))
        bridge = self.relu(self.add_conv3(a2))

        # Residual blocks
        bridge = self.res_block1(bridge)
        bridge = self.res_block2(bridge)
        bridge = self.res_block3(bridge)
        bridge = self.res_block4(bridge)
        bridge = self.res_block5(bridge)
        bridge += a1

        # Upsample
        u1 = torch.cat([self.deconv1(bridge), d4], dim=1)  # (B, 512, 16, 16)
        u2 = torch.cat([self.deconv2(u1), d3], dim=1)      # (B, 512, 32, 32)
        u3 = torch.cat([self.deconv3(u2), d2], dim=1)      # (B, 256, 64, 64)
        u4 = torch.cat([self.deconv4(u3), d1], dim=1)      # (B, 64, 128, 128)
        u4 = self.up(u4)                                   # (B, 64, 256, 256)
        return self.act(self.final(u4))


class FUnIEUpDiscriminator(nn.Module):
    def __init__(self, n_feats=32):
        super(FUnIEUpDiscriminator, self).__init__()

        # Build discriminator blocks
        self.block1 = build_conv_block(
            3, n_feats, use_bn=False, use_leaky=True)
        self.block2 = build_conv_block(n_feats, n_feats*2, use_leaky=True)
        self.block3 = build_conv_block(n_feats*2, n_feats*4, use_leaky=True)
        self.block4 = build_conv_block(n_feats*4, n_feats*8, use_leaky=True)
        self.block5 = build_conv_block(
            n_feats*8, n_feats*8, stride=1, use_leaky=True)

        # Validility block
        # In this work, kernel size is 3 instead of 4
        self.validility = nn.Conv2d(n_feats*8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.block1(x)              # (B, 32, 128, 128)
        x = self.block2(x)              # (B, 64, 64, 64)
        x = self.block3(x)              # (B, 128, 32, 32)
        x = self.block4(x)              # (B, 256, 16, 16)
        x = self.block5(x)              # (B, 256, 16, 16)
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

    model = FUnIEUpGenerator()
    x = torch.rand(1, 3, 256, 256)
    print(model(x).size())

    model = FUnIEUpDiscriminator()
    x = torch.rand(1, 3, 256, 256)
    print(model(x).size())
