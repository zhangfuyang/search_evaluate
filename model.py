import torch
import torch.nn as nn
from drn import drn_c_26

norm = nn.BatchNorm2d

class cornerModel(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(cornerModel, self).__init__()
        self.unet = UNet(in_channels,out_channels)

    def forward(self, x):
        mask = self.unet(x)
        return mask

class region_model(nn.Module):
    def __init__(self, in_channels=6, iters=2):
        super(region_model, self).__init__()
        self.iters = iters
        drn = drn_c_26(pretrained=True, num_classes=2, in_channels=in_channels)
        self.backbone = nn.Sequential(*list(drn.children())[:-7])
        self.unet_list = nn.ModuleList([UNet(128,64) for _ in range(iters)])
        self.decode = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.AdaptiveAvgPool2d((2,2))
        self.fc = nn.Linear(4*256, 2)

    def forward(self, img, edge_mask, region_masks):
        x = torch.cat((img, edge_mask), 1)
        x = x.expand(region_masks.shape[0], -1, -1, -1)
        x = torch.cat((x, region_masks), 1)
        feature_volume = self.backbone(x)
        index = torch.arange(region_masks.shape[0])
        if region_masks.shape[0] > 1:
            for iter_i in range(self.iters):
                neighbor_list = []
                for region_i in range(region_masks.shape[0]):
                    neighbors = feature_volume[index!=region_i]
                    neighbor = torch.max(neighbors, 0)[0]
                    neighbor_list.append(neighbor)
                neighbor = torch.stack(neighbor_list)
                feature_volume = self.unet_list[iter_i](torch.cat((feature_volume, neighbor), 1))

        pred = self.decode(feature_volume)
        pred = self.maxpool(pred)
        pred = torch.flatten(pred, 1)
        pred = self.fc(pred)
        return pred


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            norm(mid_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2),
            norm(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet_big(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_big, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        factor = 2 if bilinear else 1
        self.down6 = Down(1024,  2048// factor)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128 // factor, bilinear)
        self.up5 = Up(128, 64 // factor, bilinear)
        self.up6 = Up(64, 32, bilinear)
        self.out = nn.Sequential(
            nn.Conv2d(32, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        return self.out(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down6 = Down(512,  1024// factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        self.up5 = Up(64, 32 // factor, bilinear)
        self.up6 = Up(32, 16, bilinear)
        self.out = nn.Sequential(
            nn.Conv2d(16, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        return self.out(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            norm(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 1),
            norm(dim)
        )

    def forward(self, x):
        return x + self.block(x)
