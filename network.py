import torch
import torch.nn as nn
from typing import Sequence


ACT = {'relu': nn.ReLU, 'leaky': nn.LeakyReLU, 'prelu': nn.PReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}


class SCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
        factor: int = 8,
        dropout: float = 0.,
        mode: str = 'add',
        local_act: str = None,
        spatial_act: str = 'tanh',
    ):
        super().__init__()
        self.HLA = LocalAppearance(in_channels, num_classes, filters, dropout, mode)
        self.down = nn.AvgPool3d(factor, factor, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=factor, mode='trilinear', align_corners=True)
        self.local_act = ACT[local_act]() if local_act else None
        self.HSC = nn.Sequential(
            nn.Conv3d(filters, filters, 7, 1, 3, bias=False),
            nn.Conv3d(filters, filters, 7, 1, 3, bias=False),
            nn.Conv3d(filters, filters, 7, 1, 3, bias=False),
            nn.Conv3d(filters, num_classes, 7, 1, 3, bias=False),
        )
        self.spatial_act = ACT[spatial_act]()
        nn.init.trunc_normal_(self.HSC[-1].weight, 0, 1e-4)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        d1, HLA = self.HLA(x)
        if self.local_act:
            HLA = self.local_act(HLA)
        HSC = self.up(self.spatial_act(self.HSC(self.down(d1))))
        heatmap = HLA * HSC
        return heatmap, HLA, HSC


class LocalAppearance(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
        dropout: float = 0.,
        mode: str = 'add',
    ):
        super().__init__()
        self.mode = mode
        self.pool = nn.AvgPool3d(2, 2, ceil_mode=True)
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.in_conv = self.Block(in_channels, filters)
        self.enc1 = self.Block(filters, filters, dropout)
        self.enc2 = self.Block(filters, filters, dropout)
        self.enc3 = self.Block(filters, filters, dropout)
        self.enc4 = self.Block(filters, filters, dropout)
        if mode == 'add':
            self.dec3 = self.Block(filters, filters, dropout)
            self.dec2 = self.Block(filters, filters, dropout)
            self.dec1 = self.Block(filters, filters, dropout)
        else:
            self.dec3 = self.Block(2*filters, filters, dropout)
            self.dec2 = self.Block(2*filters, filters, dropout)
            self.dec1 = self.Block(2*filters, filters, dropout)
        self.out_conv = nn.Conv3d(filters, num_classes, 1, bias=False)
        nn.init.trunc_normal_(self.out_conv.weight, 0, 1e-4)

    def Block(self, in_channels, out_channels, dropout=0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        if self.mode == 'add':
            d3 = self.dec3(self.up(e4)+e3)
            d2 = self.dec2(self.up(d3)+e2)
            d1 = self.dec1(self.up(d2)+e1)
        else:
            d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        out = self.out_conv(d1)
        return d1, out


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
    ) -> None:
        super().__init__()
        self.in_conv = self.Block(in_channels, filters)
        self.enc1 = self.Block(filters, filters)
        self.enc2 = self.Block(filters, filters)
        self.enc3 = self.Block(filters, filters)
        self.enc4 = self.Block(filters, filters)
        self.enc5 = self.Block(filters, filters)
        self.dec4 = self.Block(2*filters, filters)
        self.dec3 = self.Block(2*filters, filters)
        self.dec2 = self.Block(2*filters, filters)
        self.dec1 = self.Block(2*filters, filters)
        self.out_conv = nn.Conv3d(filters, num_classes, 1, bias=False)
        self.pool = nn.AvgPool3d(2, 2, ceil_mode=True)
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        out = self.out_conv(d1)
        return torch.sigmoid(out)

    def Block(self, in_channels, out_channels, dropout=0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
