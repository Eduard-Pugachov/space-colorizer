import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    # model uses two conv + ReLU layers
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block == nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        def forward(self, x):
            return self.block(x)
        
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3):
        super().__init__()

        # encoder starts here, using DoubleConv class defined above
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # bottleneck implementation
        self.bottleneck = DoubleConv(128, 256)

        # decoder begins, once again using DoubleConv class
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(256, 128)

        self.up2 = nn.Conv2d(64, out_ch, kernel_size=1)
        self.out_act = nn.Sigmoid() 

    def forward(self, x):
        # below is encoder path
        e1 = self.encl1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        # bottleneck
        b = self.bottleneck(p2)

        # decoder path start
        u1 = self.up1(b)
        u1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(u2)

        out = self.final(d2)
        return self.out_act(out)
    