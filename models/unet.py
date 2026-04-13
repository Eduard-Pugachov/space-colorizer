import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    # model uses two conv + ReLU layers
    # pipeline -- conv -> ReLU -> Conv -> ReLU
    # Conv2d -- input sensor, bunch of learnable kernels/filters,
    # each like a small 3x3 window, dot products
    # padding of 1 pixel border around input so output stays same HxW
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),  #normalizes activations so gradients flow better, more stable
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
        
# UNets typically use 2 convs bc larger effective receptive field
# plus fewer parameters than 1 huge conv 
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3):
        super().__init__()

        # encoder starts here, using DoubleConv class defined above
        # progressively compresses the image into higher-level features
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        # downsample, losing spatial detail
        # takes 2x2 block, outputs max, halves H and W
        self.pool = nn.MaxPool2d(2)

        # bottleneck implementation
        # most compressed representation (512 chan at lowest resolution)
        self.bottleneck = DoubleConv(256, 512)

        # decoder begins, once again using DoubleConv class
        # expands to original size while combining info w/
        # fine details from earlier w/ skip connections
        # skips feed early high res features directly
        # network can now reconstruct sharp edges/details
        self.up1  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(512, 256)   # 256 up + 256 skip

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)   # 128 up + 128 skip

        self.up3  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)    # 64 up + 64 skip

        self.final   = nn.Conv2d(64, out_ch, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # below is encoder path
        e1 = self.enc1(x)               # (B, 64,  H,   W)
        p1 = self.pool(e1)              # (B, 64,  H/2, W/2)

        e2 = self.enc2(p1)              # (B, 128, H/2, W/2)
        p2 = self.pool(e2)              # (B, 128, H/4, W/4)

        e3 = self.enc3(p2)              # (B, 256, H/4, W/4)
        p3 = self.pool(e3)              # (B, 256, H/8, W/8)


        # bottleneck
        b = self.bottleneck(p3)         # (B, 512, H/8, W/8)

        # decoder path start
        u1 = self.up1(b)                # (B, 256, H/4, W/4)
        u1 = torch.cat([u1, e3], dim=1) # (B, 512, H/4, W/4)
        d1 = self.dec1(u1)              # (B, 256, H/4, W/4)

        u2 = self.up2(d1)               # (B, 128, H/2, W/2)
        u2 = torch.cat([u2, e2], dim=1) # (B, 256, H/2, W/2)
        d2 = self.dec2(u2)              # (B, 128, H/2, W/2)

        u3 = self.up3(d2)               # (B, 64,  H,   W)
        u3 = torch.cat([u3, e1], dim=1) # (B, 128, H,   W)
        d3 = self.dec3(u3)              # (B, 64,  H,   W)
        return self.out_act(self.final(d3))  # (B, 3, H, W)
    
    # PyTorch internally calls UNet.forward(x) when
    # you call model(x), where model is a UNet()
    