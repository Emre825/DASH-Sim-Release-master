import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Building blocks ----------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.act(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 + Pointwise 1x1 (+BN+ReLU)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.dw_bn(x); x = self.act(x)
        x = self.pw(x); x = self.pw_bn(x); x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    """Simple 2x ConvBNReLU with identity/1x1 projection if channels/stride differ."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, k=3, s=stride, p=1)
        self.conv2 = ConvBNReLU(out_ch, out_ch, k=3, s=1, p=1)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = ConvBNReLU(in_ch, out_ch, k=1, s=stride, p=0)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        return out

class SqueezeExcite(nn.Module):
    """Lightweight channel attention to add more deps/compute variety."""
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc1(s); s = self.act(s); s = self.fc2(s); s = self.gate(s)
        return x * s

# ---------- Backbone (C3, C4, C5) ----------

class TinyResBackbone(nn.Module):
    """
    Input: [B,3,224,224]
    Outputs:
      C3: [B, 128, 56, 56]
      C4: [B, 256, 28, 28]
      C5: [B, 512, 14, 14]
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, 3, 2, 1),      # 112x112
            ConvBNReLU(32, 64, 3, 1, 1),
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)  # 56x56

        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            SqueezeExcite(128),
        )  # C3: 128 @ 56x56

        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),  # 28x28
            ResidualBlock(256, 256, stride=1),
            SqueezeExcite(256),
        )  # C4: 256 @ 28x28

        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),  # 14x14
            ResidualBlock(512, 512, stride=1),
            SqueezeExcite(512),
        )  # C5: 512 @ 14x14

    def forward(self, x):
        x = self.stem(x)         # [B,64,112,112]
        x = self.pool(x)         # [B,64,56,56]
        c3 = self.stage2(x)      # [B,128,56,56]
        c4 = self.stage3(c3)     # [B,256,28,28]
        c5 = self.stage4(c4)     # [B,512,14,14]
        return c3, c4, c5

# ---------- BiFPN-like fusion (top-down + bottom-up) ----------

class BiFPNBlock(nn.Module):
    """
    Unifies channels (e.g., 160), then:
      Top-down:   P5 -> P4 -> P3 (with upsample + add) + separable conv
      Bottom-up:  P3 -> P4 -> P5 (with downsample + add) + separable conv
    """
    def __init__(self, in_c3=128, in_c4=256, in_c5=512, fpn_ch=160):
        super().__init__()
        # lateral projections
        self.l3 = nn.Conv2d(in_c3, fpn_ch, 1)
        self.l4 = nn.Conv2d(in_c4, fpn_ch, 1)
        self.l5 = nn.Conv2d(in_c5, fpn_ch, 1)

        # convs after fusion
        self.p3_td = DepthwiseSeparableConv(fpn_ch, fpn_ch)
        self.p4_td = DepthwiseSeparableConv(fpn_ch, fpn_ch)
        self.p5_td = DepthwiseSeparableConv(fpn_ch, fpn_ch)

        self.p3_bu = DepthwiseSeparableConv(fpn_ch, fpn_ch)
        self.p4_bu = DepthwiseSeparableConv(fpn_ch, fpn_ch)
        self.p5_bu = DepthwiseSeparableConv(fpn_ch, fpn_ch)

        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, c3, c4, c5):
        # Lateral
        p3 = self.l3(c3)     # [B,fpn,56,56]
        p4 = self.l4(c4)     # [B,fpn,28,28]
        p5 = self.l5(c5)     # [B,fpn,14,14]

        # Top-down pathway
        p5_td = self.p5_td(p5)
        p4_td = self.p4_td(p4 + F.interpolate(p5_td, scale_factor=2, mode='nearest'))      # 28x28
        p3_td = self.p3_td(p3 + F.interpolate(p4_td, scale_factor=2, mode='nearest'))      # 56x56

        # Bottom-up pathway
        p4_bu = self.p4_bu(p4_td + self.pool(p3_td))   # 28x28
        p5_bu = self.p5_bu(p5_td + self.pool(p4_bu))   # 14x14
        p3_bu = self.p3_bu(p3_td)                      # keep 56x56 refined

        return p3_bu, p4_bu, p5_bu   # (P3,P4,P5)

# ---------- Head with multi-scale concat & prediction ----------

class MultiScaleHead(nn.Module):
    """
    Upsample P4 and P5 to P3 resolution; concatenate [P3, up(P4), up2(P5)].
    Process with separable convs, then project to 1 channel + Sigmoid.
    """
    def __init__(self, fpn_ch=160, mid_ch=128):
        super().__init__()
        self.conv_p3 = DepthwiseSeparableConv(fpn_ch, mid_ch)
        self.conv_p4 = DepthwiseSeparableConv(fpn_ch, mid_ch)
        self.conv_p5 = DepthwiseSeparableConv(fpn_ch, mid_ch)

        self.fuse = DepthwiseSeparableConv(mid_ch * 3, 128)
        self.refine1 = DepthwiseSeparableConv(128, 64)
        self.refine2 = DepthwiseSeparableConv(64, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)
        self.sig = nn.Sigmoid()

    def forward(self, p3, p4, p5):
        # Normalize channels first
        p3 = self.conv_p3(p3)                        # [B,128,56,56]
        p4 = self.conv_p4(F.interpolate(p4, scale_factor=2, mode='nearest'))  # -> 56x56
        p5 = self.conv_p5(F.interpolate(p5, scale_factor=4, mode='nearest'))  # -> 56x56

        x = torch.cat([p3, p4, p5], dim=1)           # [B,384,56,56]
        x = self.fuse(x)                              # [B,128,56,56]

        # Up to 224x224 with two stages and concat a light skip from stem-like feature
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 112x112
        x = self.refine1(x)                                   # [B,64,112,112]
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 224x224
        x = self.refine2(x)                                   # [B,32,224,224]

        x = self.out_conv(x)                                  # [B,1,224,224]
        x = self.sig(x)
        return x

# ---------- Full Model ----------

class BiFPNNet(nn.Module):
    """
    Input:  [B,3,224,224]
    Output: [B,1,224,224] (sigmoid map)
    Rich in: MaxPool, Upsample, cat, adds, residuals, depthwise separable convs.
    """
    def __init__(self, fpn_ch=160):
        super().__init__()
        self.backbone = TinyResBackbone()
        self.bifpn1 = BiFPNBlock(128, 256, 512, fpn_ch=fpn_ch)
        self.bifpn2 = BiFPNBlock(fpn_ch, fpn_ch, fpn_ch, fpn_ch=fpn_ch)  # stacked (same channels)
        self.head = MultiScaleHead(fpn_ch=fpn_ch, mid_ch=128)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)      # multi-scale features
        p3, p4, p5 = self.bifpn1(c3, c4, c5)
        p3, p4, p5 = self.bifpn2(p3, p4, p5)  # second fusion pass
        y = self.head(p3, p4, p5)
        return y

# ---------- quick smoke test ----------
if __name__ == "__main__":
    model = BiFPNNet().eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print("Output:", y.shape)  # [1,1,224,224]
