import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BASIC BUILDING BLOCKS
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()

        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()

        hidden = c // 2

        self.cv1 = ConvBNAct(
            c,
            hidden,
            1,
            1
        )

        self.cv2 = ConvBNAct(
            hidden,
            c,
            3,
            1
        )

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    C2f-style feature block.
    """

    def __init__(self, c1, c2, n=2):
        super().__init__()

        hidden = c2 // 2

        self.cv1 = ConvBNAct(
            c1,
            hidden * 2,
            1,
            1
        )

        self.blocks = nn.ModuleList(
            [
                Bottleneck(hidden)
                for _ in range(n)
            ]
        )

        self.cv2 = ConvBNAct(
            hidden * (2 + n),
            c2,
            1,
            1
        )

    def forward(self, x):

        y = list(
            self.cv1(x).chunk(2, dim=1)
        )

        for block in self.blocks:
            y.append(block(y[-1]))

        return self.cv2(
            torch.cat(y, dim=1)
        )


# ============================================================
# SPPF
# ============================================================

class SPPF(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()

        hidden = c1 // 2

        self.cv1 = ConvBNAct(
            c1,
            hidden,
            1,
            1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=5,
            stride=1,
            padding=2
        )

        self.cv2 = ConvBNAct(
            hidden * 4,
            c2,
            1,
            1
        )

    def forward(self, x):

        x = self.cv1(x)

        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        return self.cv2(
            torch.cat(
                [x, y1, y2, y3],
                dim=1
            )
        )


# ============================================================
# CSPDARKNET BACKBONE
# ============================================================

class CSPDarknet(nn.Module):
    """
    AquaYOLO CSPDarknet backbone.

    Input:
        640 x 640 x 3

    Feature outputs:

        P3 = 80  x 80  x 256
        P4 = 40  x 40  x 512
        P5 = 20  x 20  x 1024
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Stem
        # 640 -> 320
        # ----------------------------------------------------

        self.stem = ConvBNAct(
            3,
            64,
            3,
            2
        )

        # ----------------------------------------------------
        # P2
        # 320 -> 160
        # ----------------------------------------------------

        self.p2_down = ConvBNAct(
            64,
            128,
            3,
            2
        )

        self.p2_c2f = C2f(
            128,
            128,
            n=2
        )

        # ----------------------------------------------------
        # P3
        # 160 -> 80
        # ----------------------------------------------------

        self.p3_down = ConvBNAct(
            128,
            256,
            3,
            2
        )

        self.p3_c2f = C2f(
            256,
            256,
            n=3
        )

        # ----------------------------------------------------
        # P4
        # 80 -> 40
        # ----------------------------------------------------

        self.p4_down = ConvBNAct(
            256,
            512,
            3,
            2
        )

        self.p4_c2f = C2f(
            512,
            512,
            n=3
        )

        # ----------------------------------------------------
        # P5
        # 40 -> 20
        # ----------------------------------------------------

        self.p5_down = ConvBNAct(
            512,
            1024,
            3,
            2
        )

        self.p5_c2f = C2f(
            1024,
            1024,
            n=2
        )

        # ----------------------------------------------------
        # SPPF
        # ----------------------------------------------------

        self.sppf = SPPF(
            1024,
            1024
        )

    def forward(self, x):

        # Stem
        x = self.stem(x)

        # P2
        x = self.p2_down(x)
        x = self.p2_c2f(x)

        # P3
        x = self.p3_down(x)
        p3 = self.p3_c2f(x)

        # P4
        x = self.p4_down(p3)
        p4 = self.p4_c2f(x)

        # P5
        x = self.p5_down(p4)
        p5 = self.p5_c2f(x)

        # SPPF
        p5 = self.sppf(p5)

        return p3, p4, p5


# ============================================================
# AQUA-ATTN
# ============================================================

class SpectralAttenuationCompensator(nn.Module):
    """
    SAC - Spectral Attenuation Compensator.

    Learns channel-wise attention for underwater
    spectral attenuation compensation.
    """

    def __init__(
        self,
        channels,
        reduction=16
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            32
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden,
                1
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                channels,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        weights = self.fc(
            self.pool(x)
        )

        return x * weights


class TurbidityContrastGate(nn.Module):
    """
    TCG - Turbidity Contrast Gate.

    Learns a spatial attention mask for suppressing
    low-information underwater haze/background regions.
    """

    def __init__(self, channels):
        super().__init__()

        hidden = channels // 4

        self.gate = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden,
                1,
                1
            ),

            nn.BatchNorm2d(
                hidden
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                1,
                3,
                1,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        gate = self.gate(x)

        return x * gate


class AquaAttn(nn.Module):
    """
    Aqua-Attn.

    Position:

        CSPDarknet
             |
            SPPF
             |
         Aqua-Attn
         /       \
       SAC       TCG
         \       /
          Fusion
             |
            PANet
    """

    def __init__(
        self,
        channels=1024
    ):
        super().__init__()

        self.sac = SpectralAttenuationCompensator(
            channels
        )

        self.tcg = TurbidityContrastGate(
            channels
        )

        self.proj = ConvBNAct(
            channels,
            channels,
            1,
            1
        )

    def forward(self, x):

        spectral = self.sac(x)

        # Use the actual spatial gate for multiplicative fusion
        gate = self.tcg.gate(x)

        fused = spectral * gate

        return self.proj(fused)


# ============================================================
# PANET NECK
# ============================================================

class PANet(nn.Module):
    """
    PANet multi-scale feature fusion.

    Input:

        P3 = 80  x 80  x 256
        P4 = 40  x 40  x 512
        P5 = 20  x 20  x 1024

    Output:

        P3' = 80  x 80  x 256
        P4' = 40  x 40  x 512
        P5' = 20  x 20  x 1024
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # TOP-DOWN PATH
        # ----------------------------------------------------

        self.p5_reduce = ConvBNAct(
            1024,
            512,
            1,
            1
        )

        # P5-up = 512
        # P4 = 512
        # total = 1024

        self.p4_td = C2f(
            1024,
            512,
            n=2
        )

        self.p4_reduce = ConvBNAct(
            512,
            256,
            1,
            1
        )

        # P4-up = 256
        # P3 = 256
        # total = 512

        self.p3_td = C2f(
            512,
            256,
            n=2
        )

        # ----------------------------------------------------
        # BOTTOM-UP PATH
        # ----------------------------------------------------

        self.p3_down = ConvBNAct(
            256,
            256,
            3,
            2
        )

        # p3_down = 256
        # p4_td   = 512
        # p4      = 512
        #
        # total = 1280

        self.p4_bu = C2f(
            1280,
            512,
            n=2
        )

        self.p4_down = ConvBNAct(
            512,
            512,
            3,
            2
        )

        # p4_down = 512
        # p5      = 1024
        # p5_td   = 512
        #
        # total = 2048

        self.p5_bu = C2f(
            2048,
            1024,
            n=2
        )

    def forward(
        self,
        p3,
        p4,
        p5
    ):

        # ====================================================
        # TOP-DOWN
        # ====================================================

        p5_td = self.p5_reduce(
            p5
        )

        p5_up = F.interpolate(
            p5_td,
            size=p4.shape[-2:],
            mode="nearest"
        )

        p4_td = self.p4_td(
            torch.cat(
                [
                    p5_up,
                    p4
                ],
                dim=1
            )
        )

        p4_up = self.p4_reduce(
            p4_td
        )

        p4_up = F.interpolate(
            p4_up,
            size=p3.shape[-2:],
            mode="nearest"
        )

        p3_td = self.p3_td(
            torch.cat(
                [
                    p4_up,
                    p3
                ],
                dim=1
            )
        )

        # ====================================================
        # BOTTOM-UP
        # ====================================================

        p3_down = self.p3_down(
            p3_td
        )

        p4_out = self.p4_bu(
            torch.cat(
                [
                    p3_down,
                    p4_td,
                    p4
                ],
                dim=1
            )
        )

        p4_down = self.p4_down(
            p4_out
        )

        p5_out = self.p5_bu(
            torch.cat(
                [
                    p4_down,
                    p5,
                    p5_td
                ],
                dim=1
            )
        )

        return (
            p3_td,
            p4_out,
            p5_out
        )


# ============================================================
# DECOUPLED ANCHOR-FREE DETECTION HEAD
# ============================================================

class DetectionBranch(nn.Module):

    def __init__(
        self,
        channels,
        nc
    ):
        super().__init__()

        self.stem = ConvBNAct(
            channels,
            channels,
            3,
            1
        )

        # Objectness
        self.obj = nn.Conv2d(
            channels,
            1,
            1
        )

        # Classification
        self.cls = nn.Conv2d(
            channels,
            nc,
            1
        )

        # Bounding box
        self.box = nn.Conv2d(
            channels,
            4,
            1
        )

    def forward(self, x):

        x = self.stem(x)

        return {
            "obj": self.obj(x),
            "cls": self.cls(x),
            "box": self.box(x)
        }


class DecoupledAnchorFreeHead(nn.Module):

    def __init__(
        self,
        nc
    ):
        super().__init__()

        # Small objects
        self.p3_head = DetectionBranch(
            256,
            nc
        )

        # Medium objects
        self.p4_head = DetectionBranch(
            512,
            nc
        )

        # Large objects
        self.p5_head = DetectionBranch(
            1024,
            nc
        )

    def forward(
        self,
        p3,
        p4,
        p5
    ):

        return [
            self.p3_head(p3),
            self.p4_head(p4),
            self.p5_head(p5)
        ]


# ============================================================
# COMPLETE AQUAYOLO
# ============================================================

class AquaYOLO(nn.Module):
    """
    Complete AquaYOLO model.

    Pipeline:

        Input
          |
          v
    CSPDarknet
          |
          v
        SPPF
          |
          v
      Aqua-Attn
       /     \
     SAC     TCG
       \     /
        Fusion
          |
          v
        PANet
          |
          v
    Decoupled
    Anchor-Free
       Head
      / | \
    P3 P4 P5
    """

    def __init__(
        self,
        nc=26
    ):
        super().__init__()

        self.nc = nc

        # Backbone
        self.backbone = CSPDarknet()

        # Aqua-Attn
        self.aqua_attn = AquaAttn(
            channels=1024
        )

        # PANet
        self.neck = PANet()

        # Detection head
        self.head = DecoupledAnchorFreeHead(
            nc
        )

    def forward(self, x):

        # Backbone
        p3, p4, p5 = self.backbone(x)

        # Aqua-Attn is inserted after SPPF
        # and before PANet.
        p5 = self.aqua_attn(p5)

        # PANet
        p3_out, p4_out, p5_out = self.neck(
            p3,
            p4,
            p5
        )

        # Detection head
        return self.head(
            p3_out,
            p4_out,
            p5_out
        )