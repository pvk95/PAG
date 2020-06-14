import torch
import torch.nn as nn
import torch.nn.functional as F
from models.grid_attention_layer import GridAttentionBlock3D
from models.mask import Mask3D


def _make_block(block, input_channels=None):
    if input_channels is not None:
        layers = [block(input_channels)]
    else:
        layers = [block()]
    return nn.Sequential(*layers)


# ============================================
# ================= UpSample =================
# ============================================


class UpSample(nn.Module):
    def __init__(self, input_channels):
        super(UpSample, self).__init__()

        output_channels = input_channels // 2

        self.conv = nn.Conv3d(in_channels=input_channels, out_channels=output_channels,
                              kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, inputs):
        x = F.leaky_relu_(self.conv(inputs))
        x = self.upsample(x)

        return x


# ============================================
# ================= DownSample ===============
# ============================================


class DownSample(nn.Module):
    def __init__(self, input_channels):
        super(DownSample, self).__init__()

        self.input_channels = input_channels
        self.output_channels = input_channels * 2

        self.conv_1 = nn.Conv3d(in_channels=self.input_channels, out_channels=self.output_channels,
                                kernel_size=3, stride=2, padding=1)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.leaky_relu_(self.conv_1(inputs))

        return x


class ResNet(nn.Module):
    def __init__(self, n_channels):
        super(ResNet, self).__init__()

        self.n_channels = n_channels

        self.layer_1 = nn.GroupNorm(num_groups=8, num_channels=self.n_channels)

        # self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv3d(in_channels=self.n_channels, out_channels=self.n_channels,
                                kernel_size=3, padding=1)
        self.layer_2 = nn.GroupNorm(num_groups=8, num_channels=self.n_channels)
        self.conv_2 = nn.Conv3d(in_channels=self.n_channels, out_channels=self.n_channels,
                                kernel_size=3, padding=1)

    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = F.leaky_relu_(x)
        x = self.conv_1(x)
        x = self.layer_2(x)
        x = F.leaky_relu_(x)
        x = self.conv_2(x)
        x = x + inputs

        return x


# ============================================
# ============== Regularizer =================
# ============================================


class Regularizer(nn.Module):
    def __init__(self):
        super(Regularizer, self).__init__()

        self.upsample_1 = _make_block(UpSample, input_channels=256)
        self.resnet_10 = _make_block(ResNet, input_channels=128)

        self.upsample_2 = _make_block(UpSample, input_channels=128)
        self.resnet_11 = _make_block(ResNet, input_channels=64)

        self.upsample_3 = _make_block(UpSample, input_channels=64)
        self.resnet_12 = _make_block(ResNet, input_channels=32)

        self.conv_2 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample_1(x)
        x = self.resnet_10(x)
        x = self.upsample_2(x)
        x = self.resnet_11(x)
        x = self.upsample_3(x)
        x = self.resnet_12(x)
        x = F.leaky_relu_(self.conv_2(x))

        return x


# ============================================
# =================== PET ====================
# ============================================


class PET(nn.Module):
    def __init__(self):
        super(PET, self).__init__()

        self.upsample_1 = _make_block(UpSample, input_channels=256)
        self.resnet_10 = _make_block(ResNet, input_channels=128)

        self.upsample_2 = _make_block(UpSample, input_channels=128)
        self.resnet_11 = _make_block(ResNet, input_channels=64)

        self.upsample_3 = _make_block(UpSample, input_channels=64)
        self.resnet_12 = _make_block(ResNet, input_channels=32)

        self.conv_2 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

    def call(self, inputs):
        x, skip_1, skip_2, skip_3 = inputs

        x = self.upsample_1(x) + skip_3
        del skip_3
        x = self.resnet_10(x)
        x = self.upsample_2(x) + skip_2
        del skip_2
        x = self.resnet_11(x)
        x = self.upsample_3(x) + skip_1
        del skip_1
        x = self.resnet_12(x)

        x = self.conv_2(x)

        return x


# ============================================
# ================ Segmentation ==============
# ============================================


class Segmentation(nn.Module):

    def __init__(self):
        super(Segmentation, self).__init__()

        self.upsample_1 = _make_block(UpSample, input_channels=256)
        self.resnet_10 = _make_block(ResNet, input_channels=128)

        self.upsample_2 = _make_block(UpSample, input_channels=128)
        self.resnet_11 = _make_block(ResNet, input_channels=64)

        self.upsample_3 = _make_block(UpSample, input_channels=64)
        self.resnet_12 = _make_block(ResNet, input_channels=32)

        self.conv_2 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

    def call(self, inputs):

        x, skip_1, skip_2, skip_3 = inputs

        x = self.upsample_1(x) + skip_3
        del skip_3
        x = self.resnet_10(x)
        x = self.upsample_2(x) + skip_2
        del skip_2
        x = self.resnet_11(x)
        x = self.upsample_3(x) + skip_1
        del skip_1
        x = self.resnet_12(x)
        x = torch.sigmoid(self.conv_2(x))

        return x


# ============================================
# ================ Attn Seg ==================
# ============================================


class AttnSeg(nn.Module):
    def __init__(self):
        super(AttnSeg, self).__init__()

        self.upsample_1 = _make_block(UpSample, input_channels=256)
        self.resnet_10 = _make_block(ResNet, input_channels=128)
        self.gate_1 = GridAttentionBlock3D(in_channels=128, gating_channels=256, inter_channels=128)

        self.upsample_2 = _make_block(UpSample, input_channels=128)
        self.resnet_11 = _make_block(ResNet, input_channels=64)
        self.gate_2 = GridAttentionBlock3D(in_channels=64, gating_channels=256, inter_channels=64)

        self.upsample_3 = _make_block(UpSample, input_channels=64)
        self.resnet_12 = _make_block(ResNet, input_channels=32)
        self.gate_3 = GridAttentionBlock3D(in_channels=32, gating_channels=256, inter_channels=32)

        self.conv_2 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

    def call(self, inputs):
        x, skip_1, skip_2, skip_3 = inputs
        gate = x

        x = self.upsample_1(x) + self.gate_1(skip_3, gate)
        del skip_3
        x = self.resnet_10(x)

        x = self.upsample_2(x) + self.gate_2(skip_2, gate)
        del skip_2
        x = self.resnet_11(x)

        x = self.upsample_3(x) + self.gate_3(skip_1, gate)
        del skip_1
        x = self.resnet_12(x)

        x = torch.sigmoid(self.conv_2(x))

        return x
