import torch
from torch import nn
from torch.nn import functional as F
from models.model_utils import _make_block
from models.model_utils import DownSample
from models.model_utils import UpSample
from models.model_utils import ResNet


# ============================================
# ========== PET feature extractor ===========
# ============================================

# Extract features from PET
class PetExtract(nn.Module):
    def __init__(self):
        super(PetExtract, self).__init__()

        in_channels = 1

        self.conv_1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.resnet_1 = _make_block(ResNet, input_channels=32)

    def forward(self, inputs):
        x = F.relu(self.conv_1(inputs))
        x = self.resnet_1(x)

        return x


class _PetAttentionBlockND(nn.Module):
    def __init__(self, in_channels, pet_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(_PetAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.pet_channels = pet_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)

        self.theta_pet = conv_nd(in_channels=self.pet_channels, out_channels=self.inter_channels,
                                 kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                                 bias=False)

        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)

        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g, pet_features=None):

        output = self.operation_function(x, g, pet_features)
        return output

    def _concatenation(self, x, g, pet_features):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)

        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)

        if pet_features is not None:
            pet_features = self.theta_pet(pet_features)
            pet_g = F.interpolate(pet_features, size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)
            f = theta_x + phi_g + pet_g
        else:
            f = theta_x + phi_g
        f = F.relu_(f)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y


class PetAttentionBlock3D(_PetAttentionBlockND):
    def __init__(self, in_channels, pet_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(PetAttentionBlock3D, self).__init__(in_channels,
                                                  inter_channels=inter_channels,
                                                  pet_channels=pet_channels,
                                                  gating_channels=gating_channels,
                                                  dimension=3, mode=mode,
                                                  sub_sample_factor=sub_sample_factor)


class AttnPet(nn.Module):
    def __init__(self):
        super(AttnPet, self).__init__()

        self.upsample_1 = _make_block(UpSample, input_channels=256)
        self.resnet_10 = _make_block(ResNet, input_channels=128)
        self.gate_1 = PetAttentionBlock3D(in_channels=128, pet_channels=32,
                                          gating_channels=256, inter_channels=128)

        self.upsample_2 = _make_block(UpSample, input_channels=128)
        self.resnet_11 = _make_block(ResNet, input_channels=64)
        self.gate_2 = PetAttentionBlock3D(in_channels=64, pet_channels=32,
                                          gating_channels=256, inter_channels=64)

        self.upsample_3 = _make_block(UpSample, input_channels=64)
        self.resnet_12 = _make_block(ResNet, input_channels=32)
        self.gate_3 = PetAttentionBlock3D(in_channels=32, pet_channels=32,
                                          gating_channels=256, inter_channels=32)

        self.conv_2 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

    def call(self, inputs):
        x, skip_1, skip_2, skip_3, pet_features = inputs
        gate = x

        x = self.upsample_1(x) + self.gate_1(skip_3, gate, pet_features)
        del skip_3
        x = self.resnet_10(x)

        x = self.upsample_2(x) + self.gate_2(skip_2, gate, pet_features)
        del skip_2
        x = self.resnet_11(x)

        x = self.upsample_3(x) + self.gate_3(skip_1, gate, pet_features)
        del skip_1
        x = self.resnet_12(x)

        x = torch.sigmoid(self.conv_2(x))

        return x


# PET-guided Segmentation
class SegAttnPet(nn.Module):
    def __init__(self):
        super(SegAttnPet, self).__init__()

        in_channels = 1

        self.conv_1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)

        self.drop = nn.Dropout(p=0.2)

        self.resnet_1 = _make_block(ResNet, input_channels=32)

        self.downsample_1 = _make_block(DownSample, input_channels=32)
        self.resnet_2 = _make_block(ResNet, input_channels=64)
        self.resnet_3 = _make_block(ResNet, input_channels=64)

        self.downsample_2 = _make_block(DownSample, input_channels=64)
        self.resnet_4 = _make_block(ResNet, input_channels=128)
        self.resnet_5 = _make_block(ResNet, input_channels=128)

        self.downsample_3 = _make_block(DownSample, input_channels=128)
        self.resnet_6 = _make_block(ResNet, input_channels=256)
        self.resnet_7 = _make_block(ResNet, input_channels=256)
        self.resnet_8 = _make_block(ResNet, input_channels=256)
        self.resnet_9 = _make_block(ResNet, input_channels=256)

        self.pet_extract = PetExtract()

        self.seg = AttnPet()

    def forward(self, inputs, pet_image=None):
        x = F.relu(self.drop(self.conv_1(inputs)))
        x = self.resnet_1(x)
        skip_1 = x

        x = self.downsample_1(x)
        x = self.resnet_2(x)
        x = self.resnet_3(x)
        skip_2 = x

        x = self.downsample_2(x)
        x = self.resnet_4(x)
        x = self.resnet_5(x)
        skip_3 = x

        x = self.downsample_3(x)
        x = self.resnet_6(x)
        x = self.resnet_7(x)
        x = self.resnet_8(x)
        x = self.resnet_9(x)

        if pet_image is not None:
            pet_features = self.pet_extract(pet_image)
        else:
            pet_features = None

        seg = self.seg.call((x, skip_1, skip_2, skip_3, pet_features))

        del x

        return None, None, seg, None