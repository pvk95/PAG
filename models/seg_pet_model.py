from models.model_utils import *
from models.model_utils import _make_block
from models.grid_attention_layer import GridAttentionBlock3D


# Segmentation w/ PET
class SegPET(nn.Module):
    def __init__(self, mask=None):
        super(SegPET, self).__init__()

        self.mask = mask

        self.conv_1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

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

        self.seg = Segmentation()
        self.pet = PET()

        self.grid_layer = None
        if self.mask == 'mask_prf':
            self.grid_layer = GridAttentionBlock3D(in_channels=1, gating_channels=256, inter_channels=1)

    def forward(self, inputs):
        x = F.leaky_relu_(self.drop(self.conv_1(inputs)))
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

        seg = self.seg.call((x, skip_1, skip_2, skip_3))
        pet = self.pet.call((x, skip_1, skip_2, skip_3))

        seg_pet = None

        if self.mask is not None:
            if self.mask == 'mask_basic':
                seg_pet = seg*pet
            elif self.mask == 'mask_prf':
                pet_attn = self.grid_layer(pet, x)
                seg_pet = seg*pet_attn

        del x
        return None, pet, seg, seg_pet
