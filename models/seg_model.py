from models.model_utils import *
from models.model_utils import _make_block


# Segmentation Only
class Seg(nn.Module):
    def __init__(self, both=False, attention=False):
        super(Seg, self).__init__()

        if both:
            in_channels = 2
        else:
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

        if attention:
            self.seg = AttnSeg()
        else:
            self.seg = Segmentation()

    def forward(self, inputs):

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

        seg = self.seg.call((x, skip_1, skip_2, skip_3))

        del x

        return None, None, seg, None
