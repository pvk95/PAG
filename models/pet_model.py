from models.model_utils import *
from models.model_utils import _make_block


# PET only
class PetModel(nn.Module):
    def __init__(self):
        super(PetModel, self).__init__()

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

        self.pet = PET()

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

        pet = self.pet.call((x, skip_1, skip_2, skip_3))

        del x

        return None, pet, None

'''
from models.model_utils import *
from models.model_utils import _make_block


# Regularizer Only (Auto Encoder)
class PetModel(nn.Module):
    def __init__(self):
        super(PetModel, self).__init__()

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

        self.reg = _make_block(Regularizer)

    def forward(self, inputs):

        x = F.relu(self.drop(self.conv_1(inputs)))
        x = self.resnet_1(x)

        x = self.downsample_1(x)
        x = self.resnet_2(x)
        x = self.resnet_3(x)

        x = self.downsample_2(x)
        x = self.resnet_4(x)
        x = self.resnet_5(x)

        x = self.downsample_3(x)
        x = self.resnet_6(x)
        x = self.resnet_7(x)
        x = self.resnet_8(x)
        x = self.resnet_9(x)

        pet = self.reg(x)

        del x

        return None, pet, None
'''