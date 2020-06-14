import torch
from torch import nn
from torch.nn import functional as F


class _MaskND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation'):
        super(_MaskND, self).__init__()

        assert dimension == 3
        assert mode == 'concatenation'

        if in_channels != 1:
            raise Warning(f"Not expected in_channels as {in_channels}")
        if gating_channels != 1:
            raise Warning(f"Not expected gating channels as {gating_channels}")

        # Default parameter set
        self.mode = mode
        self.dimension = dimension

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv3d

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Define the operation
        self.operation_function = self._concatenation

    def forward(self, x, g):
        """
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        """

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)

        phi_g = self.phi(g)
        f = F.leaky_relu_(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        # sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.leaky_relu_(self.psi(f))

        # upsample the attentions and multiply
        y = torch.sigmoid(sigm_psi_f * x)

        return y


class Mask3D(_MaskND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation'):
        super(Mask3D, self).__init__(in_channels=in_channels,
                                     inter_channels=inter_channels,
                                     gating_channels=gating_channels,
                                     dimension=3, mode=mode)


if __name__ == '__main__':

    mode_list = ['concatenation']

    for mode in mode_list:
        img = torch.rand(2, 1, 10, 10, 10)
        gat = torch.rand(2, 1, 10, 10, 10)
        net = Mask3D(in_channels=1, inter_channels=16, gating_channels=1, mode=mode)
        out = net(img, gat)
        print(out.size())
