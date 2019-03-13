import torch.nn as nn
import torch.nn.functional as F
import torch

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                              kernel_size, stride, padding, bias=False)
    layers.append(transpose_conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class Generator(nn.Module):

    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        self.conv_dim = conv_dim
        self.fc = nn.Linear(z_size, conv_dim * 4 * 4 * 4 * 4)
        self.t_conv0 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.t_conv1 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.t_conv2 = deconv(conv_dim * 2, conv_dim, 4)
        self.t_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        out = self.fc(x)
        out = out.view(-1, self.conv_dim * 8, 4, 4)  # (batch_size, depth, 4, 4)

        # hidden transpose conv layers + relu
        out = F.relu(self.t_conv0(out))
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))

        # last layer + tanh activation
        out = self.t_conv3(out)
        out = torch.tanh(out)
        return out
