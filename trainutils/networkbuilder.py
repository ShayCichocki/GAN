from networks.Discriminator import Discriminator
from networks.Generator import Generator
from trainutils.utils import *


def build_network(conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(conv_dim)
    G = Generator(z_size=z_size, conv_dim=conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)
    return D, G
