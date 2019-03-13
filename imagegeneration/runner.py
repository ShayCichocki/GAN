import pickle as pkl
import torch.optim as optim
from imagegeneration.Discriminator import Discriminator
from imagegeneration.Generator import Generator
from imagegeneration.utils import *

train_on_gpu = torch.cuda.is_available()
assert train_on_gpu is True, "No GPU detected"
torch.cuda.empty_cache()

data_dir = '/media/scichocki/Storage/DataSets/samples/cars/train'

batch_size = 64
img_size = 128
z_size = 5000
lr = 0.001
beta1 = 0.5
beta2 = 0.999
n_epochs = 500
sample_size = 16

train_loader = get_dataloader(batch_size, img_size, data_dir)


def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)
    return D, G


D, G = build_network(img_size, img_size, z_size)

d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

# call training function
train(
    train_loader,
    D,
    G,
    d_optimizer,
    g_optimizer,
    n_epochs=n_epochs,
    z_size=z_size
)

fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()
# move z to GPU if available
fixed_z = fixed_z.cuda()
G.eval()  # for generating samples
samples = G(fixed_z)
G.train()  # back to training mode

view_samples(-1, [samples])
