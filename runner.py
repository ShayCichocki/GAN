import torch.optim as optim
from dotenv import load_dotenv
from trainutils.networkbuilder import build_network
from trainutils.utils import *
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

train_on_gpu = torch.cuda.is_available()
assert train_on_gpu is True, "No GPU detected"
torch.cuda.empty_cache()


def view_results(G, sample_size, z_size):
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    fixed_z = fixed_z.cuda()
    G.eval()  # for generating samples
    samples = G(fixed_z)
    G.train()  # back to training mode
    view_samples(-1, [samples])


def get_config():
    load_dotenv()
    return {
        "data_dir": os.getenv("DATA_DIR"),
        "batch_size": int(os.getenv("BATCH_SIZE")),
        "img_size": int(os.getenv("IMAGE_SIZE")),
        "z_size": int(os.getenv("Z_SIZE")),
        "lr": float(os.getenv("LR")),
        "beta1": float(os.getenv("BETA1")),
        "beta2": float(os.getenv("BETA2")),
        "n_epochs": int(os.getenv("N_EPOCHS")),
        "sample_size": int(os.getenv("SAMPLE_SIZE"))
    }


if __name__ == '__main__':
    config = get_config()
    train_loader = get_dataloader(config['batch_size'], config['img_size'], config['data_dir'])
    D, G = build_network(config['img_size'], config['z_size'])
    d_optimizer = optim.Adam(D.parameters(), config['lr'], [config['beta1'], config['beta2']])
    g_optimizer = optim.Adam(G.parameters(), config['lr'], [config['beta1'], config['beta2']])

    # call training function
    D, G = train(
        train_loader,
        D,
        G,
        d_optimizer,
        g_optimizer,
        n_epochs=config['n_epochs'],
        z_size=config['z_size']
    )

    view_results(G, config['sample_size'], config['z_size'])
