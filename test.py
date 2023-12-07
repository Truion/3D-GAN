import torch
import os
import argparse
from network import Generator
from utils.read_shapes import visualize_voxelization
parser = argparse.ArgumentParser()
parser.add_argument('--generator_path', type=str, default='data', help='Path to generator model')
g_path = parser.parse_args().generator_path

if __name__=='__main__':
    # Load the generator model
    generator = Generator(200)
    generator.load_state_dict(torch.load(g_path))
    generator.cuda()
    generator.eval()

    latent_dim=200
    z = torch.randn(1, latent_dim).cuda()
    sample = generator(z).detach().cpu()
    sample = sample.squeeze().numpy()
    visualize_voxelization(sample, 0.5)
   