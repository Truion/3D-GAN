from utils.read_shapes import read_off
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from utils.read_shapes import read_off, voxelization
import torch
import torch.nn as nn
from tqdm import tqdm
from network import Generator, Discriminator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#import dataloader
import torch.utils.data 
parser = argparse.ArgumentParser()
parser.add_argument('--metadata', type=str, default='data', help='data directory')
parser.add_argument('--savepath', type=str, default='./models', help='Save path for models after training')
parser.add_argument('--object_to_train_for', type=str, default='./models', help='object name to train network for')
parser.add_argument('--batch_size', type=int, default='16', help='batch size')
metadata_path = parser.parse_args().metadata
savepath = parser.parse_args().savepath
object_to_train_for = parser.parse_args().object_to_train_for
batch_size = parser.parse_args().batch_size

def load_file_paths(metadata_path, class_to_train='airplane'):
    # Load the metadata from CSV file
    df = pd.read_csv(metadata_path)

    # Initialize lists for training and testing file paths
    file_paths = []

    # Iterate through the rows and load files
    for index, row in df.iterrows():
        # object_id = row['object_id']
        # class_label = row['class']
        # split = row['split']
        object_path = row['object_path']

        if object_path.split('/')[-3] != class_to_train:
            continue
        full_path = os.path.join(os.path.dirname(metadata_path), 'ModelNet40', object_path)

        file_paths.append(full_path)

    return file_paths



# Load the first training file
class ModelNet40_Dataset(Dataset):
    def __init__(self,paths):
        self.paths = paths
        # Initialize label encoder and one-hot encoder
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')

        # Fit label encoder on the unique labels in the dataset
        labels = [file_path.split('/')[-3] for file_path in paths]
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.one_hot_encoder.fit(encoded_labels.reshape(-1, 1))
    def __len__(self):
        return len(self.paths)
    

    def __getitem__(self, idx):
        file_path = self.paths[idx]
        label = file_path.split('/')[-3]
        
        # Convert label string to one-hot encoding
        label_encoded = self.label_encoder.transform([label])
        # label_one_hot = torch.from_numpy(self.one_hot_encoder.transform([[label_encoded]]))
        label_encoded = torch.from_numpy(label_encoded).unsqueeze(1).long()
        # Load or process the file based on your specific needs
        # For example, if you're working with 3D object files (OFF format), you might need a specific loader
        # Adjust this part based on your data format and loading requirements
        
        vertices, faces= read_off(file_path)
        voxel_grid=voxelization(vertices, faces, grid_size=64)
        voxel_grid = voxel_grid.astype(np.float32)  
        voxel_grid = torch.from_numpy(voxel_grid).unsqueeze(0)

        return voxel_grid, label_encoded
    
if __name__=='__main__':
    path = load_file_paths(metadata_path, object_to_train_for)
    train_dataset = ModelNet40_Dataset(path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    latent_dim = 200
    
    generator = Generator(latent_dim)
    discriminator = Discriminator()

    # Initialize optimizers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0025, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))

    gen_loss = nn.BCELoss()
    disc_loss = nn.BCELoss()

    generator.to(device)
    discriminator.to(device)
    
    print('Training started')
    # Training loop
    epochs = 60
    for epoch in range(1, epochs+1):
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:

            for index, (voxel_grid, labels) in enumerate(train_loader):
                generator.train()
                discriminator.train()
                discriminator_optimizer.zero_grad()
                # Move tensors to device
                voxel_grid = voxel_grid.to(device)
                # labels = labels.to(device)

                # Adversarial ground truths
                valid = torch.ones((voxel_grid.size(0), 1)).to(device)
                fake = torch.zeros((voxel_grid.size(0), 1)).to(device)

                D_real_loss = disc_loss(discriminator(voxel_grid), valid)

                # Generate a batch of noise vectors
                noise_vector = torch.randn((voxel_grid.size(0), latent_dim)).to(device)

                # Generate a batch of images
                gen_voxel_grid = generator(noise_vector)

                output = discriminator(gen_voxel_grid.detach())

                D_fake_loss = disc_loss(output, fake)

                # Total discriminator loss

                D_loss = (D_real_loss + D_fake_loss) / 2

                D_loss.backward()
                discriminator_optimizer.step()

                # Train the generator
                generator_optimizer.zero_grad()

                # Loss measures generator's ability to fool the discriminator
                G_loss = gen_loss(discriminator(gen_voxel_grid), valid)

                G_loss.backward()
                generator_optimizer.step()

                pbar.set_postfix({'D_loss': D_loss.item(), 'G_loss': G_loss.item()})
                pbar.update()
        #save model
        torch.save(generator.state_dict(), os.path.join(savepath,f'generator_{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(savepath,f'discriminator_{epoch}.pth'))

