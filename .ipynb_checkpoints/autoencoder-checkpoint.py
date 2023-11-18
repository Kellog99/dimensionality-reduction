import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from configparser import ConfigParser
import pickle
import argparse

from plot import plot
from training import training

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, device, hidden_dim = 2):
        super(Autoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 784)
        )

    
    def derivative(self, y, x):
        #print(y.shape)
        #print(x.shape)
        derivative = torch.autograd.grad(y, x, 
                                       grad_outputs = torch.ones_like(y),
                                       create_graph = True, 
                                       retain_graph = True)[0]
        return derivative

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

    def loss_functional(self, dec, x):
        jacobian_batch_comb = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
        
        for i in range(x.shape[0]):
            jacobian_rows = [torch.autograd.grad(dec[i], x, v, 
                                                 create_graph = True, 
                                                 retain_graph = True)[0][i] for v in torch.eye(x.shape[1], device = model.device).unbind()] 
            jacobian_rows = torch.stack(jacobian_rows)
            jacobian_batch_comb[i] = jacobian_rows

        l = torch.sum((jacobian_batch_comb-torch.eye(x.shape[1]).repeat(x.shape[0], 1, 1))**2)
        return l



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Base model")
    parser.add_argument("-c", "--config", type=str, required = True, help="Config file")

    args = parser.parse_args()
    config_path = args.config
    config = ConfigParser()
    config.read(config_path)

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_dataset = Subset(train_dataset, indices=range(50))
    test_dataset = Subset(test_dataset, indices=range(50))
    # Define the data loaders
    batch_size = config.getint('dataset', 'batch_size')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss(reduction = "sum")

    # 3d
    model = Autoencoder(hidden_dim=3, device = device).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), 
                           lr = config.getfloat('optimizer','lr'))
    print(device)
    model, loss_train, loss_val, piece_train, piece_val = training(model = model, 
                                                                train_loader = train_loader, 
                                                                val_loader = test_loader, 
                                                                num_epochs = config.getint('optimizer','epochs'),
                                                                criterion = criterion, 
                                                                device = device,
                                                                optimizer = optimizer)
    torch.save(model.state_dict(), os.path.join(config['paths']['net_weights'],'model.pt'))
    with open(os.path.join(config['paths']['net_weights'],'loss_train.pkl'), 'wb') as f :
        pickle.dump(loss_train, f)

    with open(os.path.join(config['paths']['net_weights'],'loss_val.pkl'), 'wb') as f :
        pickle.dump(loss_val, f)

    with open(os.path.join(config['paths']['net_weights'],'piece_train.pkl'), 'wb') as f :
        pickle.dump(piece_train, f)

    with open(os.path.join(config['paths']['net_weights'],'piece_val.pkl'), 'wb') as f :
        pickle.dump(piece_val, f)

    plot(loss_train, 
        loss_val, 
        model, 
        train_loader, 
        piece_train, 
        piece_val,
        device = device,  
        img_path = config['paths']['images'],
        n_images=10)
