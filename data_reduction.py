import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE, Autoencoder, VAE_recurrent
import yaml
from vae import train_vae
from autoencoder import train_auto
from recurrent import train_recurrent
from plot import plot
import pickle

if __name__ == '__main__':

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = config['dataset']['batch_size']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Grayscale(),
                                    transforms.Normalize((0), (0.3))])
    for data in config['dataset']['datasets']:
        print(data)
        config['model']["dataset"] = data 

        train_dataset = getattr(datasets, data)(root='./data', train=True, download=True, transform= transform)
        test_dataset = getattr(datasets, data)(root='./data', train=False, download=True, transform= transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        input_feat = len(train_dataset[0][0].flatten(0))

        criterium = nn.MSELoss(reduction = "mean")
        
        if config['model']['model'] == 1:
            model = Autoencoder(input_dim= input_feat, 
                                hidden_dim = config['model']['hidden_dimension'], 
                                device = device).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3) 

            # Training the vae
            model, loss_train, loss_val, piece_train, piece_val = train_auto(model = model, 
                                                                            train_loader = train_loader, 
                                                                            val_loader = test_loader, 
                                                                            num_epochs = config['optimizer']['epochs'],
                                                                            criterion = criterium, 
                                                                            device = device,
                                                                            config = config,
                                                                            optimizer = optimizer)
        elif config['model']['model'] == 2:
            print("model VAE")
            model = VAE(input_feat= input_feat, 
                        hidden_dim = config['model']['hidden_dimension'], 
                        device = device, 
                        criterium = criterium).to(device)
            optimizer = optim.Adam(model.parameters(), 
                                   lr = 1e-4) 

        # Training the vae
            model, loss_train, loss_val, piece_train, piece_val = train_vae(model = model, 
                                                                            train_loader = train_loader, 
                                                                            val_loader = test_loader, 
                                                                            num_epochs = config['optimizer']['epochs'], 
                                                                            optimizer = optimizer)
        
        elif config['model']['model'] == 3:
            print("recurrent model")
            model = VAE_recurrent(input_feat= input_feat, 
                                  hidden_dim = config['model']['hidden_dimension'], 
                                  device = device,
                                  criterium = criterium).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3) 

            # Training the vae
            model, loss_train, loss_val, piece_train, piece_val = train_recurrent(model = model, 
                                                                                  train_loader = train_loader, 
                                                                                  val_loader = test_loader, 
                                                                                  num_epochs = config['optimizer']['epochs'],
                                                                                  optimizer = optimizer)


        plot(loss_val, 
            loss_val, 
            model, 
            ds = train_dataset, 
            config=config,
            transform= transform ,
            pieces_of_loss_train = piece_val, 
            pieces_of_loss_val = piece_val)

        model_type = {
                1: "auto",
                2: "vae",
                3: "vae_recurrent"
            }
        model_type = model_type[config['model']['model']]
        with open(os.path.join(config['paths']['net_weights'],f'loss_train_{model_type}_{model.hidden_dim}D.pkl'), 'wb') as f :
            pickle.dump(loss_train, f)

        with open(os.path.join(config['paths']['net_weights'],f'loss_val_{model_type}_{model.hidden_dim}D.pkl'), 'wb') as f :
            pickle.dump(loss_val, f)

        with open(os.path.join(config['paths']['net_weights'],f'piece_train_{model_type}_{model.hidden_dim}D.pkl'), 'wb') as f :
            pickle.dump(piece_train, f)

        with open(os.path.join(config['paths']['net_weights'],f'piece_val_{model_type}_{model.hidden_dim}D.pkl'), 'wb') as f :
            pickle.dump(piece_val, f)