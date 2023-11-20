import os
import torch
import numpy as np
import yaml
import plotly.express as px
from matplotlib import pyplot as plt

def plot(loss_train: list, 
         loss_val: list, 
         model, 
         ds, 
         config: yaml, 
         transform: None, 
         pieces_of_loss_train:dict = None,
         pieces_of_loss_val:dict = None):
    if type(ds.targets) == list:
        targets = torch.tensor(ds.targets).unique()
    else:
        targets = ds.targets.unique()
    n_images = np.min([config['plot']['n_images'], len(targets)])
    targets = targets[:n_images]
    dataset = config['model']['dataset']             
    show_pieces = config['plot']['show_pieces']
    show_rec = config['plot']['show_rec']
    show_training = config['plot']['show_training']
    model_type = {
        1: "auto",
        2: "vae",
        3: "vae_recurrent"
    }
    model_type = model_type[config['model']['model']]
        
    l = {'epoch' :range(1, len(loss_train)+1),
         'training':loss_train, 
         'validation':loss_val}
    fig = px.line(l, 
                  x ='epoch', 
                  y=['training','validation'],
                  title = "Loss of the training",
                  width = 700, 
                  height = 600)
    if show_training:
        fig.show()
        

    y = list(pieces_of_loss_train.keys())
    y.remove('epoch')
    fig = px.line(pieces_of_loss_train, 
                  x ='epoch', 
                  y= y,
                  title = "pieces of the training loss",
                  width = 800, 
                  height = 700)
    
    fig.write_html(os.path.join(config['paths']['images'],model_type, f'piece_train_{dataset}.html'))
    if show_pieces:
        fig.show()

    fig = px.line(pieces_of_loss_val, 
                  x = 'epoch', 
                  y= y,
                  title = "pieces of the validation loss",
                  width = 800, 
                  height = 700)
    
    fig.write_html(os.path.join(config['paths']['images'],model_type, f'piece_val_{dataset}.html'))
    if show_pieces:
        fig.show()
             
    #reconstruction part

    fig, axes = plt.subplots(nrows = n_images, 
                             ncols = 2, 
                             figsize = (6, n_images*3),
                             constrained_layout=True)
    title = {
        1: "Autoencoder",
        2: "Variational autoencoder",
        3: "Vae Recurrent"
    }
    title = title[config['model']['model']]

    fig.suptitle(f"{title} {model.hidden_dim}D for {dataset}")
    with torch.no_grad():
        for i, target in enumerate(targets):
            if type(ds.targets) == list:
                data = ds.data[[True if x == i else False for x in ds.targets]][0]
            else:
                data = ds.data[ds.targets==target][0].numpy()

            data = transform(data)[0]
            data = data.unsqueeze(0) if len(data.shape) == 2 else data

            if config['model']['model'] in [2,3]: 
                recon, _, _, _,_ = model(data.float().to(model.device))
            else:
                recon = model.cpu()(data.float().flatten(1))
            axes[i,0].imshow(data[0], cmap = 'gray')
            axes[i,1].imshow(recon.detach().cpu().numpy().reshape(data.shape[1:]), cmap = 'gray')
        
            axes[i,0].set_title("real")
            axes[i,1].set_title("reconstructed")
    plt.savefig(os.path.join(config['paths']['images'],model_type, f'rec_{model.hidden_dim}D_{dataset}.png'))    
    if show_rec:
        plt.show()
    else:
        plt.close()
