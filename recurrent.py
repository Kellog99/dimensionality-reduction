import numpy as np
from tqdm import tqdm
import torch

def step(model, 
         dataloader,
         optimizer,
         pieces_of_loss: dict, 
         training: bool = False):
    loss_epoch = 0.0
    len_load = len(dataloader)
    if training:
        model.train()
    else:
        model.eval()
    for data, _ in tqdm(iter(dataloader)):
        # blocking the gradient summation 
        optimizer.zero_grad()
    
        # forward step
        x_reconstructed, u, x, u_hat, var  = model(data.to(model.device).float())
        
        # computing the loss
        l1, dens = model.loss_density(u, x, var)
        l2, func = model.loss_functional(data.to(model.device).flatten(1).float(), 
                                         x_reconstructed, u, x, u_hat, var)
        loss = l1 + l2 
        
        pieces = dens + func
        if torch.any(torch.isnan(loss)).item():
            print(dens)
            print(func)
        # Backward and optimize
        if training:
            loss.backward()
            optimizer.step()
        
        loss_epoch += loss.item()
        for i, key in enumerate(pieces_of_loss.keys()):
            if 'std_emb' == key:
                pieces_of_loss[key][-1] += torch.mean(torch.prod(var,1)).item()/len_load
            else:
                pieces_of_loss[key][-1] += pieces[i]/len_load
    return loss_epoch/len_load

def train_recurrent(model, 
             train_loader, 
             val_loader, 
             num_epochs,
             optimizer):
    loss_train = []
    loss_val = []
    be = np.inf
    bm = model
    
    pieces_of_loss_train = {'loss_derivative':[], 'loss_density_F':[], 'loss_density_F_inv':[], 'kl_loss':[],
                            'reconstruction1':[], 'reconstruction2':[], 'std_emb':[]}    
    pieces_of_loss_val = {'loss_derivative':[], 'loss_density_F':[], 'loss_density_F_inv':[], 'kl_loss':[],
                            'reconstruction1':[], 'reconstruction2':[],  'std_emb':[]} 
    model.train()

    for epoch in range(num_epochs):    
        for key in pieces_of_loss_train.keys():
            pieces_of_loss_train[key].append(0)
            pieces_of_loss_val[key].append(0)
        l = step(model, train_loader, optimizer, pieces_of_loss_train, True)
        loss_train.append(l)
        l = step(model, val_loader, optimizer,pieces_of_loss_val)
        loss_val.append(l)
        if (epoch+1)%5==0:
            print(f"loss training at the {epoch+1}-th = {loss_train[-1]}")
            print(f"loss validation at the {epoch+1}-th = {loss_val[-1]}")
            
        if loss_val[-1]<be:
            be = loss_val[-1]
            bm = model
    pieces_of_loss_train['epoch'] = list(range(1, num_epochs+1))
    pieces_of_loss_val['epoch'] = list(range(1, num_epochs+1))
    return bm, loss_train, loss_val, pieces_of_loss_train, pieces_of_loss_val
