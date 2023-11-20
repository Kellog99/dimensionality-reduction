import os
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def step(model, 
         dataloader,
         optimizer,
         criterion,
         device,
         training: bool = False):
    loss_epoch = 0.0
    if training:
        model.train()
    else:
        model.eval()
    desc = "training" if training else "validation"

    loss_rec = 0
    loss_grad = 0
    loss_momentum = 0
    for images, _ in tqdm(iter(dataloader), desc = desc):
        # blocking the gradient summation 
        optimizer.zero_grad()
    
        images = images.flatten(1).requires_grad_().to(device)
        Id = torch.eye(images.shape[1]).repeat(images.shape[0], 1, 1).to(device)
        # Forward pass
        outputs = model(images)
        jacobian_batch_comb = model.jacobian(images)
        rec = criterion(outputs, images)
        grad = criterion(jacobian_batch_comb, Id)
        momentum = model.momentum(rec = outputs, x = images)
        loss = rec + 100*grad + 50*momentum
        # Backward and optimize
        if training:
            loss.backward()
            optimizer.step()

        loss_epoch += loss.item()/len(dataloader)
        loss_momentum += momentum.item()/len(dataloader)
        loss_rec += rec.item()/len(dataloader)
        loss_grad += grad.item()/len(dataloader)
    
    
    return loss_epoch, (loss_momentum, loss_rec, loss_grad)


def train_auto(model, 
             train_loader: DataLoader, 
             val_loader: DataLoader, 
             num_epochs: int, 
             criterion,
             device,
             config: yaml, 
             optimizer):
    loss_train = []
    loss_val = []
    be = np.inf
    bm = model
    
    piece_train = {'epoch': list(range(1, num_epochs+1)), 'reconstruction':[], 'momentum':[], 'gradient':[]}
    piece_val = {'epoch': list(range(1, num_epochs+1)), 'reconstruction':[], 'momentum':[], 'gradient':[]}
    
    for epoch in range(num_epochs):
            
        # Calculate average loss for the epoch
        # loss, pieces = step(model = model, 
        #                     dataloader = train_loader, 
        #                     optimizer = optimizer, 
        #                     device = device,
        #                     criterion = criterion,
        #                     training = True)
        # loss_train.append(loss)
        # piece_train['reconstruction'].append(pieces[0])
        # piece_train['momentum'].append(pieces[1])
        # piece_train['gradient'].append(pieces[2])
        
        loss, pieces = step(model = model, 
                            dataloader = val_loader, 
                            optimizer = optimizer, 
                            device = device,
                            criterion = criterion,
                            training = False)
        piece_val['reconstruction'].append(pieces[0])
        piece_val['momentum'].append(pieces[1])
        piece_val['gradient'].append(pieces[2])
        
        loss_val.append(loss)
        
        if loss_val[-1]<be:
            print(os.path.join(config['paths']['models'], 'auto.pt'))
            be = loss_val[-1]
            torch.save(model.state_dict(), os.path.join(config['paths']['models'], 'auto.pt'))
            bm = model
            
        if epoch%1 == 0:
            # print(f"loss train epoch {epoch+1} == {loss_train[-1]}")
            print(f"loss val epoch {epoch+1} == {loss_val[-1]}")
            
    return bm, loss_train, loss_val, piece_train, piece_val
