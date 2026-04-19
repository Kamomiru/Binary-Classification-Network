import torch
import numpy as np

from DataHandler import *
from model import *

def train_epoch(model, optimizer, loss_func, training_loader, BATCH_SIZE):
    running_loss = 0
    last_loss = 0
    loss_data = []
    

    for i, data in enumerate(training_loader):
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        outputs = outputs.squeeze()      # output is of size [10,1] but labels are of size[10]. Squeeze collapses dimensions with singular entries
        loss = loss_func(outputs, labels)
        
        #backpropagation
        loss.backward() #stores gradient in param.grad

        # Adjust learning weights
        optimizer.step() #reads param.grad

        # Gather data and report
        running_loss += loss.item()
        if i % BATCH_SIZE == BATCH_SIZE-1:
            last_loss = running_loss / BATCH_SIZE # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            loss_data.append(last_loss)
            running_loss = 0.
        
    return loss_data

def compute_accuracy(model, validation_loader, device="cpu"):
    model.eval()  # evaluation mode
    
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze()   #squeeze() [10,1] -> [10]

            # convert logits → predictions (0 or 1)
            preds = (torch.sigmoid(outputs) > 0.5) #creates a tuple with all predictions that are greater 0.5

            """(preds == labels) -> creates tuple of bools
            we can sum a tuple of bools to get the number of True entries
            the number will still be in a tensor so we get that single entry with .item()"""
            correct += (preds == labels).sum().item()
            total += labels.numel() #add number of elements to total
        
    model.train()  # back to training mode

    return correct / total





