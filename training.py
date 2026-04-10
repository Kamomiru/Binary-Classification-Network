import torch
import numpy as np
import matplotlib.pyplot as plt

from DataHandler import *
from model import *

BATCH_SIZE = 10

#Setup Different Models
MyModel_SGD = Model()
MyModel_Adam = Model()

training_set = MyDataSet(40000)
validation_set = MyDataSet(4000)

"""DataLoader is simply a class wrapper that returns an iterator when called.
Also handles data batching and has some other built in features"""
training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

loss_func = torch.nn.BCEWithLogitsLoss()

#Setup Different Optimizers
optimizer_SGD = torch.optim.SGD(MyModel_SGD.parameters(), lr=0.01, momentum=0.9)
optimizer_Adam = torch.optim.Adam(MyModel_Adam.parameters(), lr=0.01)


def train_epoch(model, optimizer):
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

loss_data_SGD = train_epoch(MyModel_SGD, optimizer_SGD)
loss_data_Adam = train_epoch(MyModel_Adam, optimizer_Adam)

print("Accuracy SGD = ", compute_accuracy(MyModel_SGD, validation_loader))
print("Accuracy Adam = ", compute_accuracy(MyModel_Adam, validation_loader))




#--------------------PLOTTING--------------------#
plt.style.use('ggplot')

#-----Loss Plot-----
fig, ax = plt.subplots(1,2)

ax[0].plot([x for x in range(0, len(loss_data_SGD))], loss_data_SGD)
ax[0].set_xlabel("Batch Number")
ax[0].set_ylabel("Loss")
ax[0].set_title("Stochastic Gradient Descent")

ax[1].plot([x for x in range(0, len(loss_data_Adam))], loss_data_Adam)
ax[1].set_xlabel("Batch Number")
ax[1].set_ylabel("Loss")
ax[1].set_title("Adam Optimizer")


#-----Visual Plot-----
fig2, ax2 = plt.subplots(1,3)
ax2[0].axis("equal")
ax2[1].axis("equal")
ax2[2].axis("equal")

X, Y = np.meshgrid(np.linspace(0,10,101), np.linspace(0,10,101))

#Converting meshgrid into a matrix that fits the input vector of our model [2]
X_r = X.ravel() #Flatten grid [M x N] -> [M * N]
Y_r = Y.ravel() 

gridpoints = np.c_[X_r,Y_r] #[N*M x 2] = [Number of gridpoints x 2]

gridpoints_tensor = torch.tensor(gridpoints, dtype=torch.float32)

MyModel_SGD.eval()
MyModel_Adam.eval()

with torch.no_grad():
    outputs_SGD = MyModel_SGD(gridpoints_tensor)
    preds_SGD = torch.sigmoid(outputs_SGD)
    preds_SGD = (preds_SGD > 0.5).float()

    outputs_Adam = MyModel_Adam(gridpoints_tensor)
    preds_Adam = torch.sigmoid(outputs_Adam)
    preds_Adam = (preds_Adam > 0.5).float()

Z_SGD = preds_SGD.numpy().reshape(X.shape) #Convert preds back to numpy and reshape to same shape as X
Z_Adam = preds_Adam.numpy().reshape(X.shape) #Convert preds back to numpy and reshape to same shape as X

ax2[0].contourf(X,Y,Z_SGD, alpha = 0.3)

ax2[1].contourf(X,Y,Z_Adam, alpha = 0.3)

ax2[2].contourf(X,Y,np.sign(curve_imp(X,Y)), alpha = 0.3)

plt.show()
