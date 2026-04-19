import torch
import matplotlib.pyplot as plt
import numpy as np
from logger import load_loss_data
from model import Model
from DataHandler import curve_imp, curve2_imp
from training import compute_accuracy
from DataHandler import DataSet

def plot_loss(ax, loss, title):
    ax.plot([x for x in range(0, len(loss))], loss)
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Loss")
    ax.set_title(title)

#--------------------Mesh creation for model prediction generation--------------------
X, Y = np.meshgrid(np.linspace(0,10,101), np.linspace(0,10,101))

#Converting meshgrid into a matrix that fits the input vector of our model [2]
X_r = X.ravel() #Flatten grid [M x N] -> [M * N]
Y_r = Y.ravel() 

gridpoints = np.c_[X_r,Y_r] #[N*M x 2] = [Number of gridpoints x 2]

gridpoints_tensor = torch.tensor(gridpoints, dtype=torch.float32)

def get_predictions(model):
    
    model.eval()
    with torch.no_grad():
        model_out = model(gridpoints_tensor) #gets model output for every single gridpoint
        model_pred = torch.sigmoid(model_out) #converts model output to probability [0-1]
        model_pred = (model_pred > 0.5).float() #sets clamps all values to 0.0f or 1.0f

        model_pred = model_pred.numpy().reshape(X.shape) #Convert preds back to numpy and reshape to same shape as X
        model.train()
        return model_pred
    
def plot_decicion_bound(ax, model, title):
    ax.axis("equal")
    model_pred = get_predictions(model)
    ax.contourf(X,Y,model_pred, alpha = 0.3)
    ax.set_title(title)



def plot_results(curve):
    #load data
    loss_data = load_loss_data()

    MyModel_SGD_Sigm = Model(torch.nn.Sigmoid)
    MyModel_Adam_Sigm = Model(torch.nn.Sigmoid)
    MyModel_SGD_ReLU = Model(torch.nn.ReLU)
    MyModel_Adam_ReLU = Model(torch.nn.ReLU)

    MyModel_SGD_Sigm.load_state_dict(torch.load("data/SGD_Sigm.pth", weights_only=True))
    MyModel_Adam_Sigm.load_state_dict(torch.load("data/Adam_Sigm.pth", weights_only=True))
    MyModel_SGD_ReLU.load_state_dict(torch.load("data/SGD_ReLU.pth", weights_only=True))
    MyModel_Adam_ReLU.load_state_dict(torch.load("data/Adam_ReLU.pth", weights_only=True))

    MyModel_SGD_Sigm.eval()
    MyModel_Adam_Sigm.eval()
    MyModel_SGD_ReLU.eval()
    MyModel_Adam_ReLU.eval()

    plt.style.use('ggplot')

    #-----Loss Plot-----
    fig, ax = plt.subplots(2,2)

    plot_loss(ax[0,0], loss_data["SGD_Sigm"], "SGD Sigmoid")
    plot_loss(ax[0,1], loss_data["Adam_Sigm"], "Adam Sigmoid")
    plot_loss(ax[1,0], loss_data["SGD_ReLU"], "SGD ReLU")
    plot_loss(ax[1,1], loss_data["Adam_ReLU"], "Adam ReLU")

    #-----Decicion Boundary Plot-----
    fig2, ax2 = plt.subplots(2,3)

    validation_set = DataSet(4000, curve)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=10, shuffle=True)

    acc_SGD_Sigm = compute_accuracy(MyModel_SGD_Sigm, validation_loader)
    acc_Adam_Sigm = compute_accuracy(MyModel_Adam_Sigm, validation_loader)
    acc_SGD_ReLU = compute_accuracy(MyModel_SGD_ReLU, validation_loader)
    acc_Adam_ReLU = compute_accuracy(MyModel_Adam_ReLU, validation_loader)

    plot_decicion_bound(ax2[0,0], MyModel_SGD_Sigm, "Accuracy: " + str(acc_SGD_Sigm))
    plot_decicion_bound(ax2[0,1], MyModel_Adam_Sigm, "Accuracy: " + str(acc_Adam_Sigm))
    plot_decicion_bound(ax2[1,0], MyModel_SGD_ReLU, "Accuracy: " + str(acc_SGD_ReLU))
    plot_decicion_bound(ax2[1,1], MyModel_Adam_ReLU, "Accuracy: " + str(acc_Adam_ReLU))

    #plot actual function 
    ax2[0,2].contourf(X,Y,np.sign(curve(X,Y)), alpha = 0.3)
    ax2[0,2].set_title("Actual Function")
    ax2[0,2].axis("equal")
    plt.show()


if __name__ == "__main__":
    plot_results(curve2_imp)