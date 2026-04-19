from model import Model
from DataHandler import *
from training import *
from plotting import *
from logger import *

BATCH_SIZE = 10

#Setup Different Models
MyModel_SGD_Sigm = Model(torch.nn.Sigmoid)
MyModel_Adam_Sigm = Model(torch.nn.Sigmoid)
MyModel_SGD_ReLU = Model(torch.nn.ReLU)
MyModel_Adam_ReLU = Model(torch.nn.ReLU)

#Training Data Setup
training_set = DataSet(40000, curve_imp)
#validation_set = DataSet(4000, curve_imp)

"""DataLoader is simply a class wrapper that returns an iterator when called.
Also handles data batching and has some other built in features"""
training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
#validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

#Loss Function
loss_func = torch.nn.BCEWithLogitsLoss()

#Setup Different Optimizers
optimizer_SGD_Sigm = torch.optim.SGD(MyModel_SGD_Sigm.parameters(), lr=0.01, momentum=0.9)
optimizer_Adam_Sigm = torch.optim.Adam(MyModel_Adam_Sigm.parameters(), lr=0.01)
optimizer_SGD_ReLU = torch.optim.SGD(MyModel_SGD_ReLU.parameters(), lr=0.01, momentum=0.9)
optimizer_Adam_ReLU = torch.optim.Adam(MyModel_Adam_ReLU.parameters(), lr=0.01)

#Training
loss_data_SGD_Sigm = train_epoch(MyModel_SGD_Sigm, optimizer_SGD_Sigm, loss_func, training_loader, BATCH_SIZE)
loss_data_Adam_Sigm = train_epoch(MyModel_Adam_Sigm, optimizer_Adam_Sigm, loss_func, training_loader, BATCH_SIZE)
loss_data_SGD_ReLU = train_epoch(MyModel_SGD_ReLU, optimizer_SGD_ReLU, loss_func, training_loader, BATCH_SIZE)
loss_data_Adam_ReLU = train_epoch(MyModel_Adam_ReLU, optimizer_Adam_ReLU, loss_func, training_loader, BATCH_SIZE)


#Saving Training Data
loss_data = {"SGD_Sigm":loss_data_SGD_Sigm, "Adam_Sigm":loss_data_Adam_Sigm, "SGD_ReLU":loss_data_SGD_ReLU, "Adam_ReLU":loss_data_Adam_ReLU}
save_loss_data(loss_data)

#Saving Model Weigths
torch.save(MyModel_SGD_Sigm.state_dict(), "data/SGD_Sigm.pth")
torch.save(MyModel_Adam_Sigm.state_dict(), "data/Adam_Sigm.pth")
torch.save(MyModel_SGD_ReLU.state_dict(), "data/SGD_ReLU.pth")
torch.save(MyModel_Adam_ReLU.state_dict(), "data/Adam_ReLU.pth")