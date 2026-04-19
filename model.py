import torch

class Model(torch.nn.Module):

    def __init__(self, ActivationFunction):
        super(Model, self).__init__()
        
        self.activation = ActivationFunction()

        self.linear1 = torch.nn.Linear(2,8)
        self.linear2 = torch.nn.Linear(8,8)
        self.linear3 = torch.nn.Linear(8,4)
        self.linear4 = torch.nn.Linear(4,1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        """
        !!Loss Function torch.nn.BCEWithLogitsLoss() needs logits as output not probability between 0 and 1 to properly work
        so if we would apply the sigmoid func to the output of our network the loss would be incorrect and the model would not be abled to learn!
        #x = self.sigm(x)
        """
         
        return x
    
    def print_param(self):
        #weigths then biases...
        print('\n\nModel params:')
        for param in self.parameters():
            print(param.shape)
            print(param)
            print()
    
