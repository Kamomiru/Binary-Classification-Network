import numpy as np
from torch import utils

DIM_X = 10
DIM_Y = 10


def curve_imp(x,y) -> float:
    return (x-6)**3 + y**2 - 8

"""
def curve_imp(x,y) -> float:
    return x*3 * np.sin(y)+ y*3 * np.cos(x)
"""

def get_sample(foo): 
    x = np.random.uniform(0,DIM_X)
    y = np.random.uniform(0,DIM_Y)
    sgn = np.sign(foo(x,y)).item()

    label = -1
    if  sgn == 1:
        label = 1.0 #Loss function BCEWithLogitsLoss expects float and not int as labels
    elif sgn == -1:
        label = 0.0

    return [np.array([x,y], dtype= np.float32), label]

def get_sample_set(foo, n):
    set = []
    for _ in range(n):
        set.append(get_sample(foo))
    return set

class MyDataSet(utils.data.Dataset):
    def __init__(self, n):
        super(MyDataSet).__init__()
        self.set = get_sample_set(curve_imp, n)

    def __getitem__(self, index):
        return self.set[index]
    
    def __len__(self):
        return len(self.set)


#print(get_sample(curve_imp))

#print(get_sample_set(curve_imp, 5))

set = MyDataSet(10)


