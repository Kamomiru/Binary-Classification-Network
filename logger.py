import json
import os

LOGFILE = "data/lossData.json"

def reset_log():
    with open(LOGFILE, "w") as f:
        json.dump([], f, indent=2)

def load_loss_data():
    if not os.path.exists(LOGFILE):
        return {}

    with open(LOGFILE, "r") as f:
        loss_data = json.load(f)
        return loss_data
    
def save_loss_data(loss_data):
    with open(LOGFILE, "w") as f:
        json.dump(loss_data, f, indent=2)