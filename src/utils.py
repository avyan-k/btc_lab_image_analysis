import torch
import random
import numpy as np
def load_device(seed):
    set_seed(seed)
    # if a macOs then use mps
    if torch.backends.mps.is_built(): device = torch.device("mps")
    
    # if a GPU is available, use it
    elif torch.cuda.is_available(): device = torch.device("cuda")
    
    # revert to the default (CPU)
    else: device = torch.device("cpu")
        
    return device

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    # for cuda
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False