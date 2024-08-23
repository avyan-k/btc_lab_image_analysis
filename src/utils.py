import torch
def load_device():
    
    # if a macOs then use mps
    if torch.backends.mps.is_built(): device = torch.device("mps")
    
    # if a GPU is available, use it
    elif torch.cuda.is_available(): device = torch.device("cuda")
    
    # revert to the default (CPU)
    else: device = torch.device("cpu")
        
    return device