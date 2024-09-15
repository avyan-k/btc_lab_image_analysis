import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
def load_device(seed:int):
    set_seed(seed)
    # if a macOs then use mps
    if torch.backends.mps.is_built(): device = torch.device("mps")
    
    # if a GPU is available, use it
    elif torch.cuda.is_available(): device = torch.device("cuda")
    
    # revert to the default (CPU)
    else: device = torch.device("cpu")
        
    return device

def set_seed(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    # for cuda
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def get_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
def rename_dir(path, src, dst):
    # convert to list so that we can change elements
    parts = list(path.parts)
    
    # replace part that matches src with dst
    parts[parts.index(src)] = dst
    
    return Path(*parts)