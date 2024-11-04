import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import os
from tqdm import tqdm
import re
from cv2 import imread, imwrite

def load_device(seed: int = 99):
    set_seed(seed)
    # if a macOs then use mps
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    # if a GPU is available, use it
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    # revert to the default (CPU)
    else:
        device = torch.device("cpu")
    print(f"Loading device as {device}")
    return device


def set_seed(seed: int):
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
    if (
        type(src) is int
    ):  # the index may be given directly if path contain same name directory
        index = src
    else:
        index = parts.index(src)
    parts[index] = dst

    return Path(*parts)


def print_cuda_memory():
    print(
        "torch.cuda.memory_allocated: %fGB"
        % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.memory_reserved: %fGB"
        % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.max_memory_reserved: %fGB"
        % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
    )


def remove_image_label_abbreviations():
    regex = re.compile(r"([0-Z_ ]*)[a-z]{2}.jpg")
    for tumor_type in os.listdir("./images"):
        print(tumor_type)
        paths = [path for path in Path(f"./images/{tumor_type}").rglob("*.jpg")]
        for path in tqdm(paths, leave=False, desc="Paths"):
            basename = os.path.basename(path)
            dirname = os.path.dirname(path)
            rmatch = regex.match(str(basename))
            if rmatch is not None:
                newpath = os.path.join(dirname, str(rmatch.groups()[0]) + ".jpg")
                os.rename(path, newpath)

    return
if __name__ == "__main__":
    load_device(0)