import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import os
from tqdm import tqdm
import re


def load_device(seed: int = 99, GPU_VRAM_request = 0.75):
    set_seed(seed)
    # if a macOs then use mps
    if torch.backends.mps.is_built():
        device = torch.device("mps")
        print(f"Loading device as {device} with seed {seed}")
    # if a GPU is available, use it
    elif torch.cuda.is_available():
        device,mem_ratio = get_free_device(GPU_VRAM_request)
        print(f"Loading device as {device}, with {mem_ratio:.00%} free memory and seed {seed}")
    # revert to the default (CPU)
    else:
        device = torch.device("cpu")
        print(f"Loading device as {device} with seed {seed}")
    return device

def get_free_device(mem_request):
    device_count = torch.cuda.device_count()
    free_cuda_memories = [0.0] * device_count
    total_cuda_memories = [0.0] * device_count
    cuda_memory_ratios = [0.0] * device_count
    for dev_idx in range(device_count):
        free_cuda_memories[dev_idx], total_cuda_memories[dev_idx] = torch.cuda.mem_get_info(torch.device(f"cuda:{dev_idx}"))
        cuda_memory_ratios[dev_idx] = free_cuda_memories[dev_idx] / total_cuda_memories[dev_idx] if (total_cuda_memories[dev_idx]>0.0) else 0.0 
    max_mem = max(range(device_count),key=lambda x:cuda_memory_ratios[x]) # argmax
    if (cuda_memory_ratios[max_mem] > mem_request):
        return torch.device(f"cuda:{max_mem}"),cuda_memory_ratios[max_mem]
    else:
        raise MemoryError(f"Unable to find GPU that satisfies memory request of {mem_request:.00%}. Device with max memory is {max_mem} with total memory {free_cuda_memories[max_mem]//1024**3} GBs and free memory {free_cuda_memories[max_mem]//1024**3} GBs")


def set_seed(seed: int,GPU_deterministic = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # for cuda
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = GPU_deterministic
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
        % (torch.cuda.memory_allocated(0) / 1024 )
    )
    print(
        "torch.cuda.memory_reserved: %fGB"
        % (torch.cuda.memory_reserved(0) / 1024 )
    )
    print(
        "torch.cuda.max_memory_reserved: %fGB"
        % (torch.cuda.max_memory_reserved(0) / 1024 )
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
    get_free_device(0.5)
    load_device(0)
