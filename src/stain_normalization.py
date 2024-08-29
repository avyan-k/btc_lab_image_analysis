import loading_data as ld
import torch
from torchvision import transforms
import torchstain
import cv2
from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image
def find_all_cases(image_directory):
    paths = [path for path in Path(image_directory).rglob('*.jpg')]
    cases = list(set([path.name.split('.')[0].split('_')[0] if len(path.name.split('.')[0].split('_')) <= 2 else path.name.split('.')[0].split('_')[0]+'_'+path.name.split('.')[0].split('_')[1] for path in paths]))
    case_dict = {case:sorted([os.path.join(path.parent,path.name[:-6]+path.name[-4:]) for path in paths if case in path.name]) for case in tqdm(cases,desc="Case",position=1,leave=False)}
    # for path in tqdm(paths):
    #     os.rename(os.path.join(path.parent,path.name),os.path.join(path.parent,path.name[:-6]+path.name[-4:]))
    return case_dict
if __name__ == "__main__":
    tumor_type = "SCCOHT_1"
    image_directory = f"./images/{tumor_type}/images"
    dictionary = find_all_cases(image_directory)
    for x in dictionary:
        image = Image.open([dictionary[x]])
        image.show()
        print(x,':',dictionary[x])    