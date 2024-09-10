import random
import numpy as np
import os
from pathlib import Path
import shutil
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset,DataLoader, Subset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from datetime import datetime
import utils

class FeatureDataset(Dataset):
    def __init__(self, datapath : str,  transform=None):
        """
        Arguments:
            datapath (string): Directory in which the class folders are located.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.paths = list(Path(datapath).glob("*/*.npz")) # loads all possible numpy arrays in directory as filepaths
        self.labels = [path.parent.name for path in self.paths]
        self.transform = transform
        self.classes =  sorted(entry.name for entry in os.scandir(datapath) if entry.is_dir()) # finds all possible classes and sorts them
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)} # converts a class to its index, used in __getitem__

    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index : int) -> tuple[torch.Tensor, int]:
        array = np.load(str(self.paths[index]))["arr_0"] # load array from filepath, note that since no arg is provided when saving, the first array is arr_0
        tensor = torch.from_numpy(array) # convert to tensor
        class_name  = self.paths[index].parent.name # since we use pathlib.Path, we can call its parent for the class
        cindex = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(tensor).float(), cindex
        else:
            return tensor.float(), cindex


def load_feature_data(batch_size,tumor_type,sample = False, sample_size = -1):
    '''
    sample: return a subset of the data, sample_size must be specified
    returns
    feature_loader: DataLoader loading features
    image_filenames: filename of image being extracted
    Classes: list of possible annotations
    '''
    feature_directory = f"./features/{tumor_type}"
    image_directory = f"./images/{tumor_type}/images"
    print(f"\nLoading data from: {feature_directory}\nImages Filenames from {image_directory}")
    # get full data set 
    train_dataset = FeatureDataset(feature_directory)
    classes = train_dataset.classes
    # print(train_dataset.classes,train_dataset.classes[0],train_dataset.classes[1])
    image_filenames = [path for path in Path(image_directory).rglob('*.jpg')]
    # print(len(image_filenames),len(train_dataset.paths))
    # labels = train_dataset.labels
    if sample:
        assert sample_size>0
        # split dataset
        indices = random.sample(range(len(train_dataset)), min(sample_size,len(train_dataset)))
        train_dataset = Subset(train_dataset, indices)
        image_filenames = [image_filenames[i] for i in indices]
        # labels = [labels[i] for i in indices]
    # print(filenames,labels,sep='\n')  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=True)
    print(f"Training set size: {len(train_dataset)}")
    return train_loader,image_filenames, classes
def setup_resnet_model(seed):
    # # Defines transformations to apply on images
    processing_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(), # Converts the image to a PyTorch tensor, which also scales the pixel values to the range [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizes the image tensor using the mean and standard deviation values that were used when training the ResNet model (usually on ImageNet)
    ])
    torch.manual_seed(seed)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    return model,processing_transforms
def load_data(batch_size,tumor_type,transforms, sample = False, sample_size = -1):

    shutil.rmtree('./images/DDC_UC_1/images/possibly_undiff',ignore_errors=True)
    shutil.rmtree('./images/DDC_UC/images/possibly_undiff',ignore_errors=True)
    image_directory = f"./images/{tumor_type}/images"
    print(f"\nLoading data from: {image_directory}")

    # get full data set 
    train_dataset = datasets.ImageFolder(image_directory,transform=transforms)
    # print(train_dataset.classes,train_dataset.classes[0],train_dataset.classes[1])
    filenames = [sample[0] for sample in train_dataset.samples]
    labels = [train_dataset.classes[sample[1]] for sample in train_dataset.samples]     
    if sample:
        assert sample_size>0
        # split dataset
        indices = random.sample(range(len(train_dataset)), min(sample_size,len(train_dataset)))
        train_dataset = Subset(train_dataset, indices)
        image_filenames = [image_filenames[i] for i in indices] # type: ignore
        labels = [labels[i] for i in indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=True)
    print(f"Training set size: {len(train_dataset)}")
    return train_loader,filenames,labels
def check_for_unopenable_files(image_directory,tumor_type):
    with open(file=f"./results/{tumor_type}_corrupted_files.txt",mode='w') as f:
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Checked on {time}\n")
        images_paths = [path for path in Path(image_directory).rglob('*.jpg')]
        for image_path in tqdm(images_paths):
            try:
                Image.open(image_path)
            except(UnidentifiedImageError):
                f.write(str(image_path))
def get_case(path:str) -> str:
    '''
    extracts case from image filepath, assuming "case" is everything before the last _ for the filename
    '''
    return os.path.basename(path).rsplit('_', 1)[0]
if __name__ == "__main__":
    # load_data("vMRT",99,1000)
    # load_feature_data(100,"VMRT",99,100)
    
    tumor_type = "vMRT"
    seed = 99
    # Image.open(r'images\DDC_UC_1/images/undiff\AS15041526_227753du.jpg')
    # DEVICE = utils.load_device(seed)
    # size_of_image_dataset = len([path for path in Path(f"./images/{tumor_type}/images").rglob('*.jpg')])
    # size_of_feature_dataset = len([path for path in Path(f"./features/{tumor_type}").rglob('*.npz')])
    # model = setup_resnet_model(seed).to(DEVICE)
    # image_loader,image_filepaths,image_labels = load_data(1,tumor_type,seed,size_of_image_dataset)
    # feature_loader,feature_filepaths,feature_labels = load_feature_data(1,tumor_type,seed,size_of_image_dataset)
    # image_iterator = iter(image_loader)
    # feature_iterator = iter(feature_loader)
    # index = 0
    # total_error = 0
    # while True:
    #     if index == size_of_image_dataset:
    #         break
    #     with torch.no_grad():
    #         images = next(image_iterator)[0]
    #         saved_features = next(feature_iterator)[0]
    #         images = images.to(DEVICE)
    #         features = model(images)
            
    #         x = saved_features
    #         y = features.cpu()
    #         # print('y:',image_filepaths[index],y[0].shape)
    #         # print('x:',feature_filepaths[index],x[0].shape)
    #         print(index, torch.linalg.vector_norm(x[0]-y[0]))
    #         total_error += torch.linalg.vector_norm(x[0]-y[0])
    #         # for i in range(len(x[0])):
    #         #     print(x[0][i].item(),y[0][i].item())
    #     index += 1
    # print(total_error)
    # print(image_labels==feature_labels)

    # check_for_unopenable_files(image_directory = f"./images/{tumor_type}/images",tumor_type=tumor_type)
    # x = Image.open(f"./images/DDC_UC_1/images/undiff/AS15041526_227753du.jpg")