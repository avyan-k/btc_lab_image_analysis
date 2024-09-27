import random
import numpy as np
import os
from pathlib import Path
import shutil
import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torchvision.models as torchmodels
from torch.utils.data import Dataset,DataLoader, Subset, random_split
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from datetime import datetime
from collections import Counter

import utils

class FeatureDataset(datasets.DatasetFolder):
    def __init__(self, datapath : str,  transform=None):
        """
        Arguments:
            datapath (string): Directory in which the class folders are located.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(
            root=datapath,
            loader= lambda path : torch.from_numpy(np.load(str(path))["arr_0"]),
            extensions= (".npz",),
            transform=transform,
        )

class TumorImageDataset(datasets.ImageFolder):
    def __init__(self, datapath : str, cases = [],transform=None):
        """
        Arguments:
            datapath (string): Directory in which the class folders are located.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(
            root=datapath,
            transform=transform,
            is_valid_file= lambda x : get_case(os.path.basename(x)) in cases
        )

def load_feature_data(batch_size,model_type,tumor_type,sample = False, sample_size = -1):
    '''
    sample: return a subset of the data, sample_size must be specified
    returns
    feature_loader: DataLoader loading features
    image_filenames: filename of image being extracted
    Classes: list of possible annotations
    '''
    feature_directory = f"./features/{model_type}/{tumor_type}"
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=get_allowed_forks())
    # valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=True)
    print(f"Training set size: {len(train_dataset)}")
    return train_loader,image_filenames, classes

def load_training_feature_data(batch_size,model_type,tumor_type, normalized = False):
    if normalized:
        tumor_type = "normalized_"+tumor_type
    feature_directory = f"./features/{model_type}/{tumor_type}"
    print(f"\nLoading data from: {feature_directory}")
    # get full data set
    full_train_dataset = FeatureDataset(feature_directory)
    train_size = len(full_train_dataset) # compute total size of dataset
    # Split the datasets into training, validation, and testing sets
    train_dataset, valid_dataset, test_dataset, _ = random_split(full_train_dataset, [int(train_size*0.8),int(train_size*0.1),int(train_size*0.1),train_size - int(train_size*0.8)-2*int(train_size*0.1)])

    train_classes = dict(sorted(Counter([full_train_dataset.targets[i] for i in train_dataset.indices]).items())) # counter return a dictionnary of the counts, sort and wrap with dict to get dict sorted by key
    valid_classes = dict(sorted(Counter([full_train_dataset.targets[i] for i in valid_dataset.indices]).items()))
    test_classes = dict(sorted(Counter([full_train_dataset.targets[i] for i in test_dataset.indices]).items()))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=get_allowed_forks())
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,num_workers=get_allowed_forks())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=get_allowed_forks())

    print(f"Training set size: {len(train_dataset)}, Class Proportions: {({full_train_dataset.classes[k]:v for k,v in train_classes.items()})}")
    print(f"Validation set size: {len(valid_loader)}, Class Proportions: {({full_train_dataset.classes[k]:v for k,v in valid_classes.items()})}")
    print(f"Test set size: {len(test_loader)}, Class Proportions: {({full_train_dataset.classes[k]:v for k,v in test_classes.items()})}")

    return (train_loader, valid_loader, test_loader), (train_classes, valid_classes, test_classes)

def load_data(batch_size,image_directory,transforms = None, sample = False, sample_size = -1):

    shutil.rmtree('./images/DDC_UC_1/images/possibly_undiff',ignore_errors=True)
    shutil.rmtree('./images/DDC_UC/images/possibly_undiff',ignore_errors=True)
    print(f"\nLoading data from: {image_directory}")

    # get full data set
    train_dataset = datasets.ImageFolder(image_directory,transform=transforms)
    total = len(train_dataset)
    # print(train_dataset.classes,train_dataset.classes[0],train_dataset.classes[1])
    filenames = [sample[0] for sample in train_dataset.samples]
    labels = [train_dataset.classes[sample[1]] for sample in train_dataset.samples]
    if sample:
        assert sample_size>0
        # split dataset
        indices = random.sample(range(len(train_dataset)), min(sample_size,len(train_dataset)))
        train_dataset = Subset(train_dataset, indices)
        filenames = [filenames[i] for i in indices] # type: ignore
        labels = [labels[i] for i in indices]
        print(f"Training set size: {len(train_dataset)} sampled out of {total}")
    else:
        print(f"Training set size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=get_allowed_forks()//2)
    # valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=True)
    return train_loader,filenames,labels

def load_training_image_data(batch_size,tumor_type, transforms = None, normalized = False):
    image_directory = f"./images/{tumor_type}/images"
    if normalized:
        image_directory = f"./images/{tumor_type}/normalized_images"
    print(f"\nLoading images from: {image_directory}")

    transforms = v2.Compose([
            v2.RandomAffine(degrees = 15,translate = (0.15,0.15)),
            transforms
    ])
    # get full data set
    # full_train_dataset = datasets.ImageFolder(image_directory,transform=transforms)
    total_size = get_size_of_dataset(image_directory,'jpg') # compute total size of dataset
    train_ratio,valid_ratio,test_ratio = 0.7,0.2,0.1
    # Split the datasets into training, validation, and testing sets
    cases = {k:len(v) for k,v in find_cases(image_directory).items()}
    label_sets = []
    for label in next(os.walk(image_directory))[1]:
        label_directory = os.path.join(image_directory,label)
        label_sets.append(set(find_cases(label_directory).keys()))
    intersection = set.intersection(*label_sets) 
    reverse_sort_intersection = (sorted(intersection,key=lambda x:cases[x]))
    '''
    We want to collect the intersection of cases for all labels (e.g. tumor, normal) 
    as we want both the test and train data to contain cases from this intersection
    the remaining data (most of this is cases that are only tumor) should be training
    we want to sort and reverse the intersection so that the first half is added to training
    '''
    train_subset,test_subset = get_case_subsets(cases,reverse_sort_intersection, total_size*(1-test_ratio)) #gives us the subsets that contain equal amounts from intersection, and then fills training data until it reaches ratio
    # cases = {k:len(v) for k,v in find_cases(image_directory).items()}
    # print([cases[case] for case in train_subset if case in intersection])
    # print([cases[case] for case in test_subset if case in intersection])
    training_valid_dataset = TumorImageDataset(datapath=image_directory,cases=train_subset,transform=transforms)
    test_dataset =  TumorImageDataset(datapath=image_directory,cases=test_subset,transform=transforms)

    train_valid_size = len(training_valid_dataset)
    train_size,valid_size = int(train_valid_size * (train_ratio/(train_ratio+valid_ratio))), int(train_valid_size * (valid_ratio/(train_ratio+valid_ratio)))

    train_dataset, valid_dataset,_ = random_split(training_valid_dataset, [train_size,valid_size,train_valid_size-train_size-valid_size])

    train_classes = dict(sorted(Counter([training_valid_dataset.targets[i] for i in train_dataset.indices]).items())) # counter return a dictionnary of the counts, sort and wrap with dict to get dict sorted by key
    valid_classes = dict(sorted(Counter([training_valid_dataset.targets[i] for i in valid_dataset.indices]).items()))
    test_classes = dict(sorted(Counter(test_dataset.targets).items()))

    # print(train_classes,valid_classes,test_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=get_allowed_forks())
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,num_workers=get_allowed_forks())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=get_allowed_forks())

    print(f"Training set size: {len(train_dataset)}, Class Proportions: {({training_valid_dataset.classes[k]:v for k,v in train_classes.items()})}")
    print(f"Validation set size: {len(valid_dataset)}, Class Proportions: {({training_valid_dataset.classes[k]:v for k,v in valid_classes.items()})}")
    print(f"Test set size: {len(test_dataset)}, Class Proportions: {({training_valid_dataset.classes[k]:v for k,v in test_classes.items()})}")

    return (train_loader, valid_loader, test_loader), (train_classes, valid_classes, test_classes)
def setup_resnet_model(seed):
    # # Defines transformations to apply on images
    processing_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(), # Converts the image to a PyTorch tensor, which also scales the pixel values to the range [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizes the image tensor using the mean and standard deviation values that were used when training the ResNet model (usually on ImageNet)
    ])
    torch.manual_seed(seed)
    model = torchmodels.resnet50(weights=torchmodels.ResNet50_Weights.DEFAULT)
    model.eval()
    return model,processing_transforms
def setup_VGG16_model(seed):
    # # Defines transformations to apply on images
    processing_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(), # Converts the image to a PyTorch tensor, which also scales the pixel values to the range [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizes the image tensor using the mean and standard deviation values that were used when training the ResNet model (usually on ImageNet)
    ])
    torch.manual_seed(seed)
    model = torchmodels.vgg16(weights=torchmodels.VGG16_Weights.DEFAULT)
    model.eval()
    return model,processing_transforms

def get_size_of_dataset(directory, extension):
    return len([path for path in Path(directory).rglob(f'*.{extension}')])

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
def get_case_subsets(case_dict, intersection, max_size):
    train_subset = []
    test_subset = []
    total = 0 #used to track if we go past max size
    for i,inter_case in enumerate(intersection):
        if total+case_dict[inter_case] >= max_size: #if we get to max value from intersection (unlikely) quit
            break
        if i>(len(intersection)//2): #for the latter half (biggest values), add to train 
            train_subset.append((inter_case,case_dict[inter_case]))
            total += case_dict.pop(inter_case)
        else: #second half, add to test
            test_subset.append((inter_case,case_dict[inter_case]))
            case_dict.pop(inter_case)
    cases = iter(reversed(sorted(list(case_dict.items()),key=lambda x: x[1]))) #turn remaining cases into interable
    while True:
        try: 
            case = next(cases)
        except StopIteration: #break immediately if no more options
            break
        value = case[1]
        if total+value >= max_size:
            test_subset = test_subset + [case] + list(cases)
            break
        train_subset.append(case)
        total += value
    return [case[0] for case in train_subset], [case[0] for case in test_subset]
def get_case(path:str) -> str:
    '''
    extracts case from image filepath, assuming "case" is everything before the last _ for the filename
    '''
    return os.path.basename(path).rsplit('_', 1)[0]
def find_cases(image_directory):
    paths = [path for path in Path(image_directory).rglob('*.jpg')]
    cases = list(set([get_case(str(path)) for path in paths]))
    case_dict = {case:sorted([os.path.join(path.parent,path.name) for path in paths if case in path.name]) for case in cases}
    return case_dict
def count_dict_tensor(count_dict:dict):
    '''
    Converts a dictionnary of the count of each class (returned by load_training_feature_data) into a 1D tensor (required for weighted cross entropy loss)
    '''
    return torch.tensor([sum(count_dict.values())/count_dict[k] for k in sorted(count_dict.keys())])
def get_allowed_forks():
    if os.name == 'nt':
        return 0
    return 8
if __name__ == "__main__":
    # load_data("vMRT",99,1000)
    # load_feature_data(100,"VMRT",99,100)

    tumor_type = "DDC_UC_1"
    image_directory = f"./images/{tumor_type}/images"
    seed = 99
    load_training_image_data(100,tumor_type, normalized=False)
    # cases = {k:len(v) for k,v in find_cases(image_directory).items()}

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