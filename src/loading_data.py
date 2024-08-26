import random
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader, Subset

def load_data(batch_size,tumor_type,seed, sample_size):
    torch.manual_seed(seed)
    random.seed(seed)
    image_directory = f"./images/{tumor_type}/images"
    processing_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(), # Converts the image to a PyTorch tensor, which also scales the pixel values to the range [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizes the image tensor using the mean and standard deviation values that were used when training the ResNet model (usually on ImageNet)
    ])
    # get full data set 
    train_dataset = datasets.ImageFolder(image_directory,transform=processing_transforms)
    # print(train_dataset.classes,train_dataset.classes[0],train_dataset.classes[1])
    filenames = [sample[0] for sample in train_dataset.samples]
    labels = [train_dataset.classes[sample[1]] for sample in train_dataset.samples]     
    # split dataset
    indices = random.sample(range(len(train_dataset)), min(sample_size,len(train_dataset)))
    train_dataset = Subset(train_dataset, indices)
    filenames = [filenames[i] for i in indices]
    labels = [labels[i] for i in indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=True)
    print(f"Training set size: {len(train_dataset)}")
    return train_loader,filenames,labels

if __name__ == "__main__":
    load_data("vMRT",99,1000)