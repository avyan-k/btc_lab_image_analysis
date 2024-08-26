import random
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader, Subset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from datetime import datetime

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

if __name__ == "__main__":
    # load_data("vMRT",99,1000)
    tumor_type = "DDC_UC_1"
    check_for_unopenable_files(image_directory = f"./images/{tumor_type}/images",tumor_type=tumor_type)