import random
import numpy as np
import os
from pathlib import Path
import torch
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split, sampler as torch_sampler
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from cv2 import imread, imwrite
import time

import utils


class FeatureDataset(datasets.DatasetFolder):
    def __init__(self, datapath: str, transform=None):
        """
        Arguments:
            datapath (string): Directory in which the class folders are located.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(
            root=datapath,
            loader=lambda path: torch.from_numpy(np.load(str(path))["arr_0"]),
            extensions=(".npz",),
            transform=transform,
        )


class TumorImageDataset(datasets.ImageFolder):
    def __init__(self, root: str, transform=None,target_transform=None,cases=None):
        """
        Arguments:
            datapath (string): Directory in which the class folders are located.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if cases is not None:
            super().__init__(
                root=root,
                transform=transform,
                target_transform=target_transform,
                is_valid_file=lambda x: get_case(os.path.basename(x)) in cases,
            )
        else:
            super().__init__(
                root=root,
                transform=transform,
                target_transform=target_transform,
            )


class BalancedTumorImageData(TumorImageDataset):
    def __init__(self, datapath: str, k: int, transform=None, target_transform=None, cases=None):
        """
        k represents the number to sample from each class
        """
        super().__init__(
            root=datapath, transform=transform, target_transform=target_transform, cases=cases
        )
        self.k = k
        self.balanced_indices = self._get_balanced_indices()
        samples_array = np.array(self.samples)
        self.samples = [
            (str(sample), int(target))
            for sample, target in samples_array[self.balanced_indices]
        ]
        self.targets = [int(s[1]) for s in self.samples]

    def _get_balanced_indices(self):
        balanced_class_indices = []
        for class_label in range(len(self.classes)):
            all_indices = list(
                np.where(np.array(self.targets) == class_label)[0]
            )  # find all instance of class, 0 being success
            balanced_class_indices.extend(
                random.sample(all_indices, min(len(all_indices) - 1, self.k))
            )

        return balanced_class_indices

def get_image_dataset(
        tumor_type,
        samples_per_class = -1, 
        stain_normalized=False, 
        proven_mutation=False, 
        processing_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]),
        verbose=False
        ):
    '''
    Get entire image dataset or a balanced subset (if samples_per_class is set to a positive number). 
    If stain_normalized is set to true, the images are loaded from the normalized_images directory.
    '''
    image_directory = get_image_directory(tumor_type,stain_normalized=stain_normalized)

    if verbose:
        print(f"\nLoading images from: {image_directory}")


    if samples_per_class == -1:
        if proven_mutation:
            proven_mutation = get_proven_mutation_cases(tumor_type)
            return TumorImageDataset(root=image_directory, transform=processing_transforms, cases=proven_mutation)
        return TumorImageDataset(root=image_directory, transform=processing_transforms)
    else:
        if proven_mutation:
            proven_mutation = get_proven_mutation_cases(tumor_type)
            return BalancedTumorImageData(
                image_directory, k=samples_per_class, transform=processing_transforms, cases=proven_mutation
            )
        return BalancedTumorImageData(
            image_directory, k=samples_per_class, transform=processing_transforms
        )

def split_datasets(dataset,test_ratio,validation=False,validation_ratio=None,verbose=False,generator:torch.Generator|None=None):
    train_size = len(dataset)
    # Split the datasets into training, validation, and testing sets
    if validation:
        assert validation_ratio is not None and test_ratio+validation_ratio<1.0
        train_ratio = 1.0-test_ratio-validation_ratio
        ratios = [
                int(train_size * train_ratio),
                int(train_size * validation_ratio),
                int(train_size * test_ratio),
                train_size
                - int(train_size * train_ratio)
                - int(train_size * validation_ratio)
                - int(train_size * test_ratio),
            ]
        if generator:
            train_dataset, valid_dataset, test_dataset, _ = random_split(dataset,ratios,generator)
        else:
            train_dataset, valid_dataset, test_dataset, _ = random_split(dataset,ratios)
    else:
        assert test_ratio < 1.0
        train_ratio = 1.0-test_ratio
        ratios = [
                int(train_size * train_ratio),
                int(train_size * test_ratio),
                train_size - int(train_size * train_ratio) - int(train_size * test_ratio),
            ]
        if generator:
            train_dataset, test_dataset, _ = random_split(dataset,ratios,generator)
        else:
            train_dataset, test_dataset, _ = random_split(dataset,ratios)
        valid_dataset = None

    all_classes = dict(sorted(Counter(dataset.targets).items(),key=lambda x:x[1]))
    train_classes = dict(
        sorted(
            Counter([dataset.targets[i] for i in train_dataset.indices]).items()
        )
    )  # counter return a dictionnary of the counts, sort and wrap with dict to get dict sorted by key
    test_classes = dict(
        sorted(Counter([dataset.targets[i] for i in test_dataset.indices]).items())
    )
    valid_classes = None
    if valid_dataset is not None:
        valid_classes = dict(
            sorted(
                Counter(
                    [dataset.targets[i] for i in valid_dataset.indices]
                ).items()
            )
        )

    if verbose:
        print(
            f"Training set size: {len(train_dataset)}, Class Proportions: {({dataset.classes[k]:v for k,v in train_classes.items()})}"
        )

        print(
            f"Test set size: {len(test_dataset)}, Class Proportions: {({dataset.classes[k]:v for k,v in test_classes.items()})}"
        )
        if valid_classes is not None and valid_dataset is not None:
            print(
                f"Validation set size: {len(valid_dataset)}, Class Proportions: {({dataset.classes[k]:v for k,v in valid_classes.items()})}"
            )
    if validation:
        return train_dataset,test_dataset,valid_dataset,train_classes,test_classes,valid_classes,all_classes
    return train_dataset,test_dataset,train_classes,test_classes,all_classes

def load_training_image_data(
        batch_size,
        tumor_type,
        seed,
        samples_per_class=-1,
        normalized=True,
        proven_mutation_only=False,
        test_ratio = 0.2,
        validation=False,
        valid_ratio = None
    ):
    '''
    External wrapper for  get_loaders_training_image_data to allow any dataset to be loaded
    '''
    full_dataset = get_image_dataset(tumor_type=tumor_type,samples_per_class=samples_per_class, proven_mutation=proven_mutation_only,verbose=True)
    split_generator = torch.Generator().manual_seed(seed)
    
    if normalized:
        unnorm_train_dataset,_,_,_,_ = split_datasets(full_dataset,test_ratio,validation,valid_ratio,True,split_generator) # type: ignore
        dataset_info = get_dataset_info(tumor_type,False,False,proven_mutation_only,test_ratio,valid_ratio) 
        norm = get_norm_transform(unnorm_train_dataset,seed,dataset_info)
        full_dataset = get_image_dataset(tumor_type,samples_per_class,False ,proven_mutation_only,norm)
    
    if validation:
        train_set,test_set,valid_set,train_class,test_class,valid_class,all_class = split_datasets(full_dataset,test_ratio,True,valid_ratio,False,split_generator) # type: ignore
        assert isinstance(valid_set,datasets.DatasetFolder|torch.utils.data.Subset)
        return get_loaders_training_image_data(train_set,test_set,batch_size,True,train_class,test_class,{k:v for k,v in enumerate(full_dataset.classes)},valid_set,valid_class), (train_class,valid_class,test_class)
    else:
        train_set,test_set,train_class,test_class,_ = split_datasets(full_dataset,test_ratio,False,None,False,split_generator) # type: ignore
        assert type(train_class) == dict # since multiple return types are possible
        return get_loaders_training_image_data(train_set,test_set,batch_size,True,train_class,test_class,{k:v for k,v in enumerate(full_dataset.classes)}), (train_class,test_class) 


def get_loaders_training_image_data(
    train_dataset:datasets.DatasetFolder|torch.utils.data.Subset,
    test_dataset:datasets.DatasetFolder|torch.utils.data.Subset,
    batch_size:int,
    weighted:bool,
    train_classes:dict[str,int],
    test_classes:dict[str,int],
    label2class:dict[int,str],
    valid_dataset:torch.utils.data.Subset|None = None,
    valid_classes:dict[str,int]|None = None,
):
    if weighted:
        train_sampler = create_weighted_sampler(train_dataset,train_classes,max(train_classes.values()),True,label2class)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=get_allowed_forks(),
        )
        test_sampler = create_weighted_sampler(test_dataset,test_classes,max(test_classes.values()),True,label2class)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=get_allowed_forks(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=get_allowed_forks(),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=get_allowed_forks(),
        )
    if valid_dataset:
        assert valid_classes is not None
        if weighted:
            valid_sampler = create_weighted_sampler(valid_dataset,valid_classes,max(valid_classes.values()))
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                sampler=valid_sampler,
                num_workers=get_allowed_forks(),
            )
        else:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=get_allowed_forks(),
            )
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, test_loader
def load_image_data(
    batch_size,     
    tumor_type,
    seed,
    normalized,
    stain_normalize,
    samples_per_class=-1,
):
    '''
    Same as load_training_image_data but a single loader is returned with the entire dataset, along with a list of class names
    '''
    full_dataset = get_image_dataset(tumor_type=tumor_type,samples_per_class=samples_per_class,stain_normalized=stain_normalize,verbose=True)
    dataset_info = get_dataset_info(tumor_type,False,stain_normalize,False)
    if normalized:
        norm = get_norm_transform(full_dataset,seed,dataset_info)
        full_dataset = get_image_dataset(tumor_type,samples_per_class,False,False,norm)
    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_allowed_forks(),
    )

    classes = dict(
        sorted(
            Counter(full_dataset.targets).items()
        )
    )  # counter return a dictionnary of the counts, sort and wrap with dict to get dict sorted by key

    print(
        f"Full dataset size: {len(full_dataset)}, Class Proportions: {({full_dataset.classes[k]:v for k,v in classes.items()})}"
    )
    return loader, full_dataset.classes

def load_training_image_data_by_case(
    batch_size, tumor_type, transforms=None, normalized=False
):
    '''
    Same as load_training_image_data but the algorithm will attempt to balance the dataset by case such that each class has the same number of cases
    '''
    image_directory = get_image_directory(tumor_type,stain_normalized=normalized)
    print(f"\nLoading images from: {image_directory}")

    transforms = v2.Compose(
        [v2.RandomAffine(degrees=(-15,15), translate=(0.15, 0.15)), transforms]
    ) if transforms is not None else v2.RandomAffine(degrees=(-15,15), translate=(0.15, 0.15))

    total_size = get_size_of_dataset(
        image_directory, "jpg"
    )  # compute total size of dataset
    train_ratio, valid_ratio, test_ratio = 0.7, 0.2, 0.1
    # Split the datasets into training, validation, and testing sets
    train_subset, test_subset = get_balanced_train_test_cases(total_size, test_ratio, image_directory)
    # cases = {k:len(v) for k,v in find_cases(image_directory).items()}
    # print([cases[case] for case in train_subset if case in intersection])
    # print([cases[case] for case in test_subset if case in intersection])
    training_valid_dataset = TumorImageDataset(
        root=image_directory, cases=train_subset, transform=transforms
    )
    test_dataset = TumorImageDataset(
        root=image_directory, cases=test_subset, transform=transforms
    )

    train_valid_size = len(training_valid_dataset)
    train_size, valid_size = (
        int(train_valid_size * (train_ratio / (train_ratio + valid_ratio))),
        int(train_valid_size * (valid_ratio / (train_ratio + valid_ratio))),
    )

    train_dataset, valid_dataset, _ = random_split(
        training_valid_dataset,
        [train_size, valid_size, train_valid_size - train_size - valid_size],
    )

    train_classes = dict(
        sorted(
            Counter(
                [training_valid_dataset.targets[i] for i in train_dataset.indices]
            ).items()
        )
    )  # counter return a dictionnary of the counts, sort and wrap with dict to get dict sorted by key
    valid_classes = dict(
        sorted(
            Counter(
                [training_valid_dataset.targets[i] for i in valid_dataset.indices]
            ).items()
        )
    )
    test_classes = dict(sorted(Counter(test_dataset.targets).items()))

    # print(train_classes,valid_classes,test_classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=get_allowed_forks(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=get_allowed_forks(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=get_allowed_forks(),
    )
    print(training_valid_dataset.classes)
    print(
        f"Training set size: {len(train_dataset)}, Class Proportions: {({training_valid_dataset.classes[k]:v for k,v in train_classes.items()})}"
    )
    print(
        f"Validation set size: {len(valid_dataset)}, Class Proportions: {({training_valid_dataset.classes[k]:v for k,v in valid_classes.items()})}"
    )
    print(
        f"Test set size: {len(test_dataset)}, Class Proportions: {({training_valid_dataset.classes[k]:v for k,v in test_classes.items()})}"
    )

    return (train_loader, valid_loader, test_loader), (
        train_classes,
        valid_classes,
        test_classes,
    )


def get_size_of_dataset(directory, extension):
    return len([path for path in Path(directory).rglob(f"*.{extension}")])


def get_annotation_classes(tumor_type):
    return [
        name
        for name in os.listdir(get_image_directory(tumor_type,stain_normalized=False))
        if name not in [".DS_Store", "__MACOSX"]
    ]

def get_image_directory(tumor_type,stain_normalized):
    image_directory = f"./images/{tumor_type}/images" if not stain_normalized else f"./images/{tumor_type}/normalized_images"
    if not os.path.isdir(image_directory):
        raise FileNotFoundError(f"Unable to find image folder {image_directory}")
    return image_directory

def get_dataset_info(tumor_type,normalized,stain_normalized,proven_mutation,test_ratio=None,valid_ratio=None):
    dataset_info = tumor_type
    if test_ratio:
        dataset_info += f"-test={test_ratio}"
    if valid_ratio:
        dataset_info += f"-valid={valid_ratio}"
    if normalized:
        dataset_info += "_normalized"
    if stain_normalized:
        dataset_info += "_stain-normalized"
    if proven_mutation:
        dataset_info += "_proven-mutation-only"
    return dataset_info

def create_weighted_sampler(dataset:datasets.DatasetFolder|torch.utils.data.Subset,class_dict,samples_per_class,verbose=False,label2class = None):
    class_weights = list(class_dict.values())
    if samples_per_class == -1:
        samples_per_class = len(dataset) // len(class_weights)
    if verbose:
        assert label2class is not None
        for tumor_class,class_weight in class_dict.items():
            if class_weight<samples_per_class:
                print(f"Found {class_weight} samples for class {label2class[tumor_class]}, less than the requested {samples_per_class}. It will be upsampled (resampled with replacement) at a rate of {1-round(float(class_weight/sum(class_weights)),3)}")
            elif class_weight>samples_per_class:
                print(f"Found {class_weight}  samples for class {label2class[tumor_class]}, more than the requested {samples_per_class}. It will be downsampled (some samples will be skipped) during training")
            else:
                print(f"Found {class_weight} samples for class {label2class[tumor_class]}, as many as the requested {samples_per_class}. All samples will be equally used.")
    return torch_sampler.WeightedRandomSampler(class_weights, samples_per_class*len(class_weights))   

def check_for_unopenable_files(tumor_type, norm=False):
    image_directory = get_image_directory(tumor_type,norm)
    
    with open(file=f"./results/{tumor_type}_corrupted_files.txt", mode="w") as f:
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Checked on {time}\n")
        images_paths = [path for path in Path(image_directory).rglob("*.jpg")]
        for image_path in tqdm(images_paths):
            try:
                Image.open(image_path)
                cv2image = imread(str(image_path))
                if cv2image is None:
                    raise UnidentifiedImageError
            except UnidentifiedImageError:
                f.write(str(image_path))


def split_all_images(tumor_type, norm=False):
    """
    Splits all 512 x 512 image into 4 256 x 256 tiles, saves them and deletes original
    """
    image_directory = get_image_directory(tumor_type,norm)

    for annotation in os.listdir(image_directory):
        if annotation in [".DS_Store", "__MACOSX"]:
            continue
        for image_path in tqdm(os.listdir(os.path.join(image_directory, annotation))):
            if "tile" in image_path:
                continue
            image_full_path = os.path.join(image_directory, annotation, image_path)
            image = imread(image_full_path)
            if image.shape == (512, 512, 3):
                for idx, tile in enumerate(
                    [
                        image[x : x + 256, y : y + 256]
                        for x in range(0, 512, 256)
                        for y in range(0, 512, 256)
                    ]
                ):
                    # print(os.path.splitext(image_full_path)[0]+f"_tile_{idx+1}"+os.path.splitext(image_full_path)[1])
                    imwrite(
                        os.path.splitext(image_full_path)[0]
                        + f"_tile_{idx+1}"
                        + os.path.splitext(image_full_path)[1],
                        tile,
                    )
                os.remove(image_full_path)


def get_proven_mutation_cases(tumor_type):    
    proven_mutations = []
    with open(f"./images/{tumor_type}/proven_mutation.txt") as f:
        for line in f:
            proven_mutations.append(''.join(line.strip().split('-')))
    return proven_mutations

def get_balanced_train_test_cases(total_size,test_ratio,image_directory):
    cases = {k: len(v) for k, v in find_cases(image_directory).items()}
    label_sets = []
    for label in next(os.walk(image_directory))[1]:
        label_directory = os.path.join(image_directory, label)
        label_sets.append(set(find_cases(label_directory).keys()))
    intersection = set.intersection(*label_sets)
    reverse_sort_intersection = sorted(intersection, key=lambda x: cases[x])
    """
    We want to collect the intersection of cases for all labels (e.g. tumor, normal) 
    as we want both the test and train data to contain cases from this intersection
    the remaining data (most of this is cases that are only tumor) should be training
    we want to sort and reverse the intersection so that the first half is added to training
    """
    train_subset, test_subset = get_case_subsets(
        cases, reverse_sort_intersection, total_size * (1 - test_ratio)
    )  # gives us the subsets that contain equal amounts from intersection, and then fills training data until it reaches ratio

    return train_subset, test_subset
def get_case_subsets(case_dict, intersection, max_size):
    train_subset = []
    test_subset = []
    total = 0  # used to track if we go past max size
    for i, inter_case in enumerate(intersection):
        if (
            total + case_dict[inter_case] >= max_size
        ):  # if we get to max value from intersection (unlikely) quit
            break
        if i > (
            len(intersection) // 2
        ):  # for the latter half (biggest values), add to train
            train_subset.append((inter_case, case_dict[inter_case]))
            total += case_dict.pop(inter_case)
        else:  # second half, add to test
            test_subset.append((inter_case, case_dict[inter_case]))
            case_dict.pop(inter_case)
    cases = iter(
        reversed(sorted(list(case_dict.items()), key=lambda x: x[1]))
    )  # turn remaining cases into interable
    while True:
        try:
            case = next(cases)
        except StopIteration:  # break immediately if no more options
            break
        value = case[1]
        if total + value >= max_size:
            test_subset = test_subset + [case] + list(cases)
            break
        train_subset.append(case)
        total += value
    return [case[0] for case in train_subset], [case[0] for case in test_subset]
def get_case(path: str) -> str:
    """
    extracts case from image filepath, assuming "case" is everything before the last _ for the filename
    """
    return os.path.basename(path).rsplit("_", 1)[0]


def find_cases(image_directory):
    paths = [path for path in Path(image_directory).rglob("*.jpg")]
    cases = list(set([get_case(str(path)) for path in paths]))
    case_dict = {
        case: sorted(
            [
                os.path.join(path.parent, path.name)
                for path in paths
                if case in path.name
            ]
        )
        for case in cases
    }
    return case_dict

def get_norm_transform(train_set,seed,normalization_info):
    """
    Wrapper for get_mean_std_per_channel as torch transform
    """
    means,stds = get_mean_std_per_channel(train_set,normalization_info,seed)
    print(f"Dataset mean for RGB channels: {means}")
    print(f"Dataset standard deviation for RGB channels: {stds}")
    
    return transforms.Compose([
            # ResNet expects 224x224 images    
            transforms.Resize((224, 224)),
            # Converts the image to a PyTorch tensor, which also scales the pixel values to the range [0, 1]
            transforms.ToTensor(),
            # Normalizes the image tensor using  mean and standard deviation
            transforms.Normalize(mean=means, std=stds)
        ])

def get_mean_std_per_channel(dataset,dataset_info,seed):
    '''
    Loads mean and standard deviation of the dataset from a file if it exists, otherwise loads dataset at image_directory and computes mean and standard deviation
    '''
    mean_std_path = (
        f"./results/training/mean_stds/{dataset_info}"
    )
    mean_std_path += ".txt"
    try:
        with open(mean_std_path, "r") as f:
            means = [float(mean) for mean in f.readline().strip().split()]
            stds = [float(std) for std in f.readline().strip().split()]
    except (
        EOFError,
        FileNotFoundError,
    ):  # if the file does not exist, load dataset without transforming and compute mean and std
        means, stds = compute_and_save_mean_std_per_channel(
            dataset=dataset, path=mean_std_path, seed=seed
        )
    return means,stds


def compute_and_save_mean_std_per_channel(dataset, path, seed):
    device = utils.load_device(seed)
    if str(device) == "cpu":  # define smaller batchsize if no graphics card
        batch_size = 10
    else:
        batch_size = 200
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=get_allowed_forks(), shuffle=False
    )
    means = torch.zeros(3).to(device)
    stds = torch.zeros(3).to(device)
    for images, _ in tqdm(loader):
        images = images.to(device)
        for i in range(3):
            means[i] += images[:, i, :, :].mean()
            stds[i] += images[:, i, :, :].std()
    means.div_(len(loader)).cpu()
    stds.div_(len(loader)).cpu()
    with open(path, "w") as f:
        f.write(" ".join([str(float(mean)) for mean in means]) + "\n")
        f.write(" ".join([str(float(std)) for std in stds]))
    return means.tolist(), stds.tolist()


def count_dict_tensor(count_dict: dict):
    """
    Converts a dictionnary of the count of each class (returned by load_training_feature_data) into a 1D tensor (required for weighted cross entropy loss)
    """
    return torch.tensor(
        [sum(count_dict.values()) / count_dict[k] for k in sorted(count_dict.keys())]
    )


def get_allowed_forks():
    if os.name == "nt":
        return 0
    return 8


if __name__ == "__main__":

    seed = 99
    utils.set_seed(seed)
    k = 10000
    # print(*list(os.listdir('./images/DDC_UC_1/normalized_images/undiff')),sep='\n')
    # x = imread('./images/DDC_UC_1/normalized_images/undiff/AS19060903_275284.jpg_tile_3')
    for tumor in os.listdir("images"):
        print(tumor)
        if tumor not in ["DDC_UC_1"]:
            continue
        # for annotation in os.listdir(image_directory):
        #     print(tumor_type[0].lower() + annotation[0])
        #     if annotation in [".DS_Store", "__MACOSX"]:
        #         continue
        #     for image_path in tqdm(
        #         os.listdir(os.path.join(image_directory, annotation))
        #     ):
        #         image_full_path = os.path.join(image_directory, annotation, image_path)
        #         if tumor_type[0].lower() + annotation[0] in image_path:
        #             print(image_full_path)
        #             print(
        #                 os.path.splitext(image_full_path)[0][:-2]
        #                 + os.path.splitext(image_full_path)[1]
        #             )
        #             os.rename(
        #                 image_full_path,
        #                 os.path.splitext(image_full_path)[0][:-2]
        #                 + os.path.splitext(image_full_path)[1],
        #             )

        start_time = time.time()
        # load_training_image_data(
        #     batch_size=128, seed=seed, samples_per_class=150000, tumor_type=tumor,normalized=True, proven_mutation_only=False
        # )
        load_image_data(128,tumor,seed=seed,normalized=True,stain_normalize=False)
        load_image_data(128,tumor,seed=seed,normalized=True,stain_normalize=False)
        print(f"--- {(time.time() - start_time)} seconds ---")

        # check_for_unopenable_files(tumor_type)
