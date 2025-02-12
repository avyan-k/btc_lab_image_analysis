import random
import numpy as np
import os
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
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

def get_image_dataset(tumor_type,seed,samples_per_class = -1, normalized = True, stain_normalized=False, proven_mutation=False):
    '''
    Get entire image dataset or a balanced subset (if samples_per_class is set to a positive number). 
    If normalized is set to true, the mean and standard deviation of the dataset have been computed and saved, they are loaded from the file. Otherwise, they are computed and saved to the file. 
    If stain_normalized is set to true, the images are loaded from the normalized_images directory.
    '''

    image_directory = f"./images/{tumor_type}/images"
    if stain_normalized:
        image_directory = f"./images/{tumor_type}/normalized_images"
    print(f"\nLoading images from: {image_directory}")
    if samples_per_class == -1:
        samples_per_class = "all"


    processing_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # ResNet expects 224x224 images
            transforms.ToTensor(),  # Converts the image to a PyTorch tensor, which also scales the pixel values to the range [0, 1]
        ]
    )
    if normalized:
        means,stds = get_mean_std_per_channel(image_directory,tumor_type,samples_per_class,seed,stain_normalized)
        print(f"Dataset mean for RGB channels: {means}")
        print(f"Dataset standard deviation for RGB channels: {stds}")
        processing_transforms = transforms.Compose(
            [processing_transforms,
            transforms.Normalize(
                    mean=means, std=stds
                ),  # Normalizes the image tensor using  mean and standard deviation)
            ])
    if samples_per_class == "all":
        if proven_mutation:
            proven_mutation = get_proven_mutation_caes(tumor_type)
            return TumorImageDataset(root=image_directory, transform=processing_transforms, cases=proven_mutation)
        return TumorImageDataset(root=image_directory, transform=processing_transforms)
    else:
        if proven_mutation:
            proven_mutation = get_proven_mutation_caes(tumor_type)
            return BalancedTumorImageData(
                image_directory, k=samples_per_class, transform=processing_transforms, cases=proven_mutation
            )
        return BalancedTumorImageData(
            image_directory, k=samples_per_class, transform=processing_transforms
        )
def load_training_image_data(
        batch_size,
        tumor_type,
        seed,
        samples_per_class=-1,
        normalized=True,
        proven_mutation_only=False,
        validation=False,
    ):
    '''
    External wrapper for  get_loaders_training_image_data to allow any dataset to be loaded
    '''
    full_dataset = get_image_dataset(tumor_type=tumor_type,seed=seed,samples_per_class=samples_per_class,normalized=normalized, proven_mutation=proven_mutation_only)
    return get_loaders_training_image_data(full_dataset,batch_size,validation=validation)


def get_loaders_training_image_data(
    dataset,
    batch_size,
    validation=False,
):
    train_size = len(dataset)
    # Split the datasets into training, validation, and testing sets
    if validation:
        train_dataset, valid_dataset, test_dataset, _ = random_split(
            dataset,
            [
                int(train_size * 0.8),
                int(train_size * 0.1),
                int(train_size * 0.1),
                train_size
                - int(train_size * 0.8)
                - int(train_size * 0.1)
                - int(train_size * 0.1),
            ],
        )
    else:
        train_dataset, test_dataset, _ = random_split(
            dataset,
            [
                int(train_size * 0.9),
                int(train_size * 0.1),
                train_size - int(train_size * 0.9) - int(train_size * 0.1),
            ],
        )
        valid_dataset = None

    train_classes = dict(
        sorted(
            Counter([dataset.targets[i] for i in train_dataset.indices]).items()
        )
    )  # counter return a dictionnary of the counts, sort and wrap with dict to get dict sorted by key
    test_classes = dict(
        sorted(Counter([dataset.targets[i] for i in test_dataset.indices]).items())
    )

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

    print(
        f"Training set size: {len(train_dataset)}, Class Proportions: {({dataset.classes[k]:v for k,v in train_classes.items()})}"
    )

    print(
        f"Test set size: {len(test_dataset)}, Class Proportions: {({dataset.classes[k]:v for k,v in test_classes.items()})}"
    )

    if valid_dataset is not None:
        valid_classes = dict(
            sorted(
                Counter(
                    [dataset.targets[i] for i in valid_dataset.indices]
                ).items()
            )
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=get_allowed_forks(),
        )
        print(
            f"Validation set size: {len(valid_dataset)}, Class Proportions: {({dataset.classes[k]:v for k,v in valid_classes.items()})}"
        )
        return (train_loader, valid_loader, test_loader), (
            train_classes,
            valid_classes,
            test_classes,
        )
    else:
        return (train_loader, test_loader), (
            train_classes,
            test_classes,
        )
def load_image_data(
    batch_size,     
    tumor_type,
    seed,
    samples_per_class=-1,
    normalized=True,

):
    '''
    Same as load_training_image_data but a single loader is returned with the entire dataset, along with a list of class names
    '''
    full_dataset = get_image_dataset(tumor_type=tumor_type,seed=seed,samples_per_class=samples_per_class,normalized=normalized)

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
    image_directory = f"./images/{tumor_type}/images"
    if normalized:
        image_directory = f"./images/{tumor_type}/normalized_images"
    print(f"\nLoading images from: {image_directory}")

    transforms = v2.Compose(
        [v2.RandomAffine(degrees=15, translate=(0.15, 0.15)), transforms]
    ) if transforms is not None else v2.RandomAffine(degrees=15, translate=(0.15, 0.15))
    # get full data set
    # full_train_dataset = datasets.ImageFolder(image_directory,transform=transforms)
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
    image_directory = f"./images/{tumor_type}/images"
    return [
        name
        for name in os.listdir(image_directory)
        if name not in [".DS_Store", "__MACOSX"]
    ]


def check_for_unopenable_files(tumor_type, norm=False):
    if norm:
        image_directory = f"./images/{tumor_type}/normalized_images"
    else:
        image_directory = f"./images/{tumor_type}/images"
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
    if norm:
        image_directory = f"./images/{tumor_type}/normalized_images"
    else:
        image_directory = f"./images/{tumor_type}/images"
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


def get_proven_mutation_caes(tumor_type):    
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

def get_mean_std_per_channel(image_directory,tumor_type,samples_per_class,seed,stain_normalized=False,proven_mutation = False):
    '''
    Loads mean and standard deviation of the dataset from a file if it exists, otherwise loads dataset at image_directory and computes mean and standard deviation
    '''
    mean_std_path = (
        f"./results/training/{tumor_type}-k={samples_per_class}-seed={seed}"
    )
    if stain_normalized:
        mean_std_path += "_normalized"
    if proven_mutation:
        mean_std_path += "_proven-mutation-only"
    mean_std_path += ".txt"
    try:
        with open(mean_std_path, "r") as f:
            means = [float(mean) for mean in f.readline().strip().split()]
            stds = [float(std) for std in f.readline().strip().split()]
    except (
        EOFError,
        FileNotFoundError,
    ):  # if the file does not exist, load dataset without transforming and compute mean and stf
        if samples_per_class == "all":
            dataset = datasets.ImageFolder(
                root=image_directory, transform=transforms.ToTensor()
            )
            k = len(dataset)
        else:
            dataset = BalancedTumorImageData(
                image_directory, k=samples_per_class, transform=transforms.ToTensor()
            )
            k = samples_per_class
        means, stds = compute_and_save_mean_std_per_channel(
            dataset=dataset, path=mean_std_path, seed=seed, k=k
        )
    return means,stds


def compute_and_save_mean_std_per_channel(dataset, path, seed, k=-1):
    device = utils.load_device(seed)
    if k == -1:
        k = len(dataset)
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
        f.write(" ".join([str(float(mean)) for mean in means]))
    return means, stds


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
    for tumor_type in os.listdir("images"):
        print(tumor_type)
        if tumor_type not in ["DDC_UC_1"]:
            continue
        image_directory = f"./images/{tumor_type}/images"

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
        load_training_image_data(
            batch_size=128, seed=seed, samples_per_class=-1, tumor_type=tumor_type,normalized=True, proven_mutation_only=False
        )
        print(f"--- {(time.time() - start_time)} seconds ---")

        # check_for_unopenable_files(tumor_type)
