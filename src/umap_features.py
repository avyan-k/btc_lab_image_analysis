"""
2024/08/20

UMAP clustering of images 
    -> Coloring by annotations (normal, tumor for SCCOHT & vMRT   |   normal, well_diff, undiff for DDC_UC)

Image dataset contains .jpg image tiles for different annotations.

images/   #Primary data folder for the project
├── DDC_UC/ 
│   ├── images/
│   │   ├── normal/
│   │   |   ├── image01.jpg
│   │   |   ├── image02jpg
│   │   |   └── ...
│   │   ├── undiff/
│   │   ├── well-diff/
│   │   └── ...
├── SCCOHT/ 
│   ├── images/
│   │   ├── normal/
│   │   |   ├── image01.jpg
│   │   |   └── ...
│   │   ├── tumor/
│   │   |   ├── image10.jpg
|   │   │   └── ..


"""

import os
import numpy as np
import torch
# from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
# project files
import loading_data as ld
import utils

"""FEATURE EXTRACTION"""
def get_and_save_features_array(batch_size, model,transforms, tumor_type, size_of_dataset, sample_size, save=False,save_path = ""):
    image_loader,filepaths,labels = ld.load_data(batch_size,tumor_type,transforms=transforms,sample=size_of_dataset > sample_size, sample_size=sample_size)
    model = model.to(DEVICE)
    # get features for images in image_loader
    features_list = []
    print(f"\nExtracting features from {sample_size} images out of {size_of_dataset}")
    for images,annotations in tqdm(image_loader):
        with torch.no_grad():
            images = images.to(DEVICE)
            features = model(images)
            features_list.append(features.cpu()) # list of feature vectors for each image
    all_features = np.array(torch.cat(features_list, dim=0))
    print("Saving Data")
    # save feature vectors in numpy arrays
    if save:
        assert save_path != ""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        print(f"Saving features")
        for annotation_type in list(set(labels)):
            Path(os.path.join(save_path,annotation_type)).mkdir(parents=True, exist_ok=True)
        for i,features_array in enumerate(tqdm(all_features)):
            filename = os.path.splitext(os.path.basename(filepaths[i]))[0]
            annotation = labels[i]
            np.savez(os.path.join(save_path,annotation,filename),features_array)
    return filepaths, labels, np.array(torch.cat(features_list, dim=0)) # convert list of vectors into numpy array

def get_features_from_disk(tumor_type,model_type,size_of_dataset,sample_size):
    # feature vectors for each image from saved numpy arrays in disk
    feature_loader,filepaths, classes = ld.load_feature_data(batch_size=1,model_type=model_type,tumor_type=tumor_type,sample=size_of_dataset > sample_size, sample_size=sample_size)
    return filepaths, feature_loader, classes
def get_features_from_loader(feature_loader, classes):
    '''
    Loads all features to memory from loader and returns them along with annotations
    Arguments
    feature_loader: unshuffled DataLoader containing features
    classes: list of possible annotations sorted in same order as feature_loader (will depend on dataset from which it was loader, but should normally be sorted alphabetically)
    return
    features_list
    annotation_list
    '''
    features_list = []
    annotation_list = []
    print(f"Loading features of {len(feature_loader)} images")
    for feature,annotation in tqdm(feature_loader):
        features_list.append(feature)
        annotation_list.append(classes[annotation])
    return torch.cat(features_list, dim=0).numpy(),annotation_list
"""NORMALIZATION"""
def feature_normalizer():
    # fit: computes means and std for each vector of features in array
    # transform: normalizes so that each vector has a mean=0 and std=1
    return StandardScaler()


"""UMAP GENERATION - COLORING BASED ON ANNOTATIONS"""
def generate_umap_annotation(feature_loader,seed, tumor_type, save_plot = False, umap_annotation_output_path = '', tumor_classes = ['normal', 'tumor'], normalizer = None):
    features_array, annotations = get_features_from_loader(feature_loader,classes=tumor_classes)
    features_scaled = normalizer.fit_transform(features_array)
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=seed)
    umap_embedding = umap_model.fit_transform(features_scaled)

    print(f"\ndim of umap_embedding:\t{umap_embedding.shape}\n") # type: ignore

    # Mapping annotations to colors
    annotation_colors = {'normal': 'blue', 'undiff': 'red', 'well_diff' : 'green'} if'DDC_UC' in tumor_type else {'normal': 'blue', 'tumor': 'red'} # says which color to associate to the annotations of each file
    colors = [annotation_colors[annotation] for annotation in annotations]

    if save_plot:
        assert umap_annotation_output_path != ''
        # Generating figure
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=colors, s=5, alpha=0.7) # type: ignore

        #legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in ['blue', 'red', 'green']]
        legend_labels = ['normal', 'undiff', 'well_diff'] if 'DDC_UC' in tumor_type else ['normal', 'tumor']
        plt.legend(handles, legend_labels, title='Annotations')

        plt.title(f'{tumor_type} UMAP Projection')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.savefig(umap_annotation_output_path)
        plt.show()
    return umap_embedding


def get_size_of_dataset(directory, extension):
    return len([path for path in Path(directory).rglob(f'*.{extension}')])

def generate_umap_from_dataset(tumor_type, seed, model_type = "ResNet" ,sample = False, sample_size = -1, plot = False):
    '''
    Takes features (extracted from images using RESNET) currently in disk and generates UMAP
    Will always use all available data and will save features to disk if they do not exist
    If a sample is needed, set sample to True and specify sample size, otherwise entire dataset will be used
    If a bUMAP plot should be generated and saved, set plot to True

    returns the following fives lists, all in the same order
    image_paths: path leading to the image processed
    annotations: label placed on image (e.g. tumor vs normal)
    cases: case id of the slide from which the image was generated
    features_array: features extracted from image
    umap_embeddings: 2D umap projection of features
      
    '''
    
    # Paths to directories
    image_directory = f"./images/{tumor_type}/images"
    feature_directory = f"./features/{model_type}/{tumor_type}"
    Path(os.path.join(feature_directory)).mkdir(parents=True, exist_ok=True) # results directory for this file
    results_directory = f"./results/umap"
    Path(os.path.join(results_directory)).mkdir(parents=True, exist_ok=True) # results directory for this file
    
    assert os.path.isdir(image_directory) #TODO change to expection throwing

    # Set up parameters
    number_of_previous_results = len([name for name in os.listdir(results_directory) if os.path.isfile(name)])
    run_id = f"{model_type}_{utils.get_time()[:10]}_{number_of_previous_results}"
    batch_size = 100 # tested to be optimal at 100 batches, should be changed manually
    size_of_image_dataset = get_size_of_dataset(image_directory,extension='jpg')
    size_of_feature_dataset = get_size_of_dataset(feature_directory,extension='npz')

    if not sample: # if we do not sample use entire dataset
        sample_size = size_of_image_dataset
    assert sample_size > 0 #after previous check sample size should be a positive number
        
    # Output file
    sample_size_text = "all" if not sample else sample_size #text to indicate sample size in file name
    umap_annotation_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size_text}_annotation.png" # filename
    umap_annotation_outpath_path = os.path.join(results_directory, umap_annotation_file) # file path

    # get features
    if size_of_image_dataset != size_of_feature_dataset: # assume features have not been extracted
        print(f"Feature extraction directory {feature_directory} not found or incomplete, launching feature extraction")
        print(f"\nSetting up {model_type} model ...")
        match model_type:
                case 'ResNet': model,transforms = ld.setup_resnet_model(seed)
                case 'VGG16': model,transforms = ld.setup_VGG16_model(seed)
        model.eval()
        get_and_save_features_array(batch_size=batch_size, model= model,transforms = transforms, tumor_type=tumor_type, size_of_dataset = size_of_image_dataset,sample_size = sample_size, save=True,save_path=feature_directory)
     
    image_paths, feature_loader, tumor_classes = get_features_from_disk(tumor_type=tumor_type,model_type=model_type,size_of_dataset=size_of_feature_dataset,sample_size=sample_size)
    

    # UMAP dimension reduction on normalized features_array and coloring by annotations
    print(f"\nGenerating UMAP for the features of {len(feature_loader)} images ...")
    umap_embeddings = generate_umap_annotation(feature_loader, seed,tumor_type, save_plot = plot, umap_annotation_output_path = umap_annotation_outpath_path, tumor_classes= tumor_classes ,normalizer = feature_normalizer()) 

    print(f"\nUMAP generation completed at {utils.get_time()}")

    return image_paths, feature_loader, umap_embeddings



if __name__ == "__main__":
    seed = 99
    DEVICE = utils.load_device(seed)
    generate_umap_from_dataset(tumor_type="SCCOHT_1", seed = seed,model_type="VGG16",sample=False, sample_size = 1000, plot=True)
    # for tumor_type in os.listdir('./images'):
    #     print(tumor_type)
    #     generate_umap_from_dataset(tumor_type=tumor_type, seed = seed,model_type="VGG16",sample=False, sample_size = 1000, plot=True)
    
    # # Set up parameters
    # run_id = f"{utils.get_time()[:10]}"
    # tumor_type = "vMRT"  
    # seed = 99
    # DEVICE = utils.load_device(seed)
    

    # # Paths to directories
    # image_directory = f"./images/{tumor_type}/images"
    # feature_directory = f"./features/{tumor_type}"
    # results_directory = f"./results/umap"
    # Path(os.path.join(results_directory)).mkdir(parents=True, exist_ok=True)

    # size_of_image_dataset = get_size_of_dataset(image_directory,extension='jpg')
    # size_of_feature_dataset = get_size_of_dataset(feature_directory,extension='npz')
    # sample_size = size_of_image_dataset
    # batch_size = 100    
    # # Output file
    # umap_annotation_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}_annotation.png" # filename
    # umap_annotation_outpath_path = os.path.join(results_directory, umap_annotation_file) # file path

    # # Seed for reproducibility
    # np.random.seed(seed) # numpy random seed

    # # ResNet50 model
    # print("\nSetting up ResNet model ...")
    # model = ld.setup_resnet_model(seed)
    # model.eval()
    
    # # Feature extraction from images and saving into numpy array
    # save_features = sample_size == size_of_image_dataset # ONLY save when using entire dataset
    # image_paths, annotations, features_array = get_and_save_features_array(batch_size, model,tumor_type, size_of_dataset = size_of_dataset,sample_size = size_of_dataset, save=save_features)
    # # OR
    # # Retrieve features from disk (numpy arrays)
    # # image_paths, annotations, features_array = get_features_from_disk(tumor_type,size_of_dataset= size_of_feature_dataset,sample_size=size_of_feature_dataset)
    # print(f"\nfeatures_array.shape: (num_images, num_features)\n{features_array.shape}\n")

    # # UMAP dimension reduction on normalized features_array and coloring by annotations
    # print(f"\nGenerating UMAP for the features of {features_array.shape[0]} images ...")
    # umap_embeddings = generate_umap_annotation(normalization_features_array(features_array), seed, annotations,tumor_type, save_plot = True, umap_annotation_output_path = umap_annotation_outpath_path)    

    

