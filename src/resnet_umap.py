"""
2024/08/20

UMAP clustering of images 
Coloring by annotations (normal, tumor for SCCOHT & vMRT   |   normal, well_diff, undiff for DDC_UC)

Image dataset contains .jpg image files of tiles for different annotations.

images/   #Primary data folder for the project
├── DDC_UC/ 
│   ├── images/
│   │   ├── normal/
│   │   |   ├── image01.jpg
│   │   |   ├── image02jpg
│   │   |   └── ...
│   │   ├── undiff/
│   │   ├── well-diff
│   │   └── ...
├── SCCOHT/ 
│   ├── images/
│   │   ├── normal/
│   │   |   ├── image01.jpg
│   │   |   ├── image02jpg
│   │   |   └── ...
│   │   ├── tumor/
│   │   |   ├── image10.jpg
│   │   |   ├── image11jpg
|   │   │   └── ..

TO CHANGE
- n_clusters (int)  



"""

print("Loading packages ...")

import os
import sys
import re
import numpy as np
import pandas as pd
import random
import torch
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
# project files
import loading_data as ld
import utils

"""FEATURE EXTRACTION"""
def extract_features(image_paths, preprocess, model):
    image = Image.open(image_paths).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # Add batch dimension, "deep learning models in PyTorch expect input in the form of a batch of images, even if it's a single image. The batch size in this case is 1"
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()

def get_and_save_features_array(batch_size, model,save=False):
    # Get paths of images in image_directory
    # tumor_paths = [path for path in Path(image_directory).rglob('tumor/*.jpg')]
    # image_paths = random.sample(tumor_paths,sample_size)
    # normal_paths = [path for path in Path(image_directory).rglob('normal/*.jpg')]
    # image_paths.extend(random.sample(normal_paths,sample_size))
    # annotations = [path.parent.name for path in image_paths]
    
    # Get list of feature vectors for each image
    image_loader,filepaths,labels = ld.load_data(batch_size,tumor_type,seed,sample_size)
    model = model.to(DEVICE)
    features_list = []
    for images,annotations in tqdm(image_loader):
        with torch.no_grad():
            images = images.to(DEVICE)
            features = model(images)
            features_list.append(features.cpu())
        # features = extract_features(path, preprocess, model) # get vector of features for each image
        # features_list.append(features) # add vector to a list
    all_features = np.array(torch.cat(features_list, dim=0))
    print("Saving Data")
    if save: #TODO test this
        datapath = "./features"
        Path(os.path.join(datapath,tumor_type)).mkdir(parents=True, exist_ok=True)
        for annotation_type in list(set(labels)):
            Path(os.path.join(datapath,tumor_type,annotation_type)).mkdir(parents=True, exist_ok=True)
        for i,features_array in enumerate(tqdm(all_features)):
            filename = os.path.splitext(os.path.basename(filepaths[i]))[0]
            annotation = labels[i]
            np.savez(os.path.join(datapath,tumor_type,annotation,filename),features_array)
    return filepaths, labels, np.array(torch.cat(features_list, dim=0)) # convert list of vectors into numpy array

def get_features_from_disk(size_of_dataset):
    # Get list of feature vectors for each image
    feature_loader,filepaths,labels = ld.load_feature_data(size_of_dataset,tumor_type,seed,sample_size)
    return filepaths, labels, next(iter(feature_loader))[0].numpy()
"""NORMALIZATION"""
def normalization_features_array(features_array):
    scaler = StandardScaler()
    # fit: computes means and std for each vector of features in array
    # transform: normalizes so that each vector has a mean=0 and std=1
    return scaler.fit_transform(features_array)


"""UMAP GENERATION - COLORING BASED ON ANNOTATIONS"""
def generate_umap_annotation(features_scaled, seed, annotations, umap_annotation_output_path,tumor_type):
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=seed)
    umap_embedding = umap_model.fit_transform(features_scaled)

    print(f"\ndim of umap_embedding:\t{umap_embedding.shape}\n")

    # Mapping annotations to colors
    # print(f"annotations:\n{annotations}\n")
    annotation_colors = {'normal': 'blue', 'undiff': 'red', 'well_diff' : 'green'} if'DDC_UC' in tumor_type else {'normal': 'blue', 'tumor': 'red'} # says which color to associate to the annotations of each file
    colors = [annotation_colors[annotation] for annotation in annotations]
    # print(f"\n\n{colors}\n")

    # Generating figure settings
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=colors, s=5, alpha=0.7)

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


"""KMEANS CLUSTERING ON UMAP PROJECTION"""
def kmeans_clustering(umap_embedding, seed, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(umap_embedding)
    return n_clusters, clusters

def plot_umap_for_kmeans(n_clusters, clusters, umap_embedding, umap_kmeans_output_path):
    # Define a custom colormap with distinct colors for each cluster (max 10 colors, 10 clusters)
    num_colors = min(n_clusters, 10)  # Ensure we don't exceed the number of available colors
    colormap = plt.get_cmap('tab10', num_colors)
    cluster_colors = colormap(range(num_colors))

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=clusters, cmap=ListedColormap(cluster_colors), s=5)

    # legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[i], markersize=10, label=f'Cluster {i}') for i in range(n_clusters)]
    plt.legend(handles=handles, title='Clusters')

    plt.title(f'{tumor_type} UMAP Projection with k-means clustering on UMAP embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(umap_kmeans_output_path)
    plt.show()
    return

def kmeans_clusters_from_umap(umap_embedding, seed, umap_kmeans_output_path, cluster_csv_file_path, image_paths, annotations,n_clusters):

    # clustering from umap_embeddings
    n_clusters, clusters = kmeans_clustering(umap_embedding, seed, n_clusters)

    plot_umap_for_kmeans(n_clusters, clusters, umap_embedding, umap_kmeans_output_path)

    # save a .csv file with cluster information for each file
    cluster_info = pd.DataFrame({'ImagePath': image_paths, 'Annotations': annotations,'Cluster': clusters})
    cluster_info.to_csv(cluster_csv_file_path, index=False)
    


    
    
    return

def get_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    
    # Set up parameters
    run_id = f"{get_time()[:10]}"
    tumor_type = "vMRT"  
    seed = 99
    utils.set_seed(seed)
    DEVICE = utils.load_device()
    size_of_image_dataset = len([path for path in Path(f"./images/{tumor_type}/images").rglob('*.jpg')])
    size_of_feature_dataset = len([path for path in Path(f"./features/{tumor_type}").rglob('*.npz')])
    sample_size = size_of_image_dataset
    batch_size = 100

    # Paths
    #image_directory = "/Users/Account/Desktop/AI_project_files/image_clustering/data_sccoht_pure_08-19"
    image_directory = f"./images/{tumor_type}/images"
    feature_directory = f"./features/{tumor_type}"
    results_directory = "./results"
    
    # Output file names
    umap_annotation_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}_annotation.png"
    umap_kmeans_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}_kmeans.png"
    csv_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}.csv"
    pdf_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}.pdf"

    # Output file paths
    umap_annotation_outpath_path = os.path.join(results_directory, umap_annotation_file)
    umap_kmeans_output_path = os.path.join(results_directory, umap_kmeans_file)
    csv_output_path = os.path.join(results_directory, csv_file)
    pdf_output_path = os.path.join(results_directory, csv_file)
    
    # Setting the seeds for reproducibility
    np.random.seed(seed) # numpy random seed

    # ResNet50 model
    model = ld.setup_resnet_model(seed)
    model.eval()
    print("ResNet50 model setup complete.\n")
    
    # Feature extraction from images --> into array of features for each image
    save_features = sample_size == size_of_image_dataset # ONLY save when using entire dataset
    image_paths, annotations, features_array = get_and_save_features_array(batch_size, model,save=save_features)
    # image_paths, annotations, features_array = get_features_from_disk(size_of_feature_dataset)
    print(f"\nfeatures_array.shape: (num_images, num_features)\n{features_array.shape}\n")

    # UMAP dimension reduction with annotations in legend
    # uses normalized features_array
    print(f"Generating UMAP for the features {features_array.shape[0]} images")
    umap_embeddings = generate_umap_annotation(normalization_features_array(features_array), seed, annotations, umap_annotation_outpath_path,tumor_type)
    n_clusters = 3 if 'DDC_UC' in tumor_type else 2      
    # UMAP dimension reduction with k-means clustering on umap_embeddings in legend

    kmeans_clusters_from_umap(umap_embeddings, seed, umap_kmeans_output_path, csv_output_path, image_paths, annotations, n_clusters)

    print(f"\n\nCompleted at {get_time()}")

