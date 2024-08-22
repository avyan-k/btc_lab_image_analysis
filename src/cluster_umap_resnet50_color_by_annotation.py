"""
2024/08/20

UMAP clustering of images 
Coloring by annotations (normal, tumor for SCCOHT & vMRT   |   normal, well_diff, undiff for DDC_UC)

Image dataset contains .jpg image files of tiles for different annotations.

The .jpg image files must follow a naming system where
    - the last character of the base filename represents the annotation.
        Ex. in the file 'OC1813435_A4_4030sn.jpg' the last 'n' character represents the annotation 'normal'
    - the before-to-last character of the base filename represents the tumor_type.
        Ex. in the file 'OC1813435_A4_4030sn.jpg' the before-to-last 's' character represents the tumor_type 'sccoht'

TO CHANGE
- n_clusters (int)  



"""

print("Loading packages ...")

import os
import sys
import re
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

"""RESNET50"""
def setup_resnet_model():
    # Defines transformations to apply on images
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(), # Converts the image to a PyTorch tensor, which also scales the pixel values to the range [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizes the image tensor using the mean and standard deviation values that were used when training the ResNet model (usually on ImageNet)
    ])

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    return preprocess, model


"""FEATURE EXTRACTION"""
def extract_features(image_paths, preprocess, model):
    image = Image.open(image_paths).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # Add batch dimension, "deep learning models in PyTorch expect input in the form of a batch of images, even if it's a single image. The batch size in this case is 1"
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()

def get_features_array(image_directory, preprocess, model):
    # Get paths of images in image_directory
    image_paths = [path for path in Path(image_directory).rglob('*.jpg')]
    annotations = [path.parent.name for path in image_paths]

    
    # Get list of feature vectors for each image
    features_list = []
    for i, path in enumerate(image_paths):
        sys.stdout.write(f"\rExtracting features for images {i + 1} out of {len(image_paths)}")
        sys.stdout.flush()
        features = extract_features(path, preprocess, model) # get vector of features for each image
        features_list.append(features) # add vector to a list
    sys.stdout.write("\n")
    return image_paths, annotations, np.array(features_list) # convert list of vectors into numpy array


"""NORMALIZATION"""
def normalization_features_array(features_array):
    scaler = StandardScaler()
    # fit: computes means and std for each vector of features in array
    # transform: normalizes so that each vector has a mean=0 and std=1
    return scaler.fit_transform(features_array)


"""UMAP GENERATION - COLORING BASED ON ANNOTATIONS"""
def extract_annotation(image_path):
    match = re.search(r'(t|n)\.jpg$', image_path) # search for a match for n.jpg or t.jpg
    return match.group(1) if match else None # returns for which annotation the match was found

def map_annotations_to_colors(annotations):
    annotation_colors = {'normal': 'blue', 'tumor': 'red'}
    return [annotation_colors[annotation] for annotation in annotations] # returns which color to use based on annotation


def generate_umap_annotation(features_scaled, seed, annotations, umap_annotation_output_path):
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=seed)
    umap_embedding = umap_model.fit_transform(features_scaled)

    print(f"\ndim of umap_embedding:\t{umap_embedding.shape}\n")
    print(f"umap_embedding:\n{umap_embedding}\n")

    # Get annotations from last 2 characters before .jpg extension 
    print(f"annotations:\n{annotations}\n")
    colors = map_annotations_to_colors(annotations) # says which color to associate to the annotations of each file
    print(f"\n\n{colors}\n")

    # Generating figure settings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=colors, s=5, alpha=0.7)
    #####scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=5, alpha=0.7)

    #legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in ['blue', 'red']]
    labels = ['n', 't']
    plt.legend(handles, labels, title='Annotations')

    plt.title(f'{tumor_type} UMAP Projection') ##### HOW DOES IT KNOW TO TAKE tumor_type I NEVER PASSED IT AS AN ARGUMENT
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    #plt.savefig(umap_annotation_output_path)
    plt.show()
    return umap_embedding


"""KMEANS CLUSTERING ON UMAP PROJECTION"""
def kmeans_clustering(umap_embedding, seed):
    n_clusters = 2
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
    scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=clusters, cmap=ListedColormap(cluster_colors), s=5)

    # Create custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[i], markersize=10, label=f'Cluster {i}') for i in range(n_clusters)]
    plt.legend(handles=handles, title='Clusters')


    plt.title(f'{tumor_type} UMAP Projection with k-means clustering on UMAP embeddings') ##### HOW DOES IT KNOW TO TAKE tumor_type I NEVER PASSED IT AS AN ARGUMENT
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    #plt.savefig(umap_kmeans_output_path)
    plt.show()
    return

def kmeans_clusters_from_umap(umap_embedding, seed, umap_kmeans_output_path, cluster_csv_file_path, image_paths):

    # clustering from umap_embeddings
    n_clusters, clusters = kmeans_clustering(umap_embedding, seed)

    plot_umap_for_kmeans(n_clusters, clusters, umap_embedding, umap_kmeans_output_path)

    # save a .csv file with cluster information for each file
    cluster_info = pd.DataFrame({'ImagePath': image_paths, 'Cluster': clusters})
    cluster_info.to_csv(cluster_csv_file_path, index=False)
    
    
    return


"""MAIN UMAP GENERATION FUNCTION"""
def cluster_and_visualize_images(seed,image_directory, umap_annotation_outpath_path, umap_kmeans_output_path, csv_output_path, pdf_output_path, elbowplot_output_path):
    
    print(f"TESTING with: {image_directory}\n")
    
    # Setting the seeds for reproducibility
    np.random.seed(seed) # numpy random seed
    # torch random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}\n")
    else:
        torch.manual_seed(seed)
        device = torch.device("cpu")
        print(f"Using CPU\n")

    # ResNet50 model
    preprocess, model = setup_resnet_model()
    #print(f"Model structure:\n{model}\n")
    #print(f"type(preprocess):\t{type(preprocess)}\n")
    model.eval()
    print("ResNet50 model setup complete.\n")
    
    """
    for image_path in image_paths:
        print(image_path)
    """

    # Feature extraction from images --> into array of features for each image
    image_paths, annotations, features_array = get_features_array(image_directory, preprocess, model)
    print(f"\nfeatures_array.shape: (num_images, num_features)\n{features_array.shape}\n")

    # Normalization for features_array
    #features_scaled =  normalization_features_array(features_array)
    #print(f"\nfeatures_scaled.shape: (num_images, num_features)\n{features_scaled.shape}\n")

    # UMAP dimension reduction with annotations in legend
    # uses normalized features_array
    umap_embeddings = generate_umap_annotation(normalization_features_array(features_array), seed, annotations, umap_annotation_outpath_path)

    # UMAP dimension reduction with k-means clustering on umap_embeddings in legend
    kmeans_clusters_from_umap(umap_embeddings, seed, umap_kmeans_output_path, csv_output_path, image_paths)
    
    return

def get_time():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp


if __name__ == "__main__":

    # Set up parameters
    run_id = "test1"
    tumor_type = "sccoht"
    seed = 99
    sample_size = 100

    # Paths
    #image_directory = "/Users/Account/Desktop/AI_project_files/image_clustering/data_sccoht_pure_08-19"
    image_directory = "/Users/Account/Desktop/AI_project_files/image_clustering/data_test"

    results_directory = "/Users/Account/Desktop/AI_project_files/image_clustering/results_test"
    
    # Output file names
    umap_annotation_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}_annotation.png"
    umap_kmeans_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}_kmeans.png"
    csv_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}.csv"
    pdf_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}.pdf"
    elblowplot_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}_elbow_plot.png"

    # Output file paths
    umap_annotation_outpath_path = os.path.join(results_directory, umap_annotation_file)
    umap_kmeans_output_path = os.path.join(results_directory, umap_kmeans_file)
    csv_output_path = os.path.join(results_directory, csv_file)
    pdf_output_path = os.path.join(results_directory, csv_file)
    elbowplot_output_path = os.path.join(results_directory, elblowplot_file)


    ######### ADD UMAP_KMEANS_OUTPUT_PATH TOO IN FUNCT CALL PARAMETERS AND FUNCTION ###########
    cluster_and_visualize_images(image_directory=image_directory,
                                 umap_annotation_outpath_path=umap_annotation_outpath_path,
                                 umap_kmeans_output_path=umap_kmeans_output_path,
                                 csv_output_path=csv_output_path,
                                 pdf_output_path=pdf_output_path,
                                 elbowplot_output_path=elbowplot_output_path,
                                 seed=seed
                                 )


    timestamp = get_time()
    print(f"\n\nCompleted at {timestamp}")

