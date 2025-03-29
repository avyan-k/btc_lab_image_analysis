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
from sklearn.decomposition import PCA
import umap.umap_ as umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from tqdm import tqdm

# project files
import loading_data as ld
import models as md
import utils

"""FEATURE EXTRACTION"""

def get_features_array(model, tumor_type,stain_normalized, sample, sample_size=-1):
    image_directory = ld.get_image_directory(tumor_type=tumor_type,stain_normalized=stain_normalized)

    size_of_image_dataset = ld.get_size_of_dataset(image_directory, extension="jpg")
    sample_size = size_of_image_dataset if not sample else sample_size
    assert sample_size > 0
    features_filename = f"./results/{tumor_type}_{'normalized' if stain_normalized else ''}_{type(model).__name__}_features.npz"
    annotation_filename = f"./results/{tumor_type}_{'normalized' if stain_normalized else ''}_{type(model).__name__}_annotations.npz"

    # get features for images in image_loader
    if not os.path.isfile(features_filename) or not os.path.isfile(annotation_filename):
        image_loader,classes = ld.load_image_data(
            batch_size=100, seed=seed, samples_per_class=sample_size if sample else -1, tumor_type=tumor_type,normalized=True,stain_normalize=stain_normalized
        )

        model = model.to(DEVICE)
        features_list = []
        annotations_list = []
        print(f"Started Extracting features of {sample_size} images at {utils.get_time()}")
        for images, annotations in tqdm(image_loader):
            with torch.no_grad():
                images = images.to(DEVICE)
                features = model(images)
                for feature in features:
                    features_list.append(feature.cpu())
                for annotation in annotations:
                    annotations_list.append(classes[annotation])
        features_array = np.array(features_list)
        annotations_array = np.array(annotations_list)
    else:
        features_array = np.load(features_filename)["arr_0"]
        annotations_array = np.load(annotation_filename)["arr_0"]
    return (
        annotations_array,
        features_array,
    )  # convert list of vectors into numpy array


"""NORMALIZATION"""
def standardize(data):
    return (data-data.mean())/data.std()


"""PCA"""
def get_PCA_embeddings(features, output_dimension = 100):
    print(f"Started generating PCA for the {features.shape[1]} features of {features.shape[0]} images at {utils.get_time()}")
    scaled_features = standardize(features)
    pca = PCA(n_components=output_dimension) # reduce 1000 dimension vector to 100
    return pca.fit_transform(scaled_features)

"""UMAP GENERATION - COLORING BASED ON ANNOTATIONS"""
def generate_umap_annotation(pca, neighbours = 15,minimum_distance = 0.1):

    # UMAP dimension reduction on normalized features_array and coloring by annotations
    print(f"Started generating UMAP for the {pca.shape[1]} features of {pca.shape[0]} images at {utils.get_time()}")

    features_scaled = standardize(pca)
    umap_model = umap.UMAP(
        n_neighbors=neighbours, min_dist=minimum_distance, metric="euclidean"
    )

    umap_embedding = umap_model.fit_transform(features_scaled)

    assert isinstance(umap_embedding, np.ndarray)

    print(f"\ndim of umap_embedding:\t{umap_embedding.shape}\n")

    # Mapping annotations to colors
    return umap_embedding

def plot_umap(umap_embedding,annotations,tumor_type):
    results_directory = "./results/normalized_umap"
    Path(os.path.join(results_directory)).mkdir(
            parents=True, exist_ok=True
    )  # results directory for this file
    
    sample_size = umap_embedding.shape[0]
    umap_annotation_file = f"umap_{tumor_type}_{sample_size}_annotations.png"  # filename
    umap_annotation_output_path = os.path.join(
            results_directory, umap_annotation_file
    )  # file path

    annotation_colors = (
        {"normal": "blue", "undiff": "red", "well_diff": "green"}
        if "DDC_UC" in tumor_type
        else {"normal": "blue", "tumor": "red"}
    )  # says which color to associate to the annotations of each file
    colors = [annotation_colors[annotation] for annotation in annotations]

    # Generating figure
    plt.figure(figsize=(10, 8))
    plt.scatter(
        umap_embedding[:, 0], umap_embedding[:, 1], c=colors, s=5, alpha=0.7
    )  # type: ignore

    # legend
    handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10
        )
        for color in ["blue", "red", "green"]
    ]
    legend_labels = (
        ["normal", "undiff", "well_diff"]
        if "DDC_UC" in tumor_type
        else ["normal", "tumor"]
    )
    plt.legend(handles, legend_labels, title="Annotations")

    plt.title(f"{tumor_type} UMAP Projection")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(umap_annotation_output_path)
    plt.show()
    print(f"UMAP plot saved in {umap_annotation_output_path}")

def main(stain_normalized = False):
    for tumor_type in reversed(os.listdir("./images")):
        if tumor_type not in ["DDC_UC_1"]:
            continue

        model = md.ResNet_Tumor(classes=len(os.listdir(f"./images/{tumor_type}/images")))
        model.load_state_dict(torch.load(f"./results/training/models/ResNet_Tumor/k=10000_normalized/{tumor_type}/epochs=80-lr=0.001-seed=99.pt"))
        model.fc = torch.nn.Identity()
        annotations, model_features = get_features_array(model,tumor_type,stain_normalized=stain_normalized,sample=False)
        pca_features = get_PCA_embeddings(model_features)
        umap_embeddings = generate_umap_annotation(pca_features)

        plot_umap(umap_embeddings,annotations,tumor_type)
        print(f"\nUMAP generation completed at {utils.get_time()}")


if __name__ == "__main__":
    seed = 99
    DEVICE = utils.load_device(seed)
    main()