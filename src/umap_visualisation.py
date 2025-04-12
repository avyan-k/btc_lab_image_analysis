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
import umap
import umap.plot
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score as ARI, adjusted_mutual_info_score as NMI

# project files
import loading_data as ld
import models as md
import utils

"""FEATURE EXTRACTION"""

def get_features_array(model,model_name, tumor_type,stain_normalized, sample, sample_size=-1):
    image_directory = ld.get_image_directory(tumor_type=tumor_type,stain_normalized=stain_normalized)

    size_of_image_dataset = ld.get_size_of_dataset(image_directory, extension="jpg")
    sample_size = size_of_image_dataset if not sample else sample_size
    assert sample_size > 0
    features_filename = f"./features/{tumor_type}{'_normalized_' if stain_normalized else '_'}{model_name}_features.npz"
    annotation_filename = f"./features/{tumor_type}{'_normalized_' if stain_normalized else '_'}{model_name}_annotations.npz"

    # get features for images in image_loader
    if not os.path.isfile(features_filename) or not os.path.isfile(annotation_filename):
        image_loader,classes = ld.load_image_data(
            batch_size=1000, seed=seed, samples_per_class=sample_size if sample else -1, tumor_type=tumor_type,normalized=True,stain_normalize=stain_normalized
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
                    features_list.append(feature.cpu().numpy())
                for annotation in annotations:
                    annotations_list.append(classes[annotation])
        features_array = np.array(features_list)
        annotations_array = np.array(annotations_list)
        np.savez(features_filename,features_array)
        np.savez(annotation_filename,annotations_array)
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
def fit_umap_to_pca(pca, neighbours = 15,minimum_distance = 0.1):

    # UMAP dimension reduction on normalized features_array and coloring by annotations
    print(f"Started fitting UMAP on the {pca.shape[1]} features of {pca.shape[0]} images at {utils.get_time()}")

    features_scaled = standardize(pca)
    umap_model = umap.umap_.UMAP(
        n_neighbors=neighbours, min_dist=minimum_distance, metric="euclidean"
    )

    umap_model.fit(features_scaled)

    # Mapping annotations to colors
    return umap_model

def generate_umap_embeddings(pca, neighbours = 15,minimum_distance = 0.1):

    # UMAP dimension reduction on normalized features_array and coloring by annotations
    print(f"Started generating UMAP embeddings for the {pca.shape[1]} features of {pca.shape[0]} images at {utils.get_time()}")

    features_scaled = standardize(pca)
    umap_model = umap.umap_.UMAP(
        n_neighbors=neighbours, min_dist=minimum_distance, metric="euclidean"
    )

    umap_embedding = umap_model.fit_transform(features_scaled)

    assert isinstance(umap_embedding, np.ndarray)

    print(f"\ndim of umap_embedding:\t{umap_embedding.shape}\n")

    # Mapping annotations to colors
    return umap_model,umap_embedding

def k_means_clustering(features, tumour_classes):
    return cluster.KMeans(n_clusters=tumour_classes).fit_predict(features)

def scatter_plot(embeddings,annotations,title, save_path):

    # Generating figure
    plt.figure()
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=annotations, s=0.1);
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    print(f"UMAP plot saved in {save_path}")



def main(tumor_type,model,model_name,stain_normalized = False):

    annotations, model_features = get_features_array(model,model_name,tumor_type,stain_normalized=stain_normalized,sample=False)
    vectorized_annotations = np.vectorize({k:idx for idx,k in enumerate(set(annotations))}.get)(annotations)
    pca_features = get_PCA_embeddings(model_features)
    _,umap_embeddings = generate_umap_embeddings(pca_features)
    k_means = k_means_clustering(model_features,3)
    results_directory = "./results/normalized_umap"
    if stain_normalized:
        results_directory+="_stain_normalized"
    Path(os.path.join(results_directory)).mkdir(
            parents=True, exist_ok=True
    )
    ari = ARI(vectorized_annotations,k_means)
    nmi = NMI(vectorized_annotations,k_means)
    print(ari,nmi)
    plot_file = f"umap_{tumor_type}_{model_features.shape[0]}{'_normalized_' if stain_normalized else '_'}{model_name}_annotations_ARI={ari:0.3f}_nmi={nmi:0.3f}.png"  # filename
    plot_path = os.path.join(results_directory, plot_file)  # file path
    scatter_plot(umap_embeddings,vectorized_annotations,f"UMAP for {tumor_type} using {model_name}{' stain normalized' if stain_normalized else ''}, base labels",plot_path)
    plot_file = f"umap_{tumor_type}_{model_features.shape[0]}{'_normalized_' if stain_normalized else '_'}{model_name}_annotations_ARI={ari:0.3f}_nmi={nmi:0.3f}_K-Means.png"
    plot_path = os.path.join(results_directory, plot_file) 
    scatter_plot(umap_embeddings,k_means,f"UMAP for {tumor_type} using {model_name}{' stain normalized' if stain_normalized else ''}, K-means labels",plot_path)
    print(f"\nUMAP generation completed at {utils.get_time()}")


if __name__ == "__main__":
    seed = 99
    DEVICE = utils.load_device(seed)
    tumor = "DDC_UC_1"
    # resnet_tumor = md.ResNet_Tumor(classes=len(os.listdir(f"./images/{tumor}/images")))
    # resnet_tumor.load_state_dict(torch.load("./results/training/models/ResNet_Tumor/DDC_UC_1-10000-Normalized.pt"))
    # resnet_tumor.fc = torch.nn.Identity()
    # main(tumor,resnet_tumor,stain_normalized=True)
    # main(tumor,resnet_tumor,stain_normalized=False)
    main(tumor,md.get_resnet18_model(),"ResNet18",stain_normalized=True)
    main(tumor,md.get_resnet18_model(),"ResNet18",stain_normalized=False)
    main(tumor,md.get_resnet50_model(),"ResNet50",stain_normalized=True)
    main(tumor,md.get_resnet50_model(),"ResNet50",stain_normalized=False)
    main(tumor,md.get_VGG16_model(),"VGG16",stain_normalized=True)
    main(tumor,md.get_VGG16_model(),"VGG16",stain_normalized=False)