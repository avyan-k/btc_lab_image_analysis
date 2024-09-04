import os
from pathlib import Path
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
# project files
import utils
import loading_data as ld
import resnet_umap as ru


"""UMAP GENERATION - COLORING BASED ON ACCESSION NUMBERS"""
def generate_umap_case(features_scaled, seed, image_paths, save_plot=False, umap_annotation_output_path=''):
    # Initialize UMAP model
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=seed)
    umap_embedding = umap_model.fit_transform(features_scaled)

    print(f"\ndim of umap_embedding:\t{umap_embedding.shape}\n")

    # Extract accession numbers from the image file paths
    accession_numbers = [os.path.basename(path).split('_')[0] for path in image_paths]
    unique_accession_numbers = list(set(accession_numbers))

    # Create a color palette for the unique accession numbers
    color_palette = plt.get_cmap('tab20', len(unique_accession_numbers))  # Adjust the colormap as needed
    accession_colors = {acc_num: color_palette(i) for i, acc_num in enumerate(unique_accession_numbers)}

    # Map colors to UMAP points based on accession numbers
    colors = [accession_colors[accession] for accession in accession_numbers]

    if save_plot:
        assert umap_annotation_output_path != '', "Output path must be specified when save_plot is True."
        # Generating figure
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=colors, s=5, alpha=0.7)

        # Generate legend for each unique accession number
        handles = [plt.Line2D([0], [0], marker='o', color=color_palette(i), markersize=10) for i in range(len(unique_accession_numbers))]
        legend_labels = unique_accession_numbers
        plt.legend(handles, legend_labels, title='Accession Numbers')

        plt.title('UMAP Projection Colored by Accession Numbers')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.savefig(umap_annotation_output_path)
        plt.show()

    return umap_embedding


if __name__ == "__main__":
    tumor_type = "DDC_UC_1"
    run_id = f"{utils.get_time()[:10]}"
    seed = 99
    size_of_feature_dataset = ru.get_size_of_dataset(tumor_type=tumor_type, extension='jpg')
    sample_size = size_of_feature_dataset
    batch_size = 100

    # Directory paths
    feature_directory = f"./features/{tumor_type}"
    results_directory = f"./results/umap"
    Path(os.path.join(results_directory)).mkdir(parents=True, exist_ok=True)

    # Output file
    umap_annotation_file = f"umap_annotation_{tumor_type}_{run_id}_{seed}_{sample_size}_1.png"  # filename
    umap_annotation_outpath_path = os.path.join(results_directory, umap_annotation_file)  # file path

    # Seed for reproducibility
    np.random.seed(seed)  # numpy random seed

    # ResNet50 model
    print("\nSetting up ResNet model ...")
    model = ld.setup_resnet_model(seed)
    model.eval()
    
    # Retrieve features from disk (numpy arrays)
    image_paths, annotations, features_array = ru.get_features_from_disk(sample_size, tumor_type, seed, sample_size)

    print(image_paths)
    print(annotations)
    print(features_array)
    print(f"\nfeatures_array.shape: (num_images, num_features)\n{features_array.shape}\n")

    # UMAP dimension reduction on normalized features_array and coloring by accession numbers
    print(f"\nGenerating UMAP for the features of {features_array.shape[0]} images ...")
    umap_embeddings = generate_umap_case(ru.normalization_features_array(features_array), seed, image_paths, save_plot=True, umap_annotation_output_path=umap_annotation_outpath_path)    

    print(f"\n\nCompleted at {utils.get_time()}")
