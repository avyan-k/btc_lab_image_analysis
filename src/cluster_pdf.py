import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib.colors import ListedColormap

import utils
import umap_features as ru

def save_clusters_to_pdf(cluster_info, pdf_output_path):
    """
    Creates a pdf files displaying the image tiles that belong to each k-means cluster
        cluster_info: pandas df with col 'ImagePath' and 'Cluster'

    """
    print("\nCreating PDF with image tiles ...")
    
    # Create a PdfPages object
    with PdfPages(pdf_output_path) as pdf:
        # Get unique clusters from 'Cluster' column in df
        unique_clusters = sorted(cluster_info['Cluster'].unique())
        
        for cluster in unique_clusters:
            # Get the images in the current cluster
            cluster_images_info = cluster_info[cluster_info['Cluster'] == cluster] # subset of cluster_info
            image_paths = cluster_images_info['ImagePath'].tolist() # paths for the images in the cluster

            # Calculate the number of pages needed for this cluster
            images_per_page = 100  # 10 x 10 grid
            num_pages = len(image_paths) // images_per_page + int(len(image_paths) % images_per_page > 0)
            
            for page in range(num_pages):
                # Create a new figure for each page
                fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(8.5, 11))  # standard letter size
                plt.suptitle(f"Cluster {cluster}", fontsize=16)
                
                # Flatten the axes array for easy iteration
                axes = axes.flatten()

                # Get the images for the current page
                page_image_paths = image_paths[page * images_per_page:(page + 1) * images_per_page]

                for i, (ax, image_path) in enumerate(zip(axes, page_image_paths)):
                    # Adjust the file path to remove any redundant directory parts
                    # image_path = image_path.replace(base_image_directory, "").lstrip("/")
                    # full_image_path = os.path.join(base_image_directory, image_path)
                    
                    # Load and display the image
                    # img = Image.open(full_image_path)
                    img = Image.open(image_path)
                    ax.imshow(img)
                    ax.axis('off')  # Hide axes
                    
                    # Get the base filename and remove the .jpg extension
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    
                    # Display the filename without the extension in a smaller font
                    ax.set_title(base_filename, fontsize=3)  # Smaller font size
                
                # Hide any remaining empty subplots
                for j in range(i + 1, len(axes)): # type: ignore
                    axes[j].axis('off')

                # Save the current figure to the PDF with high DPI
                pdf.savefig(fig, dpi=300)  # Increase DPI to 300 for better quality
                plt.close(fig)

    print(f"Clusters saved to {pdf_output_path} as a PDF.")
    return
"""KMEANS CLUSTERING ON UMAP PROJECTION"""
def kmeans_clustering(umap_embedding, image_paths, annotations, seed, n_clusters,save_csv = False, cluster_csv_file_path = ''):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(umap_embedding)
    cluster_info = pd.DataFrame({'ImagePath': image_paths, 'Annotations': annotations,'Cluster': clusters})
    if save_csv:
        assert cluster_csv_file_path != ''
        cluster_info.to_csv(cluster_csv_file_path, index=False)
    return n_clusters, clusters, cluster_info

def plot_umap_for_kmeans(n_clusters, clusters, umap_embedding, umap_kmeans_output_path):
    # Define a custom colormap with distinct colors for each cluster (max 20 colors, 20 clusters)
    num_colors = min(n_clusters, 20)  # Ensure we don't exceed the number of available colors
    colormap = plt.get_cmap('tab20', num_colors)
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
if __name__ == "__main__":
    tumor_type = "SCCOHT_1"
    run_id = f"{utils.get_time()[:10]}"
    seed = 99
    model_type = "ResNet"
    feature_directory = f"./{model_type}_features/{tumor_type}"
    size_of_feature_dataset = ru.get_size_of_dataset(directory=feature_directory, extension='jpg')
    sample_size = 100
    batch_size = 100

    pdf_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}.pdf"
    umap_kmeans_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}_kmeans.png"
    csv_file = f"umap_{tumor_type}_{run_id}_{seed}_{sample_size}.csv"

    results_directory = f"./results/clusters"
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    pdf_output_path = os.path.join(results_directory, pdf_file)
    umap_kmeans_output_path = os.path.join(results_directory, umap_kmeans_file)
    csv_output_path = os.path.join(results_directory, csv_file)

    image_paths, annotations, features_array = ru.get_features_from_disk(size_of_dataset=size_of_feature_dataset,model_type=model_type,tumor_type=tumor_type,seed=seed,sample_size=sample_size)

    n_clusters = 3 if 'DDC_UC' in tumor_type else 2

    umap_embeddings = ru.generate_umap_annotation(ru.normalization_features_array(features_array), seed, annotations,tumor_type,save_plot=False)

    # UMAP dimension reduction with k-means clustering on umap_embeddings in legend
    n_clusters, clusters, cluster_info = kmeans_clustering(umap_embeddings, image_paths,annotations, seed, n_clusters,save_csv= True, cluster_csv_file_path = csv_output_path)
    plot_umap_for_kmeans(n_clusters, clusters, umap_embeddings, umap_kmeans_output_path)
    save_clusters_to_pdf(cluster_info, pdf_output_path)