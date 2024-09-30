import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

import utils
import umap_features as ru
import loading_data as ld

def save_clusters_to_pdf(cluster_info, outfile):
    '''
    Creates a pdf files displaying the image tiles that belong to each cluster
    Arguments
    cluster_info: pandas DataFrame containing col 'ImagePath' and 'Cluster'
    outfile: str of pdf file path to save
    '''
    print("\nCreating PDF with image tiles ...")
    
    # Create a PdfPages object
    with PdfPages(outfile) as pdf:
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
                    # Load and display the image
                    img = Image.open(image_path)
                    ax.imshow(img)
                    ax.axis('off')  # Hide axes
                    
                    base_filename = os.path.splitext(os.path.basename(image_path))[0] # get base filename without extension
                    ax.set_title(base_filename, fontsize=3) # display base filename
                
                # Hide any remaining empty subplots
                for j in range(i + 1, len(axes)): # type: ignore
                    axes[j].axis('off')

                pdf.savefig(fig, dpi=300) # save currrent figure to PDF, dpi quality
                plt.close(fig)

    print(f"Clusters saved to {outfile} as a PDF.")
    return

"""CLUSTERING ON UMAP PROJECTION"""
def kmeans_clustering(umap_embeddings, seed, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(umap_embeddings)
    return clusters

def agglomerative_clustering(umap_embeddings, n_clusters):
    '''
    Agglomerative clustering with provided n_clusters and Ward linkage
    '''
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = agg_clustering.fit_predict(umap_embeddings)
    return clusters

def agglomerative_clustering_no_n_clusters(umap_embeddings, distance_threshold=0):
    '''
    Agglomerative clustering without n_clusters, uses distance_threshold to form clusters
    '''
    agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    clusters = agg_clustering.fit_predict(umap_embeddings)
    return clusters

def dbscan_clustering(umap_embeddings, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(umap_embeddings)
    return clusters

def get_clusters(cluster_type, embeddings, seed, n_clusters=10, eps=0.4, min_samples=500):
    '''
    Apply chosen clustering algorithm on embeddings
    return
    clusters: numpy.darray with cluster label associated with each embeddings (umap data point)
    '''
    print(f"Getting {cluster_type} clusters...")
    if cluster_type == "k-means":
        clusters = kmeans_clustering(embeddings, seed, n_clusters)
    elif cluster_type == "agglomerative-clustering":
        clusters = agglomerative_clustering_no_n_clusters(embeddings, distance_threshold=distance_threshold)
        n_clusters = len(set(clusters)) # TODO verify is this is accurate
    elif cluster_type == f"agglomerative-clustering-{str(n_clusters)}-clusters":
        clusters = agglomerative_clustering(embeddings, n_clusters)
    elif cluster_type == "dbscan":
        clusters = dbscan_clustering(embeddings, eps=eps, min_samples=min_samples)
        n_clusters = len(set(clusters)) # note: DBSCAN assigns -1 to noise points, this will still be considered as first cluster
    else:
        print(f"cluster_type: {cluster_type} is invalid. dbscan clustering is used with eps=0.4 and min_samples=500 (ddc-uc settings).")
        clusters = dbscan_clustering(umap_embeddings, eps=eps, min_samples=min_samples) #ddc_uc settings

    return clusters

def plot_umap_for_kmeans(cluster_type, n_clusters, clusters, umap_embeddings, outfile):
    # Colormap for clusters
    if n_clusters >= 20:
        colormap = plt.get_cmap('nipy_spectral', n_clusters) # 20+ clusters
    elif n_clusters >= 10:
      colormap = plt.get_cmap('tab20', n_clusters) # 10-19 clusters
    else:
        colormap = plt.get_cmap('tab10', n_clusters) # under 10 clusters

    cluster_colors = colormap(np.linspace(0, 1, n_clusters)) # assign colors to clusters

    # Plot parameters
    plt.figure(figsize=(10, 10))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=clusters, cmap=ListedColormap(cluster_colors), s=0.4)

    # legend
    num_cols_in_legend = 5
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[i], markersize=10, label=f'Cluster {i}') for i in range(n_clusters)]
    plt.legend(handles=handles, title='Clusters', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=num_cols_in_legend, fontsize='small')

    plt.title(f'{tumor_type} UMAP Projection with {cluster_type} clustering on UMAP embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.subplots_adjust(bottom=0.25)  # bottom margin
    plt.savefig(outfile, bbox_inches='tight')
    # plt.show()
    return

def get_cases(image_paths):
    #  case: everything before the last '_' in the base filename
    return [os.path.basename(path).rsplit('_', 1)[0] for path in image_paths]

"""OUTPUT CSV FILE"""
def save_cluster_csv(image_paths, cases, annotations, clusters, save_csv = False, outfile = ''):
    cluster_info = pd.DataFrame({'ImagePath': image_paths, 'Case': cases, 'Annotations': annotations,'Cluster': clusters})
    if save_csv:
        assert outfile != ''
        cluster_info.to_csv(outfile, index=False)
    return cluster_info

if __name__ == "__main__":
    # Parameters    
    tumor_type = "SCCOHT_1"
    # tumor_type = "DDC_UC_1"
    run_id = f"{utils.get_time()[:10]}"
    seed = 99
    model_type = "ResNet"
    feature_directory = f"./features/{model_type}/{tumor_type}"
    size_of_feature_dataset = ld.get_size_of_dataset(directory=feature_directory, extension='jpg')
    sample_size = size_of_feature_dataset # 100
    batch_size = 100

    """
    n_clusters: determined based on umap
    cluster_type: 
        - k-means
        - agglomerative-clustering
        - agglomerative-clustering-{str(n_clusters)}-clusters
        - dbscan
    """
    cluster_type = "dbscan"

    # File paths
    pdf_file = f"umap_{tumor_type}_{model_type}_{cluster_type}_{run_id}_{seed}_{sample_size}.pdf"
    umap_kmeans_file = f"umap_{tumor_type}_{model_type}_{cluster_type}_{run_id}_{seed}_{sample_size}.png"
    csv_file = f"umap_{tumor_type}_{model_type}_{cluster_type}_{run_id}_{seed}_{sample_size}.csv"

    results_directory = f"./results/clusters"
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    pdf_output_path = os.path.join(results_directory, pdf_file)
    umap_kmeans_output_path = os.path.join(results_directory, umap_kmeans_file)
    csv_output_path = os.path.join(results_directory, csv_file)


    # Clustering analysis
    image_paths, feature_loader, classes = ru.get_features_from_disk(size_of_dataset=size_of_feature_dataset,model_type=model_type,tumor_type=tumor_type,sample_size=sample_size)
    features_array, annotations = ru.get_features_from_loader(feature_loader,classes=classes)

    umap_embeddings = ru.generate_umap_annotation(feature_loader, seed, tumor_type,save_plot=False, tumor_classes=classes,normalizer=ru.feature_normalizer())
    
    # Clustering of UMAP embeddings
    n_clusters = 10 # kmeans_clustering, agglometative clustering
    distance_threshold=0 # agglomerative clustering without n_clusters
    eps=0.4 # dbscan ddc-uc
    min_samples=500 # dbscan ddc-uc
    # eps=0.3 # dbscan sccoht
    # min_samples=50 # dbscan sccoht

    clusters = get_clusters(
                            cluster_type=cluster_type,
                            embeddings=umap_embeddings,
                            seed=seed,
                            n_clusters=n_clusters,
                            eps=eps,
                            min_samples=min_samples
                            )
    
    
    print("Plotting clustering umap...")
    plot_umap_for_kmeans(cluster_type=cluster_type, n_clusters=n_clusters, clusters=clusters, umap_embeddings=umap_embeddings, outfile=umap_kmeans_output_path)

    cluster_info = save_cluster_csv(image_paths, get_cases(image_paths), annotations, clusters, save_csv=True, outfile=csv_output_path)