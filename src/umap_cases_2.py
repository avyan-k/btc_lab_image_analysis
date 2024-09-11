import os
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
from pathlib import Path

import loading_data as ld
import utils
import umap_features as ru


def generate_umap_annotation(features, seed, annotations, tumor_type, save_plot = False, umap_output_path = ''):
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=seed)
    umap_embedding = umap_model.fit_transform(features)

    print(f"\ndim of umap_embedding:\t{umap_embedding.shape}\n")

    # Mapping annotations to colors
    if 'DDC_UC' in tumor_type:
        annotation_colors = {'normal': 'blue', 'undiff': 'red', 'well_diff' : 'green'}
    elif 'ALL' in tumor_type:
        annotation_colors = {
            'ddc_normal':'blue',
            'sccoht_normal':'turquoise',
            'vmrt_normal':'cyan',
            'ddc_undiff': 'red',
            'ddc_well_diff':'green',
            'sccoht_tumor':'fuchsia',
            'vmrt_tumor':'purple'
        }
    else:
        annotation_colors = {'normal': 'blue', 'tumor': 'red'}
        

    colors = [annotation_colors[annotation] for annotation in annotations]

    
    # Generating figure
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=colors, s=5, alpha=0.7)

    #legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in list(annotation_colors.values())]
    if 'DDC_UC' in tumor_type:
        legend_labels = ['normal', 'undiff', 'well_diff']
    elif 'ALL' in tumor_type:
        legend_labels = list(annotation_colors.keys())
    else:
        legend_labels = ['normal', 'tumor']

    plt.legend(handles, legend_labels, title='Annotations')

    plt.title(f'{tumor_type} UMAP Projection')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    if save_plot:
        assert umap_output_path != ''
        plt.savefig(umap_output_path)
    plt.show()
    return umap_embedding


def get_cases(image_paths):
    #  case: everything before the last '_' in the base filename
    cases = [os.path.basename(path).rsplit('_', 1)[0] for path in image_paths]

    for case in cases:
        print(case)
    print(len(cases))
    unique_cases = list(set(cases))
    print(f"num of unique_cases: {len(unique_cases)}")


    return cases


def generate_umap_cases(features, seed, cases, tumor_type, save_plot=False, umap_output_path=''):
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=seed)
    umap_embedding = umap_model.fit_transform(features)

    # Mapping cases to colors




    return






if __name__ == "__main__":
    # Set up parameters
    run_id = f"{utils.get_time()[:10]}"
    tumor_type = "ALL"  
    seed = 99
    DEVICE = utils.load_device(seed)
    size_of_image_dataset = ru.get_size_of_dataset(tumor_type,extension='jpg')
    size_of_feature_dataset = ru.get_size_of_dataset(tumor_type,extension='npz')
    sample_size = 100
    batch_size = 100
    model_type = "ResNet"
    # Paths to directories
    image_directory = f"./images/{tumor_type}/images"
    feature_directory = f"./{model_type}_features/{tumor_type}"
    results_directory = f"./results/umap"
    Path(os.path.join(results_directory)).mkdir(parents=True, exist_ok=True)
    
    # Output file
    umap_case_file = f"umap_case_{tumor_type}_{run_id}_{seed}_{sample_size}.png" # filename
    umap_case_outpath_path = os.path.join(results_directory, umap_case_file) # file path

    umap_annotation_file = f"umap_annotation_umap-unnormalized_{tumor_type}_{run_id}_{seed}_{sample_size}.png" # filename
    umap_annotation_outpath_path = os.path.join(results_directory, umap_annotation_file) # file path

    # Seed for reproducibility
    np.random.seed(seed) # numpy random seed

    # ResNet50 model
    print("\nSetting up ResNet model ...")
    model,_ = ld.setup_resnet_model(seed)
    model.eval()
    
    # Retrieve features from disk (numpy arrays)
    image_paths, annotations, features_array = ru.get_features_from_disk(size_of_dataset=size_of_feature_dataset,model_type=model_type,tumor_type=tumor_type,seed=seed,sample_size=sample_size)
    print(f"\nfeatures_array.shape: (num_images, num_features)\n{features_array.shape}\n")

    # UMAP dimension reduction on normalized features_array and coloring by annotations
    print(f"\nGenerating UMAP for the features of {features_array.shape[0]} images ...")
    

    # get cases
    cases = get_cases(image_paths)

    #umap_embeddings = generate_umap_case(ru.normalization_features_array(features_array), seed, cases,tumor_type, save_plot = True, umap_output_path = umap_case_outpath_path)    

    # umap_embeddings = generate_umap_annotation(ru.normalization_features_array(features_array), seed, annotations,tumor_type, save_plot = True, umap_output_path = umap_annotation_outpath_path)    
    umap_embeddings = generate_umap_annotation(ru.normalization_features_array(features_array), seed, annotations,tumor_type, save_plot = False, umap_output_path = umap_annotation_outpath_path)    

    print(f"\n\nCompleted at {utils.get_time()}")