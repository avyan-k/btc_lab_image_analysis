import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import utils
import umap_features as ru
import loading_data as ld

"""
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
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in list(annotation_colors.values())]
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
    """


def get_cases(image_paths):
    #  case: everything before the last '_' in the base filename
    return [os.path.basename(path).rsplit("_", 1)[0] for path in image_paths]


def generate_umap_cases(embeddings, cases, tumor_type, outfile):
    unique_cases = list(set(cases))
    num_cases = len(unique_cases)
    colormap = plt.get_cmap("rainbow", num_cases)  # colors for cases
    case_colors = colormap(np.linspace(0, 1, num_cases))  # map colors to cases

    plt.figure(figsize=(10, 10))
    # scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=case_indices, cmap=ListedColormap(case_colors), s=0.4)

    # Legnds with case labels
    num_cols_in_legend = 8
    # num_cols_in_legend = 6
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=case_colors[i],
            markersize=10,
            label=f"Case {i+1}",
        )
        for i in range(num_cases)
    ]
    plt.legend(
        handles=handles,
        title="Cases",
        bbox_to_anchor=(0.5, -0.1),
        loc="upper center",
        ncol=num_cols_in_legend,
        fontsize="small",
    )

    plt.title(f"{tumor_type} UMAP Projection of UMAP embeddings - Cases")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.subplots_adjust(
        bottom=0.25
    )  # Increase the bottom margin to fit the legend without squishing the plot
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    # Set up parameters
    seed = 99
    DEVICE = utils.load_device(seed)
    # tumor_type = "DDC_UC_1"
    tumor_type = "SCCOHT_1"
    run_id = f"{utils.get_time()[:10]}"
    model_type = "ResNet"
    feature_directory = f"./features/{model_type}/{tumor_type}"
    size_of_feature_dataset = ld.get_size_of_dataset(
        directory=feature_directory, extension="jpg"
    )
    sample_size = size_of_feature_dataset  # 100
    batch_size = 100

    # Files
    results_directory = "./results/umap-cases"
    Path(os.path.join(results_directory)).mkdir(parents=True, exist_ok=True)
    umap_cases_file = (
        f"umap-cases_{tumor_type}_{model_type}_{run_id}_{seed}_{sample_size}.png"
    )
    umap_cases_output_path = os.path.join(results_directory, umap_cases_file)

    image_paths, feature_loader, classes = ru.get_features_from_disk(
        size_of_dataset=size_of_feature_dataset,
        model_type=model_type,
        tumor_type=tumor_type,
        sample_size=sample_size,
    )
    features_array, annotations = ru.get_features_from_loader(
        feature_loader, classes=classes
    )

    umap_embeddings = ru.generate_umap_annotation(
        feature_loader,
        seed,
        tumor_type,
        save_plot=False,
        tumor_classes=classes,
        normalizer=ru.feature_normalizer(),
    )

    cases = get_cases(image_paths)
    unique_cases = list(set(cases))
    num_cases = len(unique_cases)
    print(f"num of unique_cases: {num_cases}")
    print(f"unique_cases:\n{unique_cases}")

    """
    possibly unecessary below
    """
    # Create a mapping from case labels to integers if necessary
    if isinstance(cases[0], str):
        case_to_index = {case: idx for idx, case in enumerate(unique_cases)}
        case_indices = [case_to_index[case] for case in cases]
    else:
        case_indices = cases  # If already numeric, use as-is

    # Convert sparse matrix to dense array if needed
    if hasattr(umap_embeddings, "toarray"):
        umap_embeddings = umap_embeddings.toarray()  # type: ignore
    """
    possibly unecessary above
    """

    generate_umap_cases(
        embeddings=umap_embeddings,
        cases=cases,
        tumor_type=tumor_type,
        outfile=umap_cases_output_path,
    )
