"""
To get
    - precision
    - accuracy
    - recall
    - f1-score

For testing purposes: reads & converts .csv file into pandas dataframe
"""

import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


def create_cluster2pred_mapping(n_clusters, tumor_type):
    """
    Create a dict mapping cluster label to a predicted class
    """
    cluster_to_pred = {}
    if "DDC_UC" in tumor_type:
        print(
            "\n\nAssign 'normal', 'undiff', 'well_diff', 'noise' or 'exclude' to each cluster."
        )
    else:
        print("\n\nAssign 'normal', 'tumor', 'noise' or 'exclude' to each cluster.")

    for i in range(-1, n_clusters - 1):
        while True:  # will break if valid input is given, otherwise stays in loop to ask for input again
            pred = (
                input(f"Cluster {i} is: ").strip().lower()
            )  # gets which class each cluster corresponds to (normal/tumor (or undiff, well_diff), noise, exclude)
            if "DDC_UC" in tumor_type:
                if pred in [
                    "normal",
                    "undiff",
                    "well_diff",
                    "noise",
                    "exclude",
                ]:  # checks if input is valid
                    cluster_to_pred[i] = pred
                    break
                else:
                    print(
                        "Invalid input. Please enter 'normal', 'undiff', 'well_diff', 'noise' or 'exclude'."
                    )
            else:
                if pred in ["normal", "tumor", "noise", "exclude"]:
                    cluster_to_pred[i] = pred
                    break
                else:
                    print(
                        "Invalid input. Please enter 'normal', 'tumor', 'noise' or 'exclude'."
                    )
    return cluster_to_pred


def stats_validation(csv_file, n_clusters, tumor_type, output_file=None):
    df = pd.read_csv(
        csv_file
    )  # temporary until testing is done and dataframe is passed as an argument
    cluster_to_pred = create_cluster2pred_mapping(
        n_clusters, tumor_type
    )  # maps cluster label to predicted class
    print(f"\ncluster_pred map:{cluster_to_pred}")

    df["Predicted"] = df["Cluster"].map(
        cluster_to_pred
    )  # added 'Predicted' col to df using the mapping from cluster to predicted class

    # Remove rows where 'Predicted' is 'noise' or 'exclude'
    df = df[~df["Predicted"].isin(["noise", "exclude"])]

    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to output file {output_file}")

    # Calculate accuracy, precision, recall and f1 metrics
    accuracy = accuracy_score(df["Annotations"], df["Predicted"])

    # multi-class classification with Weighted-averaging: Useful when you have class imbalance and want to give more importance to the performance on larger classes
    if "DDC_UC" in tumor_type:
        precision = precision_score(
            df["Annotations"], df["Predicted"], average="weighted"
        )
        recall = recall_score(df["Annotations"], df["Predicted"], average="weighted")
        f1 = f1_score(df["Annotations"], df["Predicted"], average="weighted")

    # binary classification postive and negative classes for SCCOHT and vMRT
    else:
        precision = precision_score(
            df["Annotations"], df["Predicted"], pos_label="tumor"
        )
        recall = recall_score(df["Annotations"], df["Predicted"], pos_label="tumor")
        f1 = f1_score(df["Annotations"], df["Predicted"], pos_label="tumor")

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    """ SCCOHT """
    tumor_type = "SCCOHT_1"
    n_clusters = 13
    csv_file = "./results/clusters/umap_SCCOHT_1_ResNet_dbscan_2024-09-21_99_good.csv"
    output_file = "./results/clusters/umap_SCCOHT_1_ResNet_dbscan_2024-09-21_99_good_prediction.csv"

    # ''' DDC-UC '''
    # tumor_type = "DDC-UC_1"
    # n_clusters = 7
    # csv_file = "./results/clusters/umap_DDC_UC_1_ResNet_dbscan_2024-09-19_99_1_good.csv"
    # output_file = f"./results/clusters/umap_DDC_UC_1_ResNet_dbscan_2024-09-19_99_1_good_prediction.csv"

    accuracy, precision, recall, f1 = stats_validation(
        csv_file, n_clusters, tumor_type, output_file
    )

    print(f"\n\nPrecision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
