"""
To get
    - precision
    - accuracy
    - recall
    - f1-score 

For testing purposes: reads & converts .csv file into pandas dataframe
* Needs to take pandas df directly as an argument instead when integrated with 'resnet_umap.py'

"""

import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


# Create a dictionary mapping cluster # to a predicted class (normal/tumor or normal/undiff/well_diff)
def create_cluster2pred_mapping(n_clusters, tumor_type):
    cluster_to_pred = {}
    if "DDC_UC" in tumor_type:
        print("Assign 'normal', 'undiff' or well_diff' to each cluster.")
    else:
        print("Assign 'normal' or 'tumor' to each cluster.")

    for i in range(n_clusters):
        while True: # will break if valid input is given, otherwise stays in loop to ask for input again
            pred = input(f"Cluster {i} is: ").strip().lower() # gets which class each cluster corresponds to (normal/tumor (or undiff, well_diff))
            if "DDC_UC" in tumor_type:
                if pred in ['normal', 'undiff', 'well_diff']: # checks if input is valid
                    cluster_to_pred[i] = pred
                    break
                else:
                    print("Invalid input. Please enter 'normal', 'undiff' or 'well_diff'.")
            else:
                if pred in ['normal', 'tumor']:
                    cluster_to_pred[i] = pred
                    break
                else:
                    print("Invalid input. Please enter 'normal' or 'tumor'.")
    return cluster_to_pred


def stats_validation(csv_file, n_clusters, tumor_type, output_file=None):
    df = pd.read_csv(csv_file) # temporary until testing is done and dataframe is passed as an argument
    cluster_to_pred = create_cluster2pred_mapping(n_clusters, tumor_type)
    print(f"\ncluster_pred map:{cluster_to_pred}")

    df['Predicted'] = df['Cluster'].map(cluster_to_pred) # added 'Predicted' col to df using the mapping from cluster to predicted class
    if output_file:
        df.to_csv(output_file, index=False)

    # Calculate accuracy, precision, recall and f1 metrics
    accuracy = accuracy_score(df['Annotations'], df['Predicted'])
    # multi-class classification with Weighted-averaging: Useful when you have class imbalance and want to give more importance to the performance on larger classes
    if "DDC_UC" in tumor_type:
        precision = precision_score(df['Annotations'], df['Predicted'], average='weighted')
        recall = recall_score(df['Annotations'], df['Predicted'], average='weighted')
        f1 = f1_score(df['Annotations'], df['Predicted'], average='weighted')

    # binary classification postive and negative classes for SCCOHT and vMRT
    else: 
        precision = precision_score(df['Annotations'], df['Predicted'], pos_label='tumor')
        recall = recall_score(df['Annotations'], df['Predicted'], pos_label='tumor')
        f1 = f1_score(df['Annotations'], df['Predicted'], pos_label='tumor')

    return accuracy, precision, recall, f1

if __name__ == "__main__":

    csv_file = "./results/umap_SCCOHT_1_2024-08-28_disk_k8_99_106039.csv"
    n_clusters = 8
    tumor_type = "SCCOHT_1"

    output_file = f"./results/umap_SCCOHT_1_2024-08-28_disk_k8_99_106039_predictions.csv"

    accuracy, precision, recall, f1 = stats_validation(csv_file, n_clusters, tumor_type, output_file)
    
    print(f"\n\nPrecision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")






