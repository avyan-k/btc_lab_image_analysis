import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

import utils


class Tumor_Classifier(nn.Module):
    def __init__(
        self, layers, neurons_per_layer, dropout=0.5, input_neurons=1000, classes=2
    ):
        super(Tumor_Classifier, self).__init__()
        self.dropout = dropout
        self.batch1d = nn.BatchNorm1d(
            input_neurons, track_running_stats=False, affine=False
        )
        self.network = nn.ModuleList()
        self.network.append(nn.Linear(input_neurons, neurons_per_layer))
        for x in range(layers - 1):
            self.network.append(nn.Linear(neurons_per_layer, neurons_per_layer * 2))
            neurons_per_layer *= 2
        self.network.append(nn.Linear(neurons_per_layer, classes))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten to a 1D vector
        if x.shape[0] > 1:
            x = self.batch1d(x)
        for layer in self.network:
            x = F.leaky_relu(layer(x))
            x = F.dropout(x, self.dropout)
        return x


class ResNet_Tumor(nn.Module):
    def __init__(self, classes=2, feature_classifier=None):
        super(ResNet_Tumor, self).__init__()
        if feature_classifier is None:
            self.fc = Tumor_Classifier(
                layers=5,
                neurons_per_layer=64,
                dropout=0,
                input_neurons=1000,
                classes=classes,
            )
        else:
            self.fc = feature_classifier
        self.resnet = timm.create_model("resnet18", pretrained=False)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ResNet_Tumor()
    utils.print_cuda_memory()
    seed = 99
