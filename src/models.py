import torch
import torch.nn as nn
import timm
from huggingface_hub import login, hf_hub_download
import torch.nn.functional as F
from uni import get_encoder
import os

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

class UNI_Tumor(nn.Module):
    def __init__(self, classes=2, feature_classifier=None, UNI_pretrained = False):
        super(UNI_Tumor, self).__init__()
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
            self.uni = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True, pretrained=False
                )
            if UNI_pretrained:
                try:
                    local_dir = "./assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
                    self.uni.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
                except FileNotFoundError:
                    download_UNI_model_weights()
    def forward(self, x):
        x = self.uni(x)
        x = self.fc(x)
        return x

def download_UNI_model_weights():
    try:
        with open("hugging_face.txt") as f:
            token = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            "Access token for hugging fase not found. Save as ./hugging_face.txt in root directory of project"
        )
    login(token)
    model_dir = "./assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    os.makedirs(model_dir, exist_ok=True)
    hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=model_dir, force_download=True)

if __name__ == "__main__":
    model = ResNet_Tumor()
    utils.print_cuda_memory()
    seed = 99
    DEVICE = utils.load_device(seed)
    if str(DEVICE) != "cpu":
        model, transform = get_encoder(enc_name="uni", device=DEVICE)
    classifier = Tumor_Classifier(
                        layers=5,
                        neurons_per_layer=64,
                        dropout=0,
                        input_neurons=1000,
                        classes=2,
                    )
    classifier.load_state_dict(torch.load("x.pt",map_location='cpu',weights_only=True))