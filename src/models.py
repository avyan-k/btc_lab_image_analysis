import torch
import torch.nn as nn
import timm
from huggingface_hub import login, hf_hub_download
import torch.nn.functional as F
try:
    from uni import get_encoder
except ModuleNotFoundError:
    raise Warning("UNI module not found, model cannot be loaded. Please install at https://github.com/mahmoodlab/UNI")
import os
from torchinfo import summary
import torchvision.models as torchmodels


import utils

"""Custom ResNet Model"""
class Tumor_Classifier(nn.Module):
    def __init__(
        self, input_neurons, classes,layers=1, neurons_per_layer=500, dropout=0.5
    ):
        """
        If apply_softmax is set to True, a final softmax activation layer is applied to output class probabilities.
        If softmax is applied to the model output after inference, then apply_softmax should be set to False (softmax is NOT idempotent)
        """
        super(Tumor_Classifier, self).__init__()
        self.dropout = dropout
        self.batch1d = nn.BatchNorm1d(
            input_neurons, track_running_stats=False, affine=False
        )
        self.first_activation = nn.LeakyReLU(inplace=True) # input is a bunch of linear features, so activate them
        self.network = nn.ModuleList()
        self.network.append(nn.Linear(input_neurons, neurons_per_layer))
        self.network.append(nn.Dropout(self.dropout,inplace=True))
        for _ in range(layers - 1):
            self.network.append(nn.LeakyReLU(inplace=True)) 
            self.network.append(nn.Linear(neurons_per_layer, neurons_per_layer // 2))
            self.network.append(nn.Dropout(self.dropout,inplace=True))
            neurons_per_layer //= 2
        self.network.append(nn.LeakyReLU(inplace=True))
        self.network.append(nn.Linear(neurons_per_layer, classes))


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten to a 1D vector
        if x.shape[0] > 1:
            x = self.batch1d(x)
        x = self.first_activation(x)
        for layer in self.network:
            x = layer(x)
        return x # last layer no activation
    
    def add_softmax(self):
        if not isinstance(self.network[-1],nn.Softmax):
            self.network.append(nn.Softmax(dim=1))


class ResNet_Tumor(nn.Module):
    def __init__(self, classes=2, training = False, feature_classifier=None):
        """if training is set to True, softmax is not applied to model output"""
        super(ResNet_Tumor, self).__init__()
        if feature_classifier is None:
            self.fc = Tumor_Classifier(input_neurons=1000,classes=classes)
            if not training:
                self.fc.add_softmax()
        else:
            self.fc = feature_classifier
        self.resnet = timm.create_model("resnet18", pretrained=False)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
    
    def save_with_softmax(self,save_path:str):
        self.fc.add_softmax()
        torch.save(self.state_dict(),save_path)



"""UNI Model"""
class UNI_Tumor(nn.Module):
    def __init__(self, classes=2, training = False, feature_classifier=None, pretrained = False):
        super(UNI_Tumor, self).__init__()
        if feature_classifier is None:
            self.fc = Tumor_Classifier(input_neurons=1024,classes=classes)
            if not training:
                self.fc.add_softmax()
        else:
            self.fc = feature_classifier
        self.uni, _ = get_encoder(enc_name="uni",device="cpu")
        assert self.uni is not None
        if pretrained:
            load_uni_pretrained_weights(self.uni)
            for parameter in self.uni.parameters():
                parameter.requires_grad = False
        
    def forward(self, x):
        x = self.uni(x)
        x = self.fc(x)
        return x


def get_uni_model(classes, device, pretrained = False):
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=classes, dynamic_img_size=True, pretrained=False
        )
    if pretrained:
        load_uni_pretrained_weights(model,strict=False)
    return model

def load_uni_pretrained_weights(model,strict=True):
    try:
        local_dir = "./assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin")), strict=strict)
    except FileNotFoundError:
        download_UNI_model_weights()

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

"""Pre-Trained Model"""
def get_resnet50_model():
    model = torchmodels.resnet50(weights=torchmodels.ResNet50_Weights.DEFAULT)
    model.eval()
    return model

def get_VGG16_model():
    model = torchmodels.vgg16(weights=torchmodels.VGG16_Weights.DEFAULT)
    model.eval()
    model.classifier = nn.Identity()
    return model

if __name__ == "__main__":
    model = ResNet_Tumor()
    utils.print_cuda_memory()
    seed = 99
    DEVICE = utils.load_device(seed)
    x = torch.rand((1, 3, 224, 224)).to(DEVICE)

    # resnet_tumor = ResNet_Tumor(classes=2)
    # uni_tumor = UNI_Tumor(classes=2,pretrained=True, feature_classifier=nn.Identity())
    # uni_tumor = uni_tumor.to(DEVICE)
    # summary(uni_tumor,(1, 3, 224, 224))
    # x = uni_tumor(x)
    # print(x.shape)
    x = torch.rand((1, 3, 224, 224)).to(DEVICE)
    model = ResNet_Tumor(classes=3)
    # model.load_state_dict(torch.load("./results/training/models/ResNet_Tumor/DDC_UC_1-10000-Normalized.pt"))
    # model = get_VGG16_model()
    model = model.to(DEVICE)
    # model.fc = nn.Identity()
    summary(model)
    x = model(x)
    print(x)
