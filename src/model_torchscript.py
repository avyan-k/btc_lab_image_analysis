import torch
from pathlib import Path
import os

import models as md
import utils
import loading_data as ld


if __name__ == "__main__":
    print(torch.__version__)
    seed = 99
    DEVICE = utils.load_device(seed)
    torch.manual_seed(seed)
    tumor_type = "DDC_UC_1"

    feature_classifier = md.Tumor_Classifier(layers=5, neurons_per_layer=64, dropout=0.0, input_neurons=1000, classes=3) 
    traced_feature_classifier = torch.jit.script(feature_classifier)
    resnet_classifier = md.ResNet_Tumor(classes=3,feature_classifier=traced_feature_classifier)
    # resnet_classifier.load_state_dict(torch.load('results/training/models/ResNet_Tumor/DDC_UC_1/4-257-0.9653239250183105.pt', map_location = DEVICE,weights_only=True))

    _,transforms = ld.setup_resnet_model(seed) 
    loader, _,_,_ = ld.load_data(1,'./images/DDC_UC_1/images/',transforms = transforms, sample = True, sample_size = 1)
    x = next(iter(loader))[0]
    
    traced_resnet_classifier = torch.jit.script(resnet_classifier)
    print(traced_resnet_classifier.code)
    print(traced_resnet_classifier(x),resnet_classifier(x))

    weight_path = './results/training/models/ResNet_Tumor/DDC_UC_1/1-771-0.9662657976150513.pt'
    weight_name = os.path.basename(weight_path)
    traced_resnet_classifier.load_state_dict(torch.load(weight_path, map_location = DEVICE,weights_only=True))

    model_path = f"./results/training/torchscript_models/{str(type(resnet_classifier).__name__)}/{tumor_type}/{weight_name}"
    print(os.path.dirname(model_path))
    Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced_resnet_classifier,model_path)
