import torch
from pathlib import Path
import os
import json

import models as md
import utils
import loading_data as ld


def get_torchscript_resnet_tumor(tumor_type, weight_path):
    classes = ld.get_annotation_classes(tumor_type)
    feature_classifier = md.Tumor_Classifier(
        layers=1,
        neurons_per_layer=500,
        dropout=0.5,
        input_neurons=1000,
        classes=len(classes),
    )
    traced_feature_classifier = torch.jit.script(feature_classifier)
    resnet_classifier = md.ResNet_Tumor(
        classes=len(classes), feature_classifier=traced_feature_classifier
    )
    traced_resnet_classifier = torch.jit.script(resnet_classifier)
    print(f"Code for Traced Resnet Tumor:\n {traced_resnet_classifier.code}")
    traced_resnet_classifier.load_state_dict(
        torch.load(weight_path, map_location=torch.device("cpu"), weights_only=True)
    )
    return traced_resnet_classifier


def generate_config_json_resnettumor(tumor_type,means,stds):
    classes = ld.get_annotation_classes(tumor_type)
    dictionnary = {
        "spec_version": "1.0",
        "architecture": "resnet50",
        "num_classes": len(classes),
        "class_names": classes,
        "name": "ResNet_Tumor",
        "patch_size_pixels": 512,
        "spacing_um_px": 0.23,
        "transform": [
            {"name": "Resize", "arguments": {"size": 224}},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "arguments": {
                    "mean": means,
                    "std": stds,
                },
            },
        ],
    }
    return json.dumps(dictionnary, indent=4)


def save_resnettumor_model(model, config, model_path):
    Path(model_path).mkdir(parents=True, exist_ok=True)
    torch.jit.save(
        model, os.path.join(model_path, "torchscript_model.pt")
    )
    with open(os.path.join(model_path, "config.json"), "w") as f:
        f.write(config)
    return

def main(weight_path, tumor_type,normalized = False,proven_mutation=False):
    image_directory = Path(f"./images/{tumor_type}/images")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Cannot find weights {weight_path} to load")
    traced_resnet_classifier = get_torchscript_resnet_tumor(
        tumor_type, weight_path
    )
    model_type = os.path.basename(os.path.splitext(weight_path)[0])
    samples_per_class = int(model_type.split("-")[1])
    means,stds = ld.get_mean_std_per_channel(image_directory,tumor_type,samples_per_class,seed,normalized,proven_mutation)
    config = generate_config_json_resnettumor(tumor_type,means,stds)
    model_path = (
        f"./results/training/torchscript_models/ResNet_Tumor/{model_type}/"
    )
    # print(model_path)
    save_resnettumor_model(traced_resnet_classifier, config, model_path)

if __name__ == "__main__":
    print(torch.__version__)
    seed = 99
    utils.set_seed(seed)
    tumor_type = "DDC_UC_1"
    main(f"./results/training/models/ResNet_Tumor/{tumor_type}-10000-Normalized.pt",tumor_type,True)
    main(f"./results/training/models/ResNet_Tumor/{tumor_type}-3000-Normalized.pt",tumor_type,True,True)



    # _,transforms = ld.setup_resnet_model(seed)
    # format_string = str(transforms.__repr__)
    # transform_string = format_string[format_string.find('(')+1:format_string.rfind(')')].strip()
    # print('a',transform_string,'a')
    # torch.jit.save(traced_resnet_classifier,model_path)
