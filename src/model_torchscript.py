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
        layers=5,
        neurons_per_layer=64,
        dropout=0.0,
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


def generate_config_json_resnettumor(tumor_type):
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
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            },
        ],
    }
    return json.dumps(dictionnary, indent=4)


def save_resnettumor_model(path, config, model_path):
    Path(model_path).mkdir(parents=True, exist_ok=True)
    torch.jit.save(
        traced_resnet_classifier, os.path.join(model_path, "torchscript_model.pt")
    )
    with open(os.path.join(model_path, "config.json"), "w") as f:
        f.write(config)
    return


if __name__ == "__main__":
    print(torch.__version__)
    seed = 99
    utils.set_seed(seed)
    samples_per_case = 10000

    for tumor_type in os.listdir("./images"):
        print(tumor_type)
        if tumor_type in [".DS_Store", "__MACOSX", "SIL"]:
            continue
        norm_weight_path = (
            f"./results/training/models/ResNet_Tumor/{tumor_type}-{samples_per_case}-Normalized.pt"
        )
        unnorm_weight_path = (
            f"./results/training/models/ResNet_Tumor/{tumor_type}-{samples_per_case}-Unnormalized.pt"
        )
        for weight_path in [norm_weight_path, unnorm_weight_path]:
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(f"Cannot find weights {weight_path} to load")
            traced_resnet_classifier = get_torchscript_resnet_tumor(
                tumor_type, weight_path
            )
            config = generate_config_json_resnettumor(tumor_type)
            model_type = os.path.basename(os.path.splitext(weight_path)[0])
            model_path = (
                f"./results/training/torchscript_models/ResNet_Tumor/{model_type}/"
            )
            print(model_path)
            save_resnettumor_model(traced_resnet_classifier, config, model_path)

    # _,transforms = ld.setup_resnet_model(seed)
    # format_string = str(transforms.__repr__)
    # transform_string = format_string[format_string.find('(')+1:format_string.rfind(')')].strip()
    # print('a',transform_string,'a')
    # torch.jit.save(traced_resnet_classifier,model_path)
