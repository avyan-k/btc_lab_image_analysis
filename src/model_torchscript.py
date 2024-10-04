import torch

import models as md
import utils


if __name__ == "__main__":
    print(torch.__version__)
    seed = 99
    DEVICE = utils.load_device(seed)
    torch.manual_seed(seed)

    resnet_classifier = md.ResNet_Tumor(classes=3)