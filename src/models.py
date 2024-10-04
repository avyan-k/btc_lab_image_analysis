import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from PIL import Image

import utils
import loading_data as ld

DEVICE = utils.load_device()

class Tumor_Classifier(nn.Module):

  def __init__(self,layers, neurons_per_layer,dropout=0.5, input_neurons = 1000, classes = 2):
    super(Tumor_Classifier, self).__init__() 
    self.dropout = dropout
    self.batch1d = nn.BatchNorm1d(input_neurons,track_running_stats=False,affine=False)
    self.network = nn.ModuleList()
    self.network.append(nn.Linear(input_neurons, neurons_per_layer))
    for x in range(layers-1):
        self.network.append(nn.Linear(neurons_per_layer, neurons_per_layer*2))
        neurons_per_layer *= 2
    self.network.append(nn.Linear(neurons_per_layer, classes))

  def forward(self, x):
      x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector
      x = self.batch1d(x)
      for layer in self.network:
          x = F.leaky_relu(layer(x))
          x = F.dropout(x,self.dropout)
      return x

class ResNet_Tumor(Tumor_Classifier):

  def __init__(self,classes = 2):
    super().__init__(
      layers=5,
      neurons_per_layer=64,
      dropout=0,
      input_neurons=1000,
      classes=classes
    )
    self.resnet = timm.create_model('resnet50', pretrained=False)

  def forward(self, x):
    print(x.shape)
    x = self.resnet(x)
    x = super().forward(x)
    return x


if __name__ == "__main__":
	utils.print_cuda_memory()
	seed = 99
	DEVICE = utils.load_device(seed)

	_,transforms = ld.setup_resnet_model(seed) 
	loader, filenames, labels = ld.load_data(1,'./images/DDC_UC_1/images/',transforms = transforms, sample = True, sample_size = 1)
	print(next(loader))

