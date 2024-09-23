import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torchmetrics
import os
import sys
from pathlib import Path
import time
from datetime import datetime
from torchinfo import summary

import loading_data as ld
import utils
class Tumor_Classifier(nn.Module):

  def __init__(self,layers, neurons_per_layer,dropout=0.5, input_neurons = 1000, classes = 2):
    super(Tumor_Classifier, self).__init__() 
    self.dropout = dropout
    self.network = nn.ModuleList()
    self.network.append(nn.Linear(input_neurons, neurons_per_layer))
    for x in range(layers-1):
        self.network.append(nn.Linear(neurons_per_layer, neurons_per_layer*2))
        neurons_per_layer *= 2
    self.network.append(nn.Linear(neurons_per_layer, classes))

  def forward(self, x):
      x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector
      x = (F.batch_norm(x.T, training=True,running_mean=torch.zeros(x.shape[0]).to(DEVICE),running_var=torch.ones(x.shape[0]).to(DEVICE))).T
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
      x = self.resnet(x)
      x = super().forward(x)
      return x

def train_model(model,tumor_type,input_shape,train_loader,valid_loader, train_count,valid_count,num_epochs = 200,number_of_validations = 3,learning_rate = 0.001, weight_decay=0.001):
  
  losses = np.empty(num_epochs)
  start = time.time()
  Path(f'./results/training/models/{str(type(model).__name__)}/{tumor_type}').mkdir(parents=True, exist_ok=True)
  text_file = open(f"./results/training/losses_{str(type(model).__name__)}_{tumor_type}.txt", "w",encoding="utf-8")
  text_file.write(f"Attempting {num_epochs} epochs on date of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n with model:")
  model_stats = summary(model, (1,*input_shape), verbose=0)
  text_file.write(f"Model Summary:{str(model_stats)}\n")
  text_file.write('\n')
  text_file.close()

  model = model.to(DEVICE)

  # Initializes the Adam optimizer with the model's parameters
  optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
  train_loss_function = nn.CrossEntropyLoss(weight=ld.count_dict_tensor(train_count)).to(DEVICE)
  valid_loss_function = nn.CrossEntropyLoss(weight=ld.count_dict_tensor(valid_count)).to(DEVICE)
  accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(train_count.keys())).to(DEVICE)

  val_iteration = len(train_loader) // number_of_validations
  for epoch in tqdm(range(num_epochs),desc="Epoch",position=3,leave=False):
    for iteration, (X_train, y_train) in enumerate(tqdm((train_loader),desc="Iteration",position=4,leave=False)):
      # resets all gradients to 0 after each batch
      optimizer.zero_grad()

      X_train = X_train.to(DEVICE)
      y_train = y_train.to(DEVICE)
      # forward pass of the CNN model on the input data to get predictions
      y_hat = model(X_train)
      
      # comparing the model's predictions with the truth labels
      train_loss = train_loss_function(y_hat, y_train)
      
      # backpropagating the loss through the model
      train_loss.backward()

      # takes a step in the direction that minimizes the loss
      optimizer.step()

      # checks if should compute the validation metrics for plotting later
      if iteration % val_iteration == 0 and epoch % 5 == 0:
        done = valid_model(model,tumor_type,valid_loader,epoch,iteration,accuracy,valid_loss_function)
        
        if done:
          with open(f"./results/training/losses_{str(type(model).__name__)}_{tumor_type}.txt", "a") as f:
            f.write(f"\n3 satisfying models trained\n")
            f.write(f"Losses: \n{losses}\n")
            return losses

    # logging results
    logging_result(model,train_loss,epoch,start,losses)

  with open(f"./results/training/losses_{str(type(model).__name__)}_{tumor_type}.txt", "a") as f:
    f.write(f"Losses: \n{losses}\n")
    return losses

def valid_model(model,tumor_type,valid_loader,epoch,iteration,accuracy,loss):

  # stops computing gradients on the validation set
  with torch.no_grad():

    # Keep track of the losses & accuracies
    val_accuracy_sum = 0
    val_loss_sum = 0

    # Make a predictions on the full validation set, batch by batch
    for X_val, y_val in tqdm(valid_loader,desc="Validation Iteration",position=5,leave=False):

      # Move the batch to GPU if it's available
      X_val = X_val.to(DEVICE)
      y_val = y_val.to(DEVICE)

      y_hat = model(X_val)
      val_accuracy_sum += accuracy(y_hat, y_val)
      val_loss_sum += loss(y_hat, y_val)

    # Divide by the number of iterations (and move back to CPU)
    val_accuracy = (val_accuracy_sum / len(valid_loader)).cpu()
    val_loss = (val_loss_sum / len(valid_loader)).cpu()

    # Store the values in the dictionary
    # Out to console
    text_file = open(f"./results/training/losses_{str(type(model).__name__)}_{tumor_type}.txt", "a") 
    text_file.write(f"\nEPOCH = {epoch} --- ITERATION = {iteration}\n")
    text_file.write(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}\n\n")
    text_file.close()
    # print(f"EPOCH = {epoch} --- ITERATION = {iteration}")
    # print(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}")
    if val_accuracy > 0.95:
      torch.save(model.state_dict(), f"./results/training/models/{str(type(model).__name__)}/{tumor_type}/{epoch}-{iteration}-{val_accuracy}.pt")
      if len([name for name in os.listdir(f"./results/training/models/{str(type(model).__name__)}/{tumor_type}")]) > 5:
        return True
    return False

def logging_result(model,loss,epoch,start,losses):
  
  training_loss = loss.cpu()
  losses[epoch] = float(training_loss)
  # print(f"\n\nloss: {training_loss.item()} epoch: {epoch}")
  # print("It has now been "+ time.strftime("%Mm%Ss", time.gmtime(time.time() - start))  +"  since the beginning of the program")
  text_file = open(f"./results/training/losses_{str(type(model).__name__)}_{tumor_type}.txt", "a")  
  text_file.write(f"loss: {training_loss.item()} epoch: {epoch}\n")
  current =  time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start))
  text_file.write(f"It has now been {current} since the beginning of the program/n")
  text_file.close()
  
  
def to_see_model(path):
  
  model=torch.load(path, map_location=DEVICE)
  text_file = open(r"our_models/model1.txt", "w") 
  print(model, file=text_file)
  text_file.close()
  
def plot_losses(losses):
  xh = np.arange(0,number_of_epochs)
  plt.plot(xh, losses, color = 'b', marker = ',',label = "Loss") 
  plt.xlabel("Epochs Traversed")
  plt.ylabel("Training Loss")
  plt.grid() 
  plt.legend() 
  plt.show()
  
  
def test(cnn, test_loader, classes =2):
  print(classes)
  testing_accuracy_sum = 0
  accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=classes).to(DEVICE)
  cnn = cnn.to(DEVICE)
  for (X_test, y_test) in test_loader:
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    test_predictions = cnn(X_test)
    testing_accuracy_sum += accuracy(test_predictions, y_test)
  
  test_accuracy = testing_accuracy_sum / len(test_loader)
  
  return test_accuracy

if __name__ == "__main__":
  seed = 99
  DEVICE = utils.load_device(seed)
  number_of_epochs = 20         

  _,transforms = ld.setup_resnet_model(seed)
  for tumor_type in os.listdir('./images'):
      print(tumor_type)
      if tumor_type in ['.DS_Store','__MACOSX'] :
          continue
      loaders, count_dict = ld.load_training_image_data(batch_size=300,tumor_type=tumor_type,transforms=transforms, normalized=True)
      train_loader, valid_loader, test_loader = loaders
      train_count, valid_count, test_count = count_dict

      resnet_classifier = ResNet_Tumor(classes=len(train_count.keys()))
      # summary(resnet_classifier,(1, 3,224,224))
      
      train_model(resnet_classifier,tumor_type,input_shape=(3,224,224),train_loader=train_loader,valid_loader=valid_loader,train_count=train_count, valid_count=valid_count,num_epochs = number_of_epochs,number_of_validations = 3,learning_rate = 0.001, weight_decay=0.001)

  # test_dict = {}
  # for filename in os.listdir(f"results/training/models/{type(resnet_classifier).__name__}/{tumor_type}"):
  #   model_path = os.path.join(f"results/training/models/{type(resnet_classifier).__name__}/{tumor_type}", filename)
  #   if os.path.isfile(model_path) and model_path.endswith('.pt'):
  #     resnet_classifier = ResNet_Tumor(classes=len(train_count.keys()))
  #     resnet_classifier.load_state_dict(torch.load(model_path, map_location = DEVICE,weights_only=True))
  #     print(test(resnet_classifier, test_loader))
  #     test_dict[model_path] = test(resnet_classifier, test_loader, classes=len(train_count.keys()))
  # if test_dict:
  #   print(max(test_dict, key=test_dict.get))