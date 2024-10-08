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
import models as md


def train_model(model,tumor_type,input_shape,train_loader,valid_loader, train_count,valid_count,num_epochs = 200,number_of_validations = 3,learning_rate = 0.001, weight_decay=0.001):
  
  losses = np.empty(num_epochs)
  start = time.time()
  val_iteration = max(len(train_loader) // number_of_validations,3) #validate at least every 3 iterations

  model_path = f"./results/training/models/{str(type(model).__name__)}/{tumor_type}"
  Path(model_path).mkdir(parents=True, exist_ok=True)
  loss_path = os.path.join(model_path,f"losses_{str(type(model).__name__)}_{tumor_type}.txt")
  log_model(model,train_count, valid_count, num_epochs, input_shape, loss_path)

  model = model.to(DEVICE)
  # Initializes the Adam optimizer with the model's parameters
  optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
  # Define weighted loss functions and accuracy metrics
  train_loss_function = nn.CrossEntropyLoss(weight=ld.count_dict_tensor(train_count)).to(DEVICE)
  accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(train_count.keys()),average = 'weighted').to(DEVICE)

  for epoch in tqdm(range(num_epochs),desc="Epoch",position=3,leave=False):
    train_loss = float('inf')
    for iteration, (X_train, y_train) in enumerate(tqdm((train_loader),desc="Iteration",position=4,leave=False)):
      # resets all gradients to 0 after each batch
      optimizer.zero_grad()
      # pass all data to GPU
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
      if iteration % val_iteration == 0 and epoch % 3 == 1:
        done = valid_model(model,tumor_type,valid_loader,valid_count,epoch,iteration,accuracy, model_path)
      
        if done:
          log_training_results(loss_path,losses,epoch,start, current_loss=train_loss)
          return losses

    # logging results
    log_training_results(loss_path, losses,epoch,start, current_loss=train_loss)
  plot_losses(losses,epochs,model_path)
  return losses

def valid_model(model,tumor_type,valid_loader, valid_count, epoch,iteration,accuracy, model_directory):
  loss_function = nn.CrossEntropyLoss(weight=ld.count_dict_tensor(valid_count)).to(DEVICE)
  loss_path = os.path.join(model_directory,f"losses_{str(type(model).__name__)}_{tumor_type}.txt")
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
      val_loss_sum += loss_function(y_hat, y_val)

    # Divide by the number of iterations (and move back to CPU)
    val_accuracy = (val_accuracy_sum / len(valid_loader)).cpu()
    val_loss = (val_loss_sum / len(valid_loader)).cpu()

    log_validation_results(loss_path,epoch,iteration,loss=val_loss,accuracy=val_accuracy)
    # print(f"EPOCH = {epoch} --- ITERATION = {iteration}")
    # print(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}")
    if val_accuracy > 0.95:
      torch.save(model.state_dict(),os.path.join(model_directory,f"{epoch}-{iteration}-{val_accuracy}.pt"))
      if len([name for name in os.listdir(model_directory)]) > 5:
        return True
    return False

def log_model(model, train_count, valid_count, epochs, input_shape, filepath):
  with  open(filepath, "w",encoding="utf-8") as f:
    f.write(f"Attempting {epochs} epochs on date of {utils.get_time()} with model:\n")
    model_stats = summary(model, input_size=input_shape, verbose=0)
    f.write(f"Model Summary:{str(model_stats)}\n")
    f.write(f"Training Data Weights: {train_count}\n")
    f.write(f"Validation Data Weights: {valid_count}\n")
  return

def log_training_results(filepath,losses,epoch,start, current_loss):
  
  training_loss = current_loss.cpu()
  losses[epoch] = float(training_loss)
  # print(f"\n\nloss: {training_loss.item()} epoch: {epoch}")
  # print("It has now been "+ time.strftime("%Mm%Ss", time.gmtime(time.time() - start))  +"  since the beginning of the program")
  with open(filepath, "a",encoding="utf-8") as f:
    f.write(f"Training Results for Epoch {epoch}\n")
    f.write(f"Training loss: {training_loss.item()}\n")
    current =  time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start))
    f.write(f"It has now been {current} since the beginning of the program\n\n")
  return

def log_validation_results(filepath, epoch, iteration, loss, accuracy):
  # Out to console
  with open(filepath, "a",encoding="utf-8") as f:
    f.write(f"Validation Results for Epoch {epoch} Iteration {iteration}\n")
    f.write(f"Validation loss = {loss} --- Validation accuracy = {accuracy}\n\n")
    # print(f"Validation Results for Epoch {epoch} Iteration {iteration}")
    # print(f"Validation loss = {loss} --- Validation accuracy = {accuracy}")
  return

def to_see_model(path):
  
  model=torch.load(path, map_location=DEVICE)
  text_file = open(r"our_models/model1.txt", "w") 
  print(model, file=text_file)
  text_file.close()
  
def plot_losses(losses,number_of_epochs,path):
  xh = np.arange(0,number_of_epochs)
  plt.plot(xh, losses, color = 'b', marker = ',',label = "Loss") 
  plt.xlabel("Epochs Traversed")
  plt.ylabel("Training Loss")
  plt.grid() 
  plt.legend() 
  plt.show()
  plt.savefig(os.path.join(path,"losses.png"))
  

def test(cnn, test_loader, test_count):
  testing_accuracy_sum = 0
  cnn = cnn.to(DEVICE)
  accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(test_count.keys()), average = 'weighted').to(DEVICE)
  for (X_test, y_test) in test_loader:
    with torch.no_grad():
      X_test = X_test.to(DEVICE)
      y_test = y_test.to(DEVICE)
      test_predictions = cnn(X_test)
      testing_accuracy_sum += accuracy(test_predictions, y_test).cpu()
  
  test_accuracy = testing_accuracy_sum / len(test_loader)
  
  return test_accuracy

if __name__ == "__main__":
  utils.print_cuda_memory()
  seed = 99
  DEVICE = utils.load_device(seed)
  number_of_epochs = 20
  batch_size = 300

  _,transforms = ld.setup_resnet_model(seed) 
  first_tumor_type = True #only print summary for first tumor_type
  for tumor_type in os.listdir('./images'):
      print(tumor_type)
      if tumor_type in ['.DS_Store','__MACOSX'] :
          continue
      loaders, count_dict = ld.load_training_image_data(batch_size=batch_size,tumor_type=tumor_type,transforms=transforms, normalized=True)  
      # loaders, count_dict = ld.load_training_feature_data(batch_size=50,model_type="ResNet",tumor_type=tumor_type) 
      train_loader, valid_loader, test_loader = loaders
      train_count, valid_count, test_count = count_dict

    #   feature_classifier = Tumor_Classifier(layers=5, neurons_per_layer=64, dropout=0, input_neurons=1000, classes=len(train_count.keys())) 
    #   if first_tumor_type:
    #     first_tumor_type = False
    #     summary(feature_classifier,(1, 1000 ,1))

      resnet_classifier = md.ResNet_Tumor(classes=len(train_count.keys()))
      if first_tumor_type:
        first_tumor_type = False
        summary(resnet_classifier,input_size=(batch_size, 3,224,224))
      train_model(resnet_classifier,tumor_type,input_shape=(batch_size,3,224,224),train_loader=train_loader,valid_loader=valid_loader,train_count=train_count, valid_count=valid_count,num_epochs = number_of_epochs,number_of_validations = 3,learning_rate = 0.001, weight_decay=0.001)

    #   test_dict = {}
    #   for filename in os.listdir(f"results/training/models/ResNet_Tumor/{tumor_type}"):
    #     model_path = os.path.join(f"results/training/models/ResNet_Tumor/{tumor_type}", filename)
    #     if os.path.isfile(model_path) and model_path.endswith('.pt'):
    #       resnet_classifier = ResNet_Tumor(classes=len(train_count.keys()))
    #       resnet_classifier.load_state_dict(torch.load(model_path, map_location = DEVICE,weights_only=True))
    #       print(test(cnn=resnet_classifier, test_loader=test_loader, test_count=test_count))
    #       test_dict[model_path] = test(resnet_classifier, test_loader, test_count=test_count)
    #   if test_dict:
    #     print(max(test_dict, key=test_dict.get))