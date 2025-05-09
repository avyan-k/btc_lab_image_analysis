import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch.optim as optim
from tqdm import tqdm
from torcheval.metrics.functional import multiclass_auroc,multiclass_f1_score
from torcheval.metrics import MulticlassAUROC
import os
import time
from torchinfo import summary
from typing import Tuple

import loading_data as ld
import utils
import models as md


def train_model(
    model:nn.Module,
    tumor_type,
    seed,
    input_shape,
    train_loader,
    valid_loader,
    train_count,
    valid_count,
    save_path,
    num_epochs: int = 200,
    number_of_validations: int = 3,
    learning_rate: float = 0.001,
    weight_decay: float = 0.001,
    imbalanced: bool = False,
    early_stopping: bool = False
):
    losses = np.empty((num_epochs, 2))  # list of tuple train_loss,val_loss
    accuracies = np.empty((num_epochs, 2))  # list of tuple train_acc,val_acc

    start = time.time()
    val_iteration = max(
        len(train_loader) // number_of_validations, 3
    )  # validate at least every 3 iterations

    loss_path = os.path.join(
        save_path, f"losses_{str(type(model).__name__)}_{tumor_type}.txt"
    )
    log_model(model, train_count, valid_count, num_epochs, input_shape, loss_path, seed)

    model = model.to(DEVICE)
    # Initializes the Adam optimizer with the model's parameters
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # Define weighted loss functions and accuracy metrics
    train_loss_function = nn.CrossEntropyLoss(
        weight=ld.count_dict_tensor(train_count)
    ).to(DEVICE)
    if imbalanced:
        accuracy = lambda y_hat, y: multiclass_f1_score(y_hat.cpu(), y.cpu(), num_classes=len(train_count.keys()))
    else:
        accuracy = lambda y_hat, y: multiclass_auroc(y_hat.cpu(),y.cpu(),num_classes=len(train_count.keys()))
    for epoch in tqdm(range(num_epochs), desc="Epoch", position=3, leave=False):
        train_loss_sum = 0
        train_loss = float("inf")
        val_loss = float("inf")
        val_accuracy = 0
        train_accuracy_sum = 0
        for iteration, (X_train, y_train) in enumerate(
            tqdm((train_loader), desc="Iteration", position=4, leave=False)
        ):
            # resets all gradients to 0 after each batch
            optimizer.zero_grad()
            # pass all data to GPU
            X_train = X_train.to(DEVICE)
            y_train = y_train.to(DEVICE)
            # forward pass of the CNN model on the input data to get predictions
            y_hat = model(X_train)

            train_accuracy_sum += accuracy(y_hat, y_train)
            # comparing the model's predictions with the truth labels
            train_loss = train_loss_function(y_hat, y_train)

            # SGD makes loss fluctuates a lot, so we take average over epoch
            train_loss_sum += train_loss

            # backpropagating the loss through the model
            train_loss.backward()

            # takes a step in the direction that minimizes the loss
            optimizer.step()

            # checks if should compute the validation metrics for plotting later
            if iteration % val_iteration == 0:
                val_loss, val_accuracy = valid_model(
                    model,
                    tumor_type,
                    valid_loader,
                    valid_count,
                    epoch,
                    iteration,
                    accuracy,
                    save_path,
                )

        train_accuracy = train_accuracy_sum / len(train_loader)
        train_loss = train_loss_sum / len(train_loader)
        # logging results
        log_training_results(
            filepath=loss_path,
            losses=losses,
            accuracies=accuracies,
            epoch=epoch,
            start=start,
            current_loss=train_loss,
            current_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )
        if early_stopping:
            if check_if_overfit(valid_losses = losses[:,1].squeeze(),filepath=loss_path):
                break       
    return model, losses, accuracies


def valid_model(
    model,
    tumor_type,
    valid_loader,
    valid_count,
    epoch,
    iteration,
    accuracy_function,
    model_directory,
):
    loss_function = nn.CrossEntropyLoss(weight=ld.count_dict_tensor(valid_count)).to(DEVICE)
    loss_path = os.path.join(
        model_directory, f"losses_{str(type(model).__name__)}_{tumor_type}.txt"
    )
    # stops computing gradients on the validation set
    with torch.no_grad():
        # Keep track of the losses & accuracies
        val_accuracy_sum = 0
        val_loss_sum = 0

        # Make a predictions on the full validation set, batch by batch
        for X_val, y_val in tqdm(
            valid_loader, desc="Validation Iteration", position=5, leave=False
        ):
            # Move the batch to GPU if it's available
            X_val = X_val.to(DEVICE)
            y_val = y_val.to(DEVICE)

            y_hat = model(X_val)
            val_accuracy_sum += accuracy_function(y_hat, y_val)
            val_loss_sum += loss_function(y_hat, y_val).cpu()

        # Divide by the number of iterations (and move back to CPU)
        val_accuracy = (val_accuracy_sum / len(valid_loader))
        val_loss = (val_loss_sum / len(valid_loader))

        log_validation_results(
            loss_path, epoch, iteration, loss=val_loss, accuracy=val_accuracy
        )
        # print(f"EPOCH = {epoch} --- ITERATION = {iteration}")
        # print(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}")
        # if val_accuracy > 0.95:
        #     torch.save(
        #         model.state_dict(),
        #         os.path.join(model_directory, f"ep={epoch}-iter={iteration}-seed={torch.seed()}-acc={val_accuracy}.pt"),
        #     )
        return val_loss, val_accuracy


def log_model(model, train_count, valid_count, epochs, input_shape, filepath, seed):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(
            f"Attempting {epochs} epochs on date of {utils.get_time()} on seed {seed} with model:\n"
        )
        model_stats = summary(model, input_size=input_shape, verbose=0)
        f.write(f"Model Summary:{str(model_stats)}\n")
        f.write(f"Training Data Weights: {train_count}\n")
        f.write(f"Validation Data Weights: {valid_count}\n")
    return


def log_training_results(
    filepath,
    losses,
    accuracies,
    epoch,
    start,
    current_loss,
    current_accuracy,
    val_loss,
    val_accuracy,
):
    losses[epoch][0] = float(current_loss.cpu())
    losses[epoch][1] = float(val_loss.cpu())
    accuracies[epoch][0] = float(current_accuracy.cpu())
    accuracies[epoch][1] = float(val_accuracy.cpu())

    # print(f"\n\nloss: {training_loss.item()} epoch: {epoch}")
    # print("It has now been "+ time.strftime("%Mm%Ss", time.gmtime(time.time() - start))  +"  since the beginning of the program")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"Results for Epoch {epoch}\n")
        f.write(
            f"Training loss = {current_loss.cpu().item()} --- Training accuracy = {current_accuracy.cpu().item()}\n"
        )
        current = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start))
        f.write(f"It has now been {current} since the beginning of the program\n\n")
    return


def log_validation_results(filepath, epoch, iteration, loss, accuracy):
    # Out to console
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"Validation Results for Epoch {epoch} Iteration {iteration}\n")
        f.write(f"Validation loss = {loss} --- Validation accuracy = {accuracy}\n\n")
        # print(f"Validation Results for Epoch {epoch} Iteration {iteration}")
        # print(f"Validation loss = {loss} --- Validation accuracy = {accuracy}")
    return

def plot_losses(losses, number_of_epochs, path):
    plt.figure()
    xh = np.arange(0, number_of_epochs)
    plt.plot(xh, losses[:, 0], color="b", marker=",", label="Training Loss")
    plt.plot(xh, losses[:, 1], color="r", marker=",", label="Test Loss")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    plt.xlabel("Epochs Traversed")
    plt.ylabel("Losses")
    plt.grid()
    plt.legend()
    img_file = os.path.join(path, "losses.png")
    if os.path.isfile(img_file):
        os.remove(img_file)
    plt.savefig(img_file)
    plt.show()


def plot_accuracies(accuracies, number_of_epochs, path):
    plt.figure()
    xh = np.arange(0, number_of_epochs)
    plt.plot(xh, accuracies[:, 0], color="b", marker=",", label="Training Accuracy")
    plt.plot(xh, accuracies[:, 1], color="r", marker=",", label="Test Accuracy")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    plt.xlabel("Epochs Traversed")
    plt.ylabel("Accuracies")
    plt.grid()
    plt.legend()
    img_file = os.path.join(path, "accuracies.png")
    if os.path.isfile(img_file):
        os.remove(img_file)
    plt.savefig(img_file)
    plt.show()
    
def test(model, test_loader, test_count):
    testing_accuracy_sum = torch.tensor(0.0)
    model = model.to(DEVICE)
    accuracy = MulticlassAUROC(num_classes=len(test_count.keys())).to(DEVICE)
    for X_test, y_test in tqdm(test_loader):
        with torch.no_grad():
            X_test = X_test.to(DEVICE)
            y_test = y_test.to(DEVICE)
            test_predictions = model(X_test)
            accuracy.update(test_predictions, y_test)
            testing_accuracy_sum += accuracy.compute().cpu()
            accuracy.reset()

    test_accuracy = testing_accuracy_sum / len(test_loader)

    return test_accuracy

def check_if_overfit(valid_losses,filepath):
    assert len(valid_losses.shape) == 1
    if len(valid_losses) > 3:
        last_n_losses = np.array(valid_losses[-4:])
        change_in_loss = last_n_losses[1:]-last_n_losses[:-1]
        if np.all(change_in_loss>0):
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"Validation Loss increasing,\
                        change in validation loss from last 10 epochs {','.join([str(x) for x in change_in_loss])},\
                        exiting training loop") 
            return True
    return False

def main(number_of_epochs:int,samples_per_class:int,batch_size:int,stain_normalize:bool,proven_mutation_only:bool,normalize:bool,tumor_type:str,MLP=True):
    lr = 0.001
    wd = 0.001
    loaders, count_dict = ld.load_training_image_data(
    batch_size=batch_size,
    samples_per_class=samples_per_class,
    tumor_type=tumor_type,
    seed=seed,
    normalized=normalize,
    stain_normalized=stain_normalize,
    proven_mutation_only=proven_mutation_only,
    validation=False,
    )
    if len(loaders) == 3:
        train_loader, valid_loader, test_loader = loaders
    else:
        train_loader, test_loader = loaders
    if len(count_dict) == 3:
        train_count, valid_count, test_count = count_dict
    else:
        train_count, test_count = count_dict
    count_array = np.array(list(train_count.values()))
    # dataset is imbalanced if the largest class is over twice the size of the smallest class
    imbalance = count_array.max() / count_array.min() > 2
    classifier = md.ResNet_Tumor(classes=len(train_count.keys()),training=True, classif_layers=1 if MLP else 0)
    classifier = classifier.to(DEVICE)
    summary(classifier, input_size=(batch_size, 3, 224, 224))

    dataset_description = ld.get_dataset_info(tumor_type,normalize,True,proven_mutation_only)
    model_path = f"./results/training/models/{str(type(classifier).__name__)}{'_MLP' if MLP else ''}/epochs={number_of_epochs}/{dataset_description}"
    os.makedirs(model_path,exist_ok=True)
    
    trained_model,losses, accuracies = train_model(
        classifier,
        tumor_type,
        seed = seed,
        input_shape=(batch_size, 3, 224, 224),
        train_loader=train_loader,
        valid_loader=test_loader,
        train_count=train_count,
        valid_count=test_count,
        save_path=model_path,
        num_epochs=number_of_epochs,
        number_of_validations=3,
        learning_rate=lr,
        weight_decay=wd,
        imbalanced=imbalance,
        early_stopping=True
    )

    plot_losses(losses, number_of_epochs, model_path)
    plot_accuracies(accuracies, number_of_epochs, model_path)
    model_save_path = os.path.join(model_path, f"batch={batch_size}-test_acc={accuracies[1][-1]:0.3f}.pt")

    """Ideally we define a FeatureExtractor parent class to class polymorphism,
    but for now it seems torchscript does not support inheritance
    https://learn.foundry.com/nuke/developers/14.0/catfilecreationreferenceguide/inheritance.html
    thus we do a manual check
    """
    if isinstance(trained_model,md.ResNet_Tumor) or isinstance(trained_model,md.UNI_Tumor):
        trained_model.save_with_softmax(model_save_path)
    else:
        torch.save(trained_model.state_dict(),model_save_path)


if __name__ == "__main__":
    utils.print_cuda_memory()
    seed = 99
    utils.set_seed(seed)
    DEVICE = utils.load_device(seed)
    main(15,150_000,128,False,True,False,"DDC_UC_1",True)
    main(15,150_000,128,False,True,False,"DDC_UC_1",False)
    main(5,-1,128,False,True,False,"DDC_UC_1",True)
    main(5,-1,128,False,True,False,"DDC_UC_1",False)
    main(60,10_000,128,False,True,False,"DDC_UC_1",True)
    main(60,10_000,128,False,True,False,"DDC_UC_1",False)
        # print(losses,accuracies)
        # test_dict = {}
        # for filename in os.listdir(f"results/training/models/ResNet_Tumor/all/{tumor_type}"):
        #     model_path = os.path.join(f"results/training/models/ResNet_Tumor/{tumor_type}", filename)
        #     print(model_path)
        #     if os.path.isfile(model_path) and model_path.endswith('.pt'):
        #         classifier.load_state_dict(torch.load(model_path, map_location = DEVICE,weights_only=True))
        #         test_dict[model_path] = test(classifier, test_loader, test_count=test_count)
        #         print(test_dict[model_path])
        # if test_dict:
        #     print(max(test_dict, key=test_dict.get))
