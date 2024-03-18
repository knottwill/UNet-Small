"""!@file train_eval.py

@brief Functions for training and evaluating the model
"""

import torch
from tqdm import tqdm

from .losses import ComboLoss
from .metrics import dice_coefficients, mask_accuracies


def train(model, device, dataloader, optimizer):
    """!
    @brief Train model for 1 epoch

    @param model: model to train
    @param device: device to use for training
    @param dataloader: dataloader for training data
    @param optimizer: optimizer to use for training

    @return: average loss, accuracy, and dice similarity coefficient for the epoch
    """
    model.train()

    total_loss, cumulative_accuracy, cumulative_dsc = 0, 0, 0
    for x, masks in tqdm(dataloader):
        x, masks = x.to(device), masks.to(device)

        optimizer.zero_grad()  # zero the gradients
        probs = model(x)  # forward pass
        loss = ComboLoss()(probs, masks)  # calculate loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights
        total_loss += loss.item()

        # calculate accuracy and dice similarity coefficient
        probs, masks = probs.detach().cpu().squeeze().numpy(), masks.detach().cpu().squeeze().numpy()
        cumulative_accuracy += mask_accuracies(probs, masks).mean()
        cumulative_dsc += dice_coefficients(probs, masks).mean()

    # calculate average of each metric over the epoch
    loss_avg = total_loss / len(dataloader)
    accuracy = cumulative_accuracy / len(dataloader)
    dsc = cumulative_dsc / len(dataloader)

    return loss_avg, accuracy, dsc


@torch.no_grad()
def evaluate(model, device, dataloader):
    """!
    @brief Evaluate model on validation or test data

    @param model: model to evaluate
    @param device: device to use for evaluation
    @param dataloader: dataloader for validation or test data

    @return: average loss, accuracy, and dice similarity coefficient for the epoch
    """
    model.eval()

    total_loss, cumulative_accuracy, cumulative_dsc = 0, 0, 0
    for x, masks in dataloader:
        x, masks = x.to(device), masks.to(device)

        probs = model(x)  # forward pass
        loss = ComboLoss()(probs, masks)  # calculate loss
        total_loss += loss.item()  # accumulate loss

        # calculate accuracy and dice similarity coefficient
        probs, masks = probs.detach().cpu().squeeze().numpy(), masks.detach().cpu().squeeze().numpy()
        cumulative_accuracy += mask_accuracies(probs, masks).mean()
        cumulative_dsc += dice_coefficients(probs, masks).mean()

    # calculate average of each metric over the epoch
    loss_avg = total_loss / len(dataloader)
    accuracy = cumulative_accuracy / len(dataloader)
    dsc = cumulative_dsc / len(dataloader)

    return loss_avg, accuracy, dsc
