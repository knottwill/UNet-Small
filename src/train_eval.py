import tqdm
import torch
from torchmetrics.classification import BinaryAccuracy, Dice

from .losses import ComboLoss


def train(model, device, dataloader, optimizer):
    """Train 1 epoch"""
    model.train()

    total_loss, cumulative_accuracy, cumulative_dsc = 0, 0, 0
    for x, masks in tqdm(dataloader):
        x, masks = x.to(device), masks.to(device)

        optimizer.zero_grad()
        probs = model(x)
        loss = ComboLoss()(probs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        cumulative_accuracy += BinaryAccuracy(threshold=0.5)(probs, masks).item()
        cumulative_dsc += Dice(zero_division=1, threshold=0.5, average="samples")(probs, masks.int())

    accuracy = cumulative_accuracy / len(dataloader)
    dsc = cumulative_dsc / len(dataloader)

    return total_loss, accuracy, dsc


@torch.no_grad()
def evaluate(model, device, dataloader):
    model.eval()

    total_loss, cumulative_accuracy, cumulative_dsc = 0, 0, 0
    for x, masks in dataloader:
        x, masks = x.to(device), masks.to(device)

        probs = model(x)
        loss = ComboLoss()(probs, masks)

        total_loss += loss.item()
        cumulative_accuracy += BinaryAccuracy(threshold=0.5)(probs, masks).item()
        cumulative_dsc += Dice(zero_division=1, threshold=0.5, average="samples")(probs, masks.int())

    accuracy = cumulative_accuracy / len(dataloader)
    dsc = cumulative_dsc / len(dataloader)

    return total_loss, accuracy, dsc
