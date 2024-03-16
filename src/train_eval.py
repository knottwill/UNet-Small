import torch
from tqdm import tqdm

from .losses import ComboLoss
from .metrics import dice_coefficients, mask_accuracies


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

        probs, masks = probs.detach().cpu().squeeze().numpy(), masks.detach().cpu().squeeze().numpy()
        cumulative_accuracy += mask_accuracies(probs, masks).mean()
        cumulative_dsc += dice_coefficients(probs, masks).mean()

    batch_loss_avg = total_loss / len(dataloader)
    accuracy = cumulative_accuracy / len(dataloader)
    dsc = cumulative_dsc / len(dataloader)

    return batch_loss_avg, accuracy, dsc


@torch.no_grad()
def evaluate(model, device, dataloader):
    model.eval()

    total_loss, cumulative_accuracy, cumulative_dsc = 0, 0, 0
    for x, masks in dataloader:
        x, masks = x.to(device), masks.to(device)

        probs = model(x)
        loss = ComboLoss()(probs, masks)
        total_loss += loss.item()

        probs, masks = probs.detach().cpu().squeeze().numpy(), masks.detach().cpu().squeeze().numpy()
        cumulative_accuracy += mask_accuracies(probs, masks).mean()
        cumulative_dsc += dice_coefficients(probs, masks).mean()

    batch_loss_avg = total_loss / len(dataloader)
    accuracy = cumulative_accuracy / len(dataloader)
    dsc = cumulative_dsc / len(dataloader)

    return batch_loss_avg, accuracy, dsc
