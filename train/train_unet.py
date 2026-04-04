import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets.space_dataset import SpaceColorizationDataset
from models.unet import UNet


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
    
def save_samples(gray, pred, color, epoch, cfg):
    os.makedirs(cfg["sample_dir"], exist_ok=True)
    comparison = torch.cat([gray.expand(-1,3,-1,-1), pred, color], dim=0)
    save_image(
        comparison,
        f"{cfg["sample_dir"]}/epoch_{epoch:03d}.png",
        nrow=gray.size[0],
        normalize=False
    )

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for gray, color in loader:
        gray = gray.to(device)
        color = color.to(device)

        prediction = model(gray)

        loss = loss_fn(pred, color)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * gray.size(0)

    return total_loss / len(loader.dataset)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for gray, color in loader:
            gray = gray.to(device)
            color = color.to(device)
            pred = model(gray)
            loss = loss_fn(pred, color)
            total_loss += loss.item() * gray.size(0)

    return total_loss / len(loader.dataset)

def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

