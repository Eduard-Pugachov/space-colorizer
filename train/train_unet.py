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
        nrow=gray.size(0),
        normalize=False
    )

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for gray, color in loader:
        gray = gray.to(device)
        color = color.to(device)

        #calls UNet.forward(gray)
        #every operation is recorded in computation graph (directed, acyclic)
        prediction = model(gray)

        loss = loss_fn(prediction, color)

        #gradients accumulate by default, so zero them
        #always zero before backward!!
        optimizer.zero_grad()

        #using chain rule in calc, pytorch walks the computation graph backwards
        #from loss, stores partial derivative in theta.grad (backpropagation)
        loss.backward()

        #Adam reads every parameter's .grad, modifies weights in place
        optimizer.step()

        total_loss += loss.item() * gray.size(0)

    return total_loss / len(loader.dataset)

def validate(model, loader, loss_fn, device):
    #.eval() switches some layers to inference mode, good practice
    model.eval()
    total_loss = 0.0

    #tells pytorch to not build comp graph for these opers
    #since no .backward(), we can save memory, runs up to 30% faster
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
    print(f"Training on: {device}")  # add this
    print(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}")

    train_dataset = SpaceColorizationDataset(
        list_file=cfg["train_list"],
        root_dir=cfg["data_root"],
        img_size=cfg["img_size"],
        augment=True
    )
    val_ds = SpaceColorizationDataset(
        list_file=cfg["val_list"],
        root_dir=cfg["data_root"],
        img_size=cfg["img_size"],
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)


    model = UNet(in_ch=1, out_ch=3).to(device)
    def combined_loss(pred, target):     
        l1 = nn.L1Loss()(pred, target)
        # MSE pushes the model harder on large color differences
        mse = nn.MSELoss()(pred, target)
        return l1 + 0.5 * mse
    loss_fn = combined_loss
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    #automatically reduces lr when loss stops improving (learning rate scheduler)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",       # reduce when loss stops going down
    factor=0.5,       # multiply lr by 0.5 when triggered
    patience=3       # wait 3 epochs before reducing
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, cfg["epochs"]+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"Epoch{epoch:03d} -- Train Loss: {train_loss:4f} -- Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        torch.save(
            model.state_dict(),
            f"{cfg["checkpoint_dir"]}/unet_epoch{epoch:03d}.pth"
        )

        # saves a visual sample every 2 epochs
        if epoch % 2 == 0:
            model.eval()
            gray_sample, color_sample = next(iter(val_loader))
            gray_sample  = gray_sample[:4].to(device)
            color_sample = color_sample[:4].to(device)
            with torch.no_grad():
                pred_sample = model(gray_sample)
            save_samples(gray_sample, pred_sample, color_sample, epoch, cfg)

    print("training complete")

if __name__ == "__main__":
    main()