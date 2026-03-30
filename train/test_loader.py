from torch.utils.data import DataLoader
from datasets.space_dataset import SpaceColorizationDataset
import matplotlib.pyplot as plt
import torchvision

def main():
    ds = SpaceColorizationDataset(
        list_file="data/splits/train.txt",
        root_dir="data/processed",
        img_size=256,
        augment=False,
    )

    print("Dataset size:", len(ds))

    loader = DataLoader(ds, batch_size=4, shuffle=True)

    gray_batch, color_batch = next(iter(loader))
    print("Gray batch shape:", gray_batch.shape)
    print("Color batch shape:", color_batch.shape)

    grid_gray = torchvision.utils.make_grid(gray_batch, nrow=4, normalize=True)
    grid_color = torchvision.utils.make_grid(color_batch, nrow=4, normalize = True)

    plt.figure(figsize=(8, 4))

    plt.subplot(1,2,1)
    plt.title("Gray")
    plt.imshow(grid_gray.permute(1,2,0))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Color")
    plt.imshow(grid_color.permute(1,2,0))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    