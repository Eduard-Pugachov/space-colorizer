import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SpaceColorizationDataset(Dataset):
    def __init__(self, list_file, root_dir, img_size=256, augment = False):
        with open(list_file) as f:
            self.paths = [l.strip() for l in f if l.strip()]

        self.root = root_dir
        self.img_size = img_size
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        rgb_path = os.path.join(self.root, rel_path)

        rgb = Image.open(rgb_path).convert("RGB")

        gray = rgb.convert("L"); 

        gray_tensor = self.base_transform(gray)
        color_tensor = self.base_transform(rgb)

        return gray_tensor, color_tensor