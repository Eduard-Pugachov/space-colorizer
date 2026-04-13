import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


#subclassing PyTorch Dataset
class SpaceColorizationDataset(Dataset):
    def __init__(self, list_file, root_dir, img_size=256, augment = False):
        with open(list_file) as f:
            #removes newline characters, skips blank lines
            #self.paths is python list of filenames
            self.paths = [l.strip() for l in f if l.strip()]

        self.root = root_dir
        self.img_size = img_size
        self.augment = augment

        #interpolates img to 256x256 regardless of OG size
        #converts PIL image to pytorch float tensor
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    #pytorch calls this to know num of samples
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        #normalize imgs to 3 channels (RGB)
        rel_path = self.paths[idx]
        rgb_path = os.path.join(self.root, rel_path)

        rgb = Image.open(rgb_path).convert("RGB")

        #PIL converts RGB to single-channel using
        #L = 0.299R + 0.587G + 0.114B
        gray = rgb.convert("L"); 

        gray_tensor = self.base_transform(gray) # (1, 256, 256)
        color_tensor = self.base_transform(rgb) # (3, 256, 256)

        return gray_tensor, color_tensor