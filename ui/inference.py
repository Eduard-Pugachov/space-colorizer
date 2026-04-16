import torch
from PIL import Image, ImageEnhance
from torchvision import transforms

from models.unet import UNet


class SpaceColorizer:
    def __init__(self, checkpoint_path: str, img_size: int = 256):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.img_size = img_size

        self.model = UNet(in_ch=1, out_ch=3).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

        # preprocessing: resize + to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
        ])
    @torch.inference_mode()
    def colorize_pil_image(self, img: Image.Image) -> Image.Image:
        
        gray = img.convert("L")
        x = self.to_tensor(gray).unsqueeze(0).to(self.device)

        pred = self.model(x).squeeze(0).cpu()
        out = transforms.ToPILImage()(pred.clamp(0, 1))

        # just make what the model predicted more visible
        out = ImageEnhance.Color(out).enhance(2.9)      # vivid but not extreme
        out = ImageEnhance.Contrast(out).enhance(1.1)   # slight pop
        out = ImageEnhance.Sharpness(out).enhance(1.1)  # crisp edges

        return out