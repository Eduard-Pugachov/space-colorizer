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
        # ensure grayscale
        
        gray = img.convert("L")
        x = self.to_tensor(gray).unsqueeze(0).to(self.device)   # (1,1,H,W)

        pred = self.model(x).squeeze(0).cpu()                   # (3,H,W), [0,1]
        out = transforms.ToPILImage()(pred.clamp(0, 1))

        # boost saturation since model does not
        # make colors vivid
        out = ImageEnhance.Color(out).enhance(2.5)
        out = ImageEnhance.Contrast(out).enhance(1.3)

        return out