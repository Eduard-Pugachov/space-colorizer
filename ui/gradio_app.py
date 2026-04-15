import gradio as gr
from ui.inference import SpaceColorizer

CHECKPOINT_PATH = "outputs/checkpoints/unet_epoch018.pth"  # or your best
colorizer = SpaceColorizer(checkpoint_path=CHECKPOINT_PATH)


def predict(img):
    """
    img: PIL image from gr.Image(type="pil")
    returns: PIL image (colorized)
    """
    return colorizer.colorize_pil(img)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Grayscale space image"),
    outputs=gr.Image(type="pil", label="Colorized image"),
    title="Space Colorizer",
    description="Upload a grayscale telescope image (JPG/PNG). The model predicts plausible colors."
)

if __name__ == "__main__":
    demo.launch()