import gradio as gr
from ui.inference import SpaceColorizer

CHECKPOINT_PATH = "outputs/checkpoints/unet_epoch018.pth"
colorizer = SpaceColorizer(checkpoint_path=CHECKPOINT_PATH)

def predict(img):
    return colorizer.colorize_pil_image(img)

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Grayscale space image"),
    outputs=gr.Image(type="pil", label="Colorized image"),
    title="Space Colorizer",
    description="Upload a grayscale telescope image (JPG/PNG). The model predicts plausible colors.",
    theme=theme,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()