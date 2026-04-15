from PIL import Image
import os

def convert_to_grayscale(input_path: str, output_path: str = None) -> str:
    
    # if no output_path is given, saves alongside the original with '_gray' suffix.
 
    img = Image.open(input_path).convert("L")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_gray{ext}"

    img.save(output_path)
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    # batch convert everything in this folder
    folder = os.path.dirname(__file__)
    extensions = (".jpg", ".jpeg", ".png")

    for fname in os.listdir(folder):
        if fname.endswith(extensions) and "_gray" not in fname:
            convert_to_grayscale(os.path.join(folder, fname))