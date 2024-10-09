from PIL import Image, ImageDraw, ImageFont
import os

def add_text_to_image(input_path, output_path, text, position=(800, 2200), font_size=700, font_color=(255, 255, 255)):
    # Open the image
    image = Image.open(input_path)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Load a font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        print("DejaVuSans not found, using default font (fixed size)")
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=font_color)

    # Save the modified image
    image.save(output_path)


# Example usage
satellite = "GOES16"
composite = "ash"
reference_image = f"reference_image_{satellite}_{composite}.png"
input_image_path = f"./features/data/reference/{reference_image}"
output_image_path = f"./features/data/reference_different/{reference_image}"
text_to_add = 'Hello, World!'

add_text_to_image(input_image_path, output_image_path, text_to_add)
