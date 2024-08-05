from PIL import Image, ImageDraw, ImageFont


def add_text_to_image(input_path, output_path, text, position=(800, 2200), font_size=700, font_color=(255, 255, 255)):
    # Open the image
    image = Image.open(input_path)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Load a font
    font = ImageFont.load_default()

    # Specify font size and color
    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text(position, text, font=font, fill=font_color)

    # Save the modified image
    image.save(output_path)


# Example usage
input_image_path = 'C:/Users/sennlaub/IdeaProjects/DWD_Pytroll/img/reference.png'
output_image_path = 'C:/Users/sennlaub/IdeaProjects/DWD_Pytroll/img/reference_different.png'
text_to_add = 'Hello, World!'

add_text_to_image(input_image_path, output_image_path, text_to_add)
