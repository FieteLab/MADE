from PIL import Image
import glob


def stitch_images():
    # Get all PNG files in the imgs directory
    image_files = glob.glob("./imgs/*.png")

    if not image_files:
        print("No PNG images found in ./imgs directory")
        return

    # Load all images
    images = [Image.open(f) for f in image_files]

    # Get dimensions
    widths, heights = zip(*(i.size for i in images))

    # Calculate dimensions for the final image
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new image with the total width and max height
    new_image = Image.new("RGB", (total_width, max_height))

    # Paste images side by side
    current_width = 0
    for img in images:
        new_image.paste(img, (current_width, 0))
        current_width += img.size[0]

    # Save the stitched image
    output_path = "./imgs/stitched_output.png"
    new_image.save(output_path)
    print(f"Stitched image saved to: {output_path}")


if __name__ == "__main__":
    stitch_images()
