import os
import numpy as np
from PIL import Image

def process_image(file_info):
    file_name, image_dir, image_size, output_dir = file_info
    img_path = os.path.join(image_dir, file_name)

    try:
        # Process the image within a context manager
        with Image.open(img_path) as img:
            img = img.convert('L')  # Convert to grayscale
            img = img.resize(image_size)  # Resize to uniform size
            img_array = np.array(img).flatten()  # Convert to numpy array and flatten

            # Save the processed image
            output_image_path = os.path.join(output_dir, f"{file_name[:-4]}.npy")
            np.save(output_image_path, img_array)

    except Exception as e:
        print(f"[!] Error processing image {file_name}: {e}")

