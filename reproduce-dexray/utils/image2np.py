import os
import numpy as np
from PIL import Image
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor
import time


def process_image(file_info):
    file_name, image_dir, image_size, output_dir = file_info
    img_path = os.path.join(image_dir, file_name)

    try:
        # Process the image within a context manager
        with Image.open(img_path) as img:
            # Copy DexRay code here
            img = np.asarray(img)
            img = tf.convert_to_tensor(img, dtype_hint=None, name=None)

            shape = tf.numpy_function(get_shape, [img], tf.int64)

            img = tf.reshape(img, [shape, 1, 1])
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [image_size * image_size, 1])
            img = tf.reshape(img, [image_size * image_size, 1])

            img_array = img.numpy()  # Convert to numpy array
        # Save the processed image
        output_image_path = os.path.join(output_dir, f"{file_name[:-4]}.npy")
        np.save(output_image_path, img_array)

    except Exception as e:
        print(f"[!] Error processing image {file_name}: {e}")
        # Optionally, handle errors or save empty arrays if needed


def get_shape(image):
    return image.shape[0]


def load_images(image_base_dir: str, npy_base_dir: str, image_size=128):
    categories = ["goodware", "malware"]
    total_images = 0
    already_processed = 0
    to_be_processed = 0
    files_to_process = []

    for category in categories:
        image_dir = os.path.join(image_base_dir, category)
        output_dir = os.path.join(npy_base_dir, category)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(image_dir):
            print(f"[!] Image directory '{image_dir}' does not exist.")
            continue

        image_files = [
            file_name
            for file_name in sorted(os.listdir(image_dir))
            if file_name.endswith(".png")
        ]
        total_images += len(image_files)

        # Get already processed files
        processed_files = []
        if os.path.exists(output_dir):
            processed_files = [
                file_name[:-4]
                for file_name in os.listdir(output_dir)
                if file_name.endswith(".npy")
            ]
        already_processed += len(processed_files)

        # Determine files to process
        category_files_to_process = [
            (file_name, image_dir, image_size, output_dir)
            for file_name in image_files
            if file_name[:-4] not in processed_files
        ]
        to_be_processed += len(category_files_to_process)
        files_to_process.extend(category_files_to_process)

    # Output the counts before processing
    print(f"Total images in input folders: {total_images}")
    print(f"Images already processed: {already_processed}")
    print(f"Images to be processed: {to_be_processed}")
    time.sleep(5)
    if not files_to_process:
        print("No images to process. Exiting.")
        return

    # Process images in parallel
    with ProcessPoolExecutor(max_workers=20) as executor:
        list(executor.map(process_image, files_to_process))

    print("Processing completed.")


if __name__ == "__main__":
    IMAGE_BASE_DIR = "100k_download/images"
    NPY_BASE_DIR = "100k_download/npy"
    IMAGE_SIZE = 128

    try:
        load_images(IMAGE_BASE_DIR, NPY_BASE_DIR, IMAGE_SIZE)
    except Exception as e:
        print(f"[!] An error occurred: {e}")
