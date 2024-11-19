import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from image_processing import process_image  # Import the function from the new module

def load_images(image_dir: str, output_dir: str, image_size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    
    files = [
        (file_name, image_dir, image_size, output_dir)
        for file_name in sorted(os.listdir(image_dir))
        if file_name.endswith('.png')
    ]
    
    # Filter out already processed files
    files_to_process = [
        file for file in files
        if not os.path.exists(os.path.join(output_dir, f"{file[0][:-4]}.npy"))
    ]
    print(f"files: {len(files)}")
    print(f"files to process: {len(files_to_process)}")
    
    # Process images in parallel
    with ProcessPoolExecutor(max_workers=10) as executor:
        list(executor.map(process_image, files_to_process))

if __name__ == "__main__":
    MALWARE_PATH = "/scratch/users/mbenali/download_apk/images/goodware"
    try:
        load_images(MALWARE_PATH, "./npy/goodware", (128, 128))
    except Exception as e:
        print(f"[!] An error occurred: {e}")

