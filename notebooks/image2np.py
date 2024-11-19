import os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

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
        #output_image_path = os.path.join(output_dir,"errors", f"{file_name[:-4]}.npy")
        #np.save(output_image_path, np.array([]))

    # return file_name #, label, apk_date

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
    print(f"files: { len(files)}")
    print(f"files to process: {len(files_to_process)}")
    # Process images in parallel
    with ProcessPoolExecutor(15) as executor:
       list(executor.map(process_image, files_to_process))

if __name__ == "__main__":
    MALWARE_PATH = "/scratch/users/mbenali/download_apk/images/goodware"
    # started at 658
    try:
        load_images(MALWARE_PATH,"/scratch/users/mbenali/download_apk/npy/goodware",(128, 128))
    except Exception as e:
        print(f"[!] An error occurred: {e}")
