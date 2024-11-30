import os
from concurrent.futures import ProcessPoolExecutor
from androguard.core.apk import APK
from PIL import Image
import time

# Hardcoded paths
source_folder = "100k_download/apks"
destination_folder = "100k_download/images"


def get_dex_bytes(apk: APK):
    """Extract .dex file bytes from the APK."""
    for f in apk.get_files():
        if f.endswith(".dex"):
            yield apk.get_file(f)


def generate_png(apk: APK, output_image_path: str):
    """Generate a grayscale image from .dex bytes."""
    stream = bytes()
    for s in get_dex_bytes(apk):
        stream += s

    current_len = len(stream)
    if current_len == 0:
        raise ValueError("No .dex files found in the APK")

    # Create an image from the byte stream
    image = Image.frombytes(mode="L", size=(1, current_len), data=stream)
    image.save(output_image_path)


def process_apk(args):
    """Process a single APK."""
    apk_path, output_image_path = args
    try:
        apk = APK(apk_path)
        generate_png(apk, output_image_path)
        print(f"[SUCCESS] Processed: {apk_path}")
    except Exception as e:
        print(f"[ERROR] Failed to process {apk_path}: {e}")


def process_folder():
    """Process all APKs in the source folder using multiprocessing."""
    # Initialize processed_hashes set
    processed_hashes = set()

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    categories = ["goodware", "malware"]
    all_apks = []
    total_apks = 0

    for category in categories:
        source_category_path = os.path.join(source_folder, category)
        destination_category_path = os.path.join(destination_folder, category)
        os.makedirs(destination_category_path, exist_ok=True)

        # Collect already processed hashes from images in destination_category_path
        if os.path.exists(destination_category_path):
            processed_images = [
                f for f in os.listdir(destination_category_path) if f.endswith(".png")
            ]
            for img_file in processed_images:
                img_hash = os.path.splitext(os.path.basename(img_file))[0]
                processed_hashes.add(img_hash)
        else:
            processed_images = []

        if not os.path.exists(source_category_path):
            print(f"[!] Source folder '{source_category_path}' does not exist.")
            continue

        apks = [f for f in os.listdir(source_category_path) if f.endswith(".apk")]
        total_apks += len(apks)
        for apk_file in apks:
            apk_path = os.path.join(source_category_path, apk_file)
            output_image_path = os.path.join(
                destination_category_path, f"{os.path.splitext(apk_file)[0]}.png"
            )
            all_apks.append((apk_path, output_image_path))

    # Calculate numbers for reporting
    already_processed = 0
    to_be_processed = 0
    for apk_path, _ in all_apks:
        apk_hash = os.path.splitext(os.path.basename(apk_path))[0]
        if apk_hash in processed_hashes:
            already_processed += 1
        else:
            to_be_processed += 1

    # Output the counts before processing
    print(f"Total APKs in input folders: {total_apks}")
    print(f"APKs already processed: {already_processed}")
    print(f"APKs to be processed: {to_be_processed}")
    time.sleep(10)
    # Filter out already processed APKs
    apks_to_process = [
        (apk_path, output_image_path)
        for apk_path, output_image_path in all_apks
        if os.path.splitext(os.path.basename(apk_path))[0] not in processed_hashes
    ]

    if not apks_to_process:
        print("No APKs to process. Exiting.")
        return

    # Use ProcessPoolExecutor for multiprocessing
    with ProcessPoolExecutor(max_workers=20) as executor:
        executor.map(process_apk, apks_to_process)

    print("Processing completed.")


if __name__ == "__main__":
    try:
        process_folder()
    except Exception as e:
        print(f"[!] An error occurred: {e}")
