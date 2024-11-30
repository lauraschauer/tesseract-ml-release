import os
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Constants
API_KEY = ""  # Replace with your AndroZoo API key


# Constants
DOWNLOAD_URL = "https://androzoo.uni.lu/api/download"
OUTPUT_DIR = "100k_download/apks"  # Base directory to save APKs
MALWARE_DIR = os.path.join(OUTPUT_DIR, "malware")
GOODWARE_DIR = os.path.join(OUTPUT_DIR, "goodware")
MAX_WORKERS = 20  # Number of concurrent downloads


# Load hashes and associate with category
def load_hashes_with_category(file_path, category):
    with open(file_path, "r") as f:
        lines = f.readlines()
    hashes = [line.strip() for line in lines if line.strip()]
    return [(sha256, category) for sha256 in hashes]


# Get already downloaded SHA256s by reading APK directories
def get_downloaded_sha256s(apk_dirs):
    downloaded_sha256s = set()
    for apk_dir in apk_dirs:
        if os.path.exists(apk_dir):
            for filename in os.listdir(apk_dir):
                if filename.endswith(".apk"):
                    sha256 = filename[:-4]  # Remove '.apk' extension
                    downloaded_sha256s.add(sha256)
    return downloaded_sha256s


# Download APK
def download_apk(args):
    sha256, category, api_key = args
    # Determine the output directory based on category
    output_dir = MALWARE_DIR if category == "malware" else GOODWARE_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Download the APK
    params = {"apikey": api_key, "sha256": sha256}
    try:
        response = requests.get(DOWNLOAD_URL, params=params, stream=True)
        if response.status_code == 200:
            output_file = os.path.join(output_dir, f"{sha256}.apk")
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return sha256, True
        else:
            print(f"Failed to download {sha256}: HTTP {response.status_code}")
            return sha256, False
    except Exception as e:
        print(f"Error downloading {sha256}: {e}")
        return sha256, False


# Concurrent download function
def download_concurrently(apk_list, downloaded_sha256s, api_key, max_workers):
    sha256s_to_download = [
        (sha256, category, api_key)
        for sha256, category in apk_list
        if sha256 not in downloaded_sha256s
    ]

    total_to_download = len(sha256s_to_download)
    print(f"{total_to_download} APKs to download.")

    if total_to_download == 0:
        print("No APKs to download. Exiting.")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_sha256 = {
            executor.submit(download_apk, args): args[0] for args in sha256s_to_download
        }

        with tqdm(total=len(future_to_sha256), desc="Downloading APKs") as progress:
            for future in as_completed(future_to_sha256):
                sha256 = future_to_sha256[future]
                try:
                    result_sha256, success = future.result()
                    # Since we are not using a log file, we don't need to log here
                    pass
                except Exception as e:
                    print(f"Error downloading {sha256}: {e}")
                progress.update(1)


# Main function
def main():
    # Input text files containing hashes
    malware_hashes_file = "malware_hashes.txt"
    goodware_hashes_file = "goodware_hashes.txt"

    # Ensure output directories exist
    os.makedirs(MALWARE_DIR, exist_ok=True)
    os.makedirs(GOODWARE_DIR, exist_ok=True)

    # Load hashes and assign categories
    print("Loading APK hashes...")
    malware_hashes = load_hashes_with_category(malware_hashes_file, "malware")
    goodware_hashes = load_hashes_with_category(goodware_hashes_file, "goodware")

    # Combine all hashes
    all_hashes = malware_hashes + goodware_hashes

    # Get already downloaded SHA256s by reading APK directories
    downloaded_sha256s = get_downloaded_sha256s([MALWARE_DIR, GOODWARE_DIR])

    # Calculate how many APKs need to be downloaded
    total_to_download = len(
        [sha for sha, _ in all_hashes if sha not in downloaded_sha256s]
    )
    print(f"{len(downloaded_sha256s)} were downloaded ")
    print(f"{total_to_download} APKs to download.")

    # Download APKs concurrently
    download_concurrently(all_hashes, downloaded_sha256s, API_KEY, MAX_WORKERS)


if __name__ == "__main__":
    main()
