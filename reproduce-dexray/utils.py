# This file contains the functions used to create the metadata 
# dictionary (maps hashes to time), as well as the functions 
# to create and save X, y and temp numpy arrays to files. 


import csv
import json
import os
from datetime import datetime

import numpy as np
import tensorflow.keras as keras
from tqdm import tqdm

from tesseract.evaluation import predict
from tesseract import evaluation, temporal, metrics

APK_METADATA_PATH = "/scratch/users/mbenali/metadata.csv"
NUMPY_FILES_DIR = "/scratch/users/mbenali/download_apk/100k_download/npy"
# Where to save all training and testing data after splitting 
DATA_DIR = "./data/dexray_tesseract"
YEAR_START = 2010
YEAR_END = 2022

def load_apk_metadata(metadata_json_dir):
    """
    Loads the APK metadata from the CSV file into a dictionary for quick lookups.
    Also saves the metadata dictionary to a JSON file for future use.
    """
    metadata = {}
    with open(APK_METADATA_PATH, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=[
            'sha256', 'sha1', 'md5', 'date_time', 'number1',
            'package', 'number2', 'number3', 'dex_date', 'number4', 'source'
        ])
        csv_file.seek(0)  # Reset file pointer to the beginning
        for row in reader:
            metadata[row['sha256']] = row['dex_date']

    # Save metadata to JSON file
    with open(metadata_json_dir, 'w') as json_file:
        json.dump(metadata, json_file)
    print(f"Metadata saved to {metadata_json_dir}")

    return metadata

def assemble_arrays(metadata):
    """Assembles numpy arrays from .npy files along with their labels and dates."""
    X = []
    y = []
    temp = []

    dirs = [NUMPY_FILES_DIR + '/malware', NUMPY_FILES_DIR + '/goodware']

    for directory in dirs:
        if not os.path.exists(directory):
            continue

        # List files and wrap with tqdm for a progress bar
        files = [file for file in os.listdir(directory) if file.endswith('.npy')]
        if "malware" in directory:
            files = files[:10000]
        else:
            files = files[:40000]
        for file in tqdm(files, desc=f"Processing {directory}"): 
            # if stop >= 50000:
            #     break
            # stop += 1

            if not file.endswith('.npy'):
                continue

            # Obtain the date
            apk_date = _get_date_time_from_hash(file[:-4], metadata)  # remove .npy

            # Do not include if outside of date range
            if apk_date is None or apk_date.year < YEAR_START or apk_date.year > YEAR_END:
                continue

            temp.append(apk_date)

            filepath = os.path.join(directory, file)
            array = np.load(filepath)
            X.append(array.flatten())  # Flattening ensures all arrays are rows

            # Obtain the label 
            y.append(0 if 'goodware' in directory else 1)

            # Append filename to the training_data_used.txt
            with open("./data/training_data_used.txt", 'a') as f:
                f.write(f"{file[:-4]}\n")

        print(f"Done with {directory}")


    return np.stack(X), np.array(y), np.array(temp)


def _get_date_time_from_hash(search_hash, metadata):
    """
    Retrieves the `dex_date` for the given hash using preloaded metadata."""
    if search_hash in metadata:
        try:
            return datetime.strptime(metadata[search_hash], '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            print(f"Error parsing date for {search_hash}: {e}")
            return None
    return None