This repository is a fork of the original Tesseract repository, which can be found [here](https://github.com/s2labres/tesseract-ml-release?tab=readme-ov-file).

Tesseract is a framework designed to ensure that classifiers for Android malware classification are not biased, by enforcing the following three constraints:

1. Temporal Training Consistency  
2. Temporal Goodware/Malware Consistency  
3. Realistic Testing Classes Ratio  

Further details can be found in the paper *TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time*, by F. Pendlebury, F. Pierazzi, R. Jordaney, J. Kinder, and L. Cavallaro, presented at USENIX Security 2019. For up-to-date project information, including a talk at USENIX Enigma 2019, visit [this link](https://s2lab.cs.ucl.ac.uk/projects/tesseract) or [USENIX Enigma 2019 Presentation](https://www.usenix.org/conference/enigma2019/presentation/cavallaro).

---

## Part 1: Reproducing TESSERACT Results

This repository replicates the findings of Tesseract on two established Android malware classifiers, as discussed in the paper:

- **Drebin**  
- **MaMaDroid**
- **Deep Learning**

To reproduce `Figure 5 - Time Decay`, which demonstrates the performance of these algorithms over a 22-month testing period, we used the scripts provided by Tesseract with slight modifications:  
- Results for Drebin performance can be found in `notebooks/tesseract-reproduce-drebin.ipynb`.  
- Results for MaMaDroid performance can be found in `notebooks/tesseract-reproduce-mamadroid.ipynb`.  
- Results for Deep NN performance can be found in `notebooks/tesseract-reproduce-dl.ipynb`.  

Additionally, we replicated `Figure 6 - Tuning Improvement`, which shows the impact of different malware percentages during the training period on each algorithm. This analysis is available in `notebooks/tesseract-reproduce-figure6.ipynb`.

---

## Part 2: Reproducing DexRay and Applying TESSERACT  

DexRay is an algorithm proposed by the University of Luxembourg's TruX research group in the paper *DexRay: A Simple, Yet Effective Deep Learning Approach to Android Malware Detection Based on Image Representation of Bytecode* (2021) by N. Daoudi, J. Samhi, A. K. Kabore, K. Allix, T. F. Bissyandé, and J. Klein, published in *Deployable Machine Learning for Security Defense: Second International Workshop, MLHat 2021*.  

DexRay introduces a novel approach to Android malware detection by converting Android APKs into 128x128 grayscale images representing the bytecode. These images are then used to train a classifier, with results reported using the F1-Score metric.  

Interestingly, the paper tested classifiers using the first Tesseract constraint (Temporal Training Consistency) and found that F1-Score improved when this constraint was enforced.  

We replicated these results by obtaining 100K APK hashes used from the AndroZoo dataset. We created scripts that can be found in `reproduce-dexray/utils` to download the APKs, convert them to images, and process them into numpy arrays (requiring 4.2 TB of data, thank you HPC). Due to resource constraints and the time it took to train the model, we reduced the dataset from 100K to 50K APKs.  

Our findings align with DexRay's when using 20% malware in training and testing data, using Tesseract's time-aware train-test split yielded higher average F1-Scores compared to random splits. When using 10% malware in the training and testing set, DexRay performs worse on the time-aware split. The code to create DexRay models is in `reproduce-dexray`, and the demonstration of results is available in `notebooks/tesseract-reproduce-dexray.ipynb`.

---

## `reproduce-dexray` Folder Content

- **`data`**: Contains `X`, `y`, and `temp` numpy files for 50K APKs. `X` includes grayscale images, `y` contains labels, and `temp` stores timestamps.  
- **`training_data_hashes.txt`**: A list of 50,000 APK hashes used for training.  
- **`dexray_base/`**: Contains `X_train`, `X_test`, `y_train`, and `y_test` numpy arrays created using random splitting.  
- **`dexray_tesseract/`**: Contains equivalent numpy arrays, but split using the Tesseract methodology.  
- **`utils/`: Includes the following scripts:  
  - `download_apk.py`: Downloads APKs based on DexRay repository hashes.  
  - `apk2image.py`: Converts APKs to grayscale images.  
  - `images2np.py`: Converts images to numpy arrays for faster training.  

---

## Description of the Remaining Folders

- **`data`**: Generated using the `make data` command in the `Makefile`, containing the Drebin dataset. The `make mamadroid` command can be used to download the MamaDroid dataset.  
- **`examples`**: Demonstrates the usage of the Tesseract library with various code examples.  
- **`models`**: Includes models trained with the DexRay algorithm, both with and without Tesseract constraints.  
- **`notebooks`**: A space for Jupyter notebooks for detailed analysis and experimentation.  
- **`tesseract` & `tessera.egg-info`**: Contains the Tesseract library implementation and associated metadata.  
- **`test`**: Provides unit tests to ensure the reliability and functionality of the Tesseract library.  

--- 
