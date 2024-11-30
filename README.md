This repository is a fork of the original Tesseract repository, which can be
found here. 

Tesseract is a framework to make sure that classifiers for Android malware
classification are not biased by enforcing the following three constraints: 

1. Temporal Training Consistency
2. Temporal Goodware/Malware Consistency
3. Realistic Testing Classes Ratio

Further details can be found in the paper *TESSERACT: Eliminating 
Experimental Bias in Malware Classification across Space and Time*. F. 
Pendlebury, F. Pierazzi, R. Jordaney, J. Kinder, and L. Cavallaro.  USENIX
Sec 2019. Check also `https://s2lab.cs.ucl.ac.uk/projects/tesseract` for
up-to-date information on the project, e.g., a talk at USENIX Enigma 2019
at `https://www.usenix.org/conference/enigma2019/presentation/cavallaro`.

---

# Reproducing TESSERACT results

This repository replicates the findings of Tesseract on the two established
Android malware classifiers shown in the paper: 

* Drebin 
* MaMaDroid

To do this, we used the scripts provided by Tesseract with slight modifications.
The results on Drebin can be found in
`notebooks/tesseract-reproduce-drebin.ipynb` and the results on MaMaDroid in
`tesserat-reproduce-mamadroid.ipynb`. 

We have also reproduced Figure 6 shown of the paper, showing [TODO]. The code
for this is in `notebooks/tesseract-reproduce-figure6.ipynb`. 

## Using DexRay with TESSERACT 

DexRay is an algorithm proposed in the paper *Dexray: a simple, yet
effective deep learning approach to android malware detection based on image
representation of bytecode.* (2021). Daoudi, N., Samhi, J., Kabore, A. K., Allix, K.,
Bissyandé, T. F., & Klein, J. In Deployable Machine Learning for Security
Defense: Second International Workshop, MLHat 2021, Virtual Event, August 15,
2021, Proceedings 2 (pp. 81-106). Springer International Publishing.

DexRay proposes a novel approach to Android malware detection by proposing an
image classifier. The authors turned the Android APKs into 128*128 grayscale
images representing the bytecode of the app source code. With these images, they
trained a Keras classifier and report a [] F1-Score. 

In their paper, they have tested the classifier taking the first Tesseract
constraint into consideration. Surprisingly, the F1-Score is even higher when
enforcing Temporal Training Consistency. 

We have reproduced these results by obtaining the same APKs as used in the
DexRay paper from AndroZoo, converting them into their grayscale bytecode image
representation and training the DexRay algorithm with these. Due to resource
constraints, we decreased our dataset to 50.000 APKs instead of [TODO] APKs. 

We found results in the same direction as DexRay: When using Tesseract's
time-aware train test split, the average F1-Score is higher than doing a random
split. The code for this replication can be found in the `reproduce-dexray`
folder. 

### Reproduce-Dexray Folder Content 

* `/data`: contains X, y and temp numpy files containing the numpy arrays
  corresponding to the grayscale images for our 50.000 APKs in X, the labels in
  y and the timestamps in temp. 
* `training_data_hashes.txt`: contains a list of the hashes of our 50.000 APKs
  used 
* `dexray_base/`: contains `X_train, X_test, y_train, y_test` files of numpy
  arrays after random splitting (ie. the same arrays as in the parent folder's
  X and y files, but split into training and testing data)
* `dexray_tesseract/`: contains the same data as the `dexray_base/` folder but
  using the tesseract split instead of a random split. 