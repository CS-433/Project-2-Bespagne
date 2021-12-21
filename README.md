# Class Project 2: Fear-Decoding-in-Rodents
## Machine learning (CS-433) class project, in cooperation with the Integrated Neurotechnologies Laboratory (INL) at EPFL
Welcome! This repository contains the code regarding class project 2 of the Machine Learning course (CS-433) at EPFL.
## Team members:
Feel free to contact any of us in case you have questions:
 * Tomas Brants (tomas.brants@epfl.ch, @tbrants)
 * Arthur Deleu (arthur.deleu@epfl.ch, @ArthurDeleu)
 * Alicia Gomez Soria (alicia.soriagomez@epfl.ch, @aliciasoria)

## General remark:
The raw data, obtained from the INL is not in this GitHub repository, as these files are too big in size. Hence, all data and labels needed to run the ML-models in Python, are provided as .npy files and a .csv file respectively.

## Folder informations:
1) The folder 'Matlab codes' is informative and used to compute the BLA and IL power spectra, along with the PAC power spectra.
2) A visualization of the data (upper 6 figures) and -more importantly- the extracted features (lower 3 figures) can be found in the folder 'Visualization data'.
3) The folder 'Data' contains all the data (.npy-files and .csv-file) to run the Jupyter Notebooks.
  * 'Feature arrays' are the input data for the 1-channel CNN.
  * 'Labels' contains the .csv file with smoothed bar press data and will be binned in Python to create class labels.
  * 'PAC_afterCNN.npy' contains the feature vector after running the 1-channel CNN on the PAC colormaps.
  * 'input_MLP' contains the IL and BLA power spectra, which are feeded into the MLP together with 'PAC_afterCNN.npy' data.
4) The folder 'Jupyter notebooks' contains all the notebooks relevant for the machine learning models.
  * 'CNN_main.ipynb' is the notebook linked to the 1-channel CNN with input the PAC-features. It also contains the baseline model.
  * 'MLP.ipynb' is the notebook linked to the MLP with input the CNN feature vector and the BLA and IL power spectra.
  * The other notebooks were used to generate and save the data/features as .npy-files and don't need to be ran.

## How to use the code:
1) Install Python version 3.7. and the Python libraries Pytorch, Numpy, sickit-learn and Pandas.
2) Run 'run.py'
