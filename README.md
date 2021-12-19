# Class Project 2: Fear-Decoding-in-Rodents
## Machine learning (CS-433) class project, in cooperation with the Integrated Neurotechnologies Laboratory (INL) at EPFL
## Team members:
Tomas Brants (tomas.brants@epfl.ch), Arthur Deleu (arthur.deleu@epfl.ch), Alicia Gomez Soria (alicia.soriagomez@epfl.ch)

## General remark:
The raw data, obtained from the INL is not in this GitHub repository, as these files are too big in size. Hence, all data and labels needed to run the ML-models in Python, are provided as .npy files and a .csv file respectively.

## Folder informations:
1) The folder 'Matlab codes' is informative and used to compute the BLA and IL power spectra, along with the PAC power spectra.
2) A visualization of the data and the extracted features can be found in the folder 'Visualization data'.
3) The folder 'Data' contains all the data (.npy-files and .csv-file) to run the Jupyter Notebooks.
4) The folder 'Jupyter notebooks' contains all the notebooks relevant for the machine learning models.
  * 'CNN_main.ipynb' is the notebook linked to the 1-channel CNN with input the PAC-features.
  * 'MLP.ipynb' is the notebook linked to the MLP with input the CNN feature vector and the BLA and IL power spectra.
  * The other notebooks were used to generate and save the data/features as .npy-files and don't need to be ran.

## How to use the code:
1) Install Python version 3.7.
2) Run 'run.py'
