# NCM_ML

# Descriptions: 
This repository contains all the R codes written specifically for contrusting  machine learning models to predict the initial dischagre capacity (IC) and 50th cycle end discharge capacities of 169 doped Nickel-Cobalt-Managnese cathode materials. This is given to aid the dicussion in the manuscript " Machine Learning Approach for Predicting the Discharging Capacities of Doped Nickel-Cobalt-Manganese Cathode Materials in Li-ion Battery"

# Table of Content:
In this repository, six different non-linear machine learning techniques are written in Python:

Non-linear models:

- Neural_Network.ipynb
- Gradient_Boosting_Machine.ipynb
- Kernel Ridge Regression.ipynb
- K-nearest-Neighbour.ipynb
- RandomForestRegressor.ipynb
- Support_Vector_Machine.ipynb

Dataset:

NMC_numerical_new.csv

# Installation:

Operating system: windows 10, 64bit Software: Anaconda version 1.10. 0, Python version 3.7

For the required Python libraries, please see each the first line discussion in each code file


# Instructions for use and expected output

The Codes for optimising the hyperparameters are also included in the files as the reference to the labelled optimised values.

Please go ahead to the training and testing section for quick access of the models' results

## Input: NMC_numerical.csv

1st column is the ratio of **Lithium** in the material formula (Li)

2nd column is the ratio of **Nickel** in the material formula (Ni)

3nd column is the ratio of **Cobalt** in the material formula (Co)

4th column is the ratio of **Managanese** in the material formula (Mn)

5th column is the ratio of **dopant** in the material formula (M)

6th column is the lattice constant-**a** of the material's crystal structure, obtained from the reported X-ray diffraction spectrums (LC_a)

7th column is the lattice constant-**c** of the material's crystal structure, obtained from the reported X-ray diffraction spectrums (LC_c)

8th column is the crystal volume the material's crystal structure (CV)

9th column is the current density applied for both charging and discharging the battery (CD)

10th column is the minimum operating voltage (V_min)

11th column is the maximimum operating voltage (V_max)

12th column is the molar mass of the dopant (Mr_dopant)

13th column is the molar mass of the material (Mr)

14th column is the number of electrons of the dopant (No_electron_M)

15th column is the electronegativity of the dopant element (EN_M)

16th column is the number of isotopes of the dopant element (No_iso_dopant)

17th column is the first ionization energy for the dopant element (E_ion_dopant)

18th column is the electron affinity of the dopant element (EA_dopant)

19th column is the atomic radius of the dopant element (AR_dopant)

20th column is the ionic radius of the dopant element (IR_dopant)

21th column is the initial discharge capacities (IC)

22nd column is the 50th cycle end discharge capacities (EC)



## Expected output
Root_mean_Square_error and coefficient of determinations for the test-set **regression** prediction for both IC and EC


