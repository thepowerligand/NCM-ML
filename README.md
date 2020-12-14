# NCM-ML

# Descriptions: 
This repository contains all the R codes written specifically for contrusting  machine learning models to predict the initial dischagre capacity (IC) and 20th cycle end discharge capacities of 169 doped Nickel-Cobalt-Managnese cathode materials. This is given to aid the dicussion in the manuscript " Machine Learning Approach for Predicting the Discharging Capacities of Doped Nickel-Cobalt-Manganese Cathode Materials in Li-ion Battery"

# Table of Content:
In this repository, six different non-linear machine learning techniques are written in Python:

Non-linear models:

- Artificial Neural Network.R
- SVM_codes.R (Support Vector Machine)
- Random_forest.R
- Decision_tree.R
- Gradient_Boosting_Machine.R

Dataset:

LMO.csv

# Installation:

Operating system: windows 10, 64bit Software: R version 3.6.0

For the required R libraries, please see each the first line discussion in each code file


# Instructions for use and expected output

The Codes for optimising the hyperparameters are also included in the files as the reference to the labelled optimised values.

Please go ahead to the training and testing section for quick access of the models' results

## Input: LMO.csv

1st column is the ratio of dopant in the material formula (M)

2nd column is the ratio of managanese in the material formula (Mn)

3nd column is the electronegativity of the dopant element (M_EN)

4th column is the molar mass of the material (Mr)

5th column is the lattice constant a of the material's crystal structure, obtained from the reported X-ray diffraction spectrums (LC_a)

6th column is the current density applied for both charging and discharging the battery (CD)

7th column is the initial discharge capacities (IC)

8th column is the 20th cycle end discharge capacities (EC)



## Expected output
Root_mean_Square_error for the test-set **regression** prediction for both IC and EC


