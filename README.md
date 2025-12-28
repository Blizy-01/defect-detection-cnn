defect-detection-cnn
Automated Surface Defect Detection using Deep Learning (ICT423 - Group 9 Project)


# Automated Metallic Surface Defect Detection using Deep CNNs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![Accuracy](https://img.shields.io/badge/Accuracy-95.49%25-brightgreen)

# Project Overview
This repository contains the implementation of a Custom Convolutional Neural Network (CNN) designed to automate the detection of surface defects on hot-rolled steel strips.

Developed as a research project for ICT423 at Bells University of Technology, this system aims to replace manual visual inspection in manufacturing with a real-time, deep learning-based solution. The model classifies defects into six distinct categories with over 95% accuracy, utilizing data augmentation to handle industrial data scarcity.

# Key Features
* Custom Architecture: A lightweight CNN trained entirely from scratch (no pre-trained weights).
* Data Augmentation: Real-time geometric transformations (rotation, flip, zoom) to prevent overfitting.
* High Performance: Achieved **95.49% validation accuracy** on the NEU dataset.
* Modular Design: Clean separation of training, evaluation, and data handling scripts.

# Dataset
We utilize the NEU Surface Defect Database (Northeastern University).
* Classes: 6 (Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches).
* Original Size: 1,800 grayscale images (200x200).
* Input Shape: Resized to 128x128 for the model.

> **Note:** Due to GitHub file size limits, the raw data is not included in this repository.
> 📥 **[Download the NEU Dataset Here](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)**

# Installation & Setup

# 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/defect-detection-cnn.git](https://github.com/YOUR_USERNAME/defect-detection-cnn.git)
cd defect-detection-cnn
```
# 2. Install Dependencies
pip install -r requirements.txt

# 3. Data Setup
 * Download the dataset from the link above.
 * Extract the files so your folder structure looks exactly like this:
data/
└── raw/
    ├── Cr/  (Crazing)
    ├── In/  (Inclusion)
    ├── Pa/  (Patches)
    ├── PS/  (Pitted_Surface)
    ├── RS/  (Rolled_in_Scale)
    └── Sc/  (Scratches)

# 4. Usage
# Train the Model 
To train the CNN from scratch using the settings defined in the report (15 Epochs):

python src/train.py
This will generate and save the model file to saved_models/defect_classifier.h5.

# Evaluate the Model
To test the saved model and see the final accuracy score without retraining:
python src/evaluate.py

# Project Sturcture
defect-detection-cnn/
├── data/                  # Data folder (Ignored by Git)
├── notebooks/             # Jupyter Notebooks for exploration and plotting
│   └── Main_Model_Training.ipynb
├── saved_models/          # Stores trained .h5 models (Ignored if large)
├── src/                   # Source code
│   ├── train.py           # Main training script (Data Aug + CNN + Training)
│   └── evaluate.py        # Inference script to check accuracy
├── .gitignore             # Specifies files to exclude (Data/Models)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

# 5. RESULTS
The model was evaluated on a held-out validation set (20% of data).

Training Accuracy: ~91.4%

Validation Accuracy: 95.49%

Loss: 0.13

These results demonstrate that a custom, lightweight CNN can effectively differentiate between complex metallic texture defects without requiring heavy transfer learning models.

# 6. TEAM MEMBERS
ONECHOJON BLESSING 2022/11420 [GROUP LEADER]
ADEKOLA OLUWASEGUN 2022/11728


This project is submitted in partial fulfillment of the requirements for the course ICT423.
