# Master Thesis

Welcome to the repository for my Master Thesis. This repository contains all the code, data, and resources used in my thesis project, which focuses on the analysis and classification of EEG signals.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Introduction
This project is dedicated to exploring advanced techniques for processing and classifying EEG signals. The primary focus is on utilizing machine learning models such as EEGNet, S-EEGNet, EEGInception, and EfficientNetB7 to discriminate between different emotional states. The project also involves preprocessing EEG data, extracting features using the Hilbert-Huang Transform (HHT) and Continuous Wavelet Transform (CWT), and comparing the performance of various models.

## Project Structure
The repository is structured as follows:
```
.
├── data
│   ├── raw
│   ├── processed
├── models
│   ├── EEGNet
│   ├── S-EEGNet
│   ├── EEGInception
│   ├── EfficientNetB7
├── notebooks
│   ├── data_preprocessing.ipynb
│   ├── feature_extraction.ipynb
│   ├── model_training.ipynb
├── results
│   ├── figures
│   ├── logs
├── scripts
│   ├── preprocess_data.py
│   ├── train_model.py
├── README.md
├── requirements.txt
```

## Data
The data used in this project consists of EEG signals recorded from multiple channels. The data is preprocessed to remove noise and artifacts, segmented into epochs, and organized into trials. The final dataset is shaped as `(7548, 22, 600)`, representing trials, channels, and samples, respectively.

## Models
The models implemented in this project include:
- **EEGNet**: A compact convolutional neural network designed for EEG signal classification.
- **S-EEGNet**: A variant of EEGNet with additional layers and modifications.
- **EEGInception**: A neural network architecture inspired by Inception modules, tailored for EEG data.
- **EfficientNetB7**: A pre-trained model adapted for processing EEG signals.

## Results
The performance of each model is evaluated using accuracy on the validation set. Hyperparameter tuning is performed using RandomSearch. The results are documented in the `results` directory, including figures and logs of the training process.

## Usage
To run the code in this repository, follow these steps:
1. Clone the repository:
   ```
   git clone https://github.com/LukeCepza/MasterThesis.git
   ```
2. Navigate to the repository directory:
   ```
   cd MasterThesis
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Preprocess the data:
   ```
   python scripts/preprocess_data.py
   ```
5. Train the model:
   ```
   python scripts/train_model.py
   ```

## Requirements
The required Python packages are listed in `requirements.txt`. The primary dependencies include:
- TensorFlow 2.14.0
- Keras
- NumPy
- SciPy
- Matplotlib

## Acknowledgements
I would like to thank my advisor and the research team for their support and guidance throughout this project. 

## Contact
For any questions or inquiries, please contact Luis Kevin Cepeda Zapata at lkcepza@gmail.com.
