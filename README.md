# CS534-Emory
 Final project of CS534-Machine learning in Emory, 2023 Fall

 Group 17: Yuzhang Xie, Siyu Li

 Title: Online Hate Speech Detection

# Dataset
 Kaggle link: https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech

# Environment
 Python version: 3.8.10
 Python packages' version: Specified in the "requirements.txt" file

# File explanation
 "dataset" folder: This directory stores both the original dataset sourced from Kaggle and the generated embeddings. Due to their substantial size, the dataset files and the embeddings are not included in this repository. Additionally, the "create_dataset.py" file contains Python code responsible for reading the generated embeddings and preparing them for the training process.

 "docs" folder: Within this folder, you will find code files used for checking the dataset and generating embeddings, which are utilized before the actual training process.

 "engine" folder: This directory stores core code files essential for running the training process and initializing machine learning models. Specifically, "train_test_sklearn.py" is employed for training traditional machine learning models utilizing the sklearn package, while "train_test_torch.py" is dedicated to training deep learning models using the PyTorch package. Additionally, the "models" folder stores files defining specific machine learning models.

 "setting" folder: This directory contains setting files pertinent to each training process.

"main.py": The primary Python file responsible for initializing the training process.


 