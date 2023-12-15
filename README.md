# Predictive Modeling of Loan Repayment Behaviour

## Overview

"Welcome to our Fungi Classification Project!"

🍄 DeepFungi: Advanced Fungal Image Classification

The purpose of our project is to utilize advanced deep learning techniques to adeptly classify various fungi, providing a robust method for identifying fungal infections early and accurately. The goal of this project is to provide a cutting-edge tool that enhances accuracy in multi-classifying fungi, revolutionizing the process of diagnosing fungal infections.

## Dataset

[https://www.archive.ics.uci.edu/dataset/773/defungi](https://www.archive.ics.uci.edu/dataset/773/defungi)

## Github Link

[https://github.com/Vijen712/Multi-Classify_Microscopic_Fungal_Images](https://github.com/Vijen712/Multi-Classify_Microscopic_Fungal_Images)

## Requirements

* Make sure your local device has Python version 3.8.3 and above installed
* The system device should have graphviz correctly installed and its binaries are available in the system PATH. You can check the version of graphviz using dot -V in terminal.
* To ensure that there is no package conflict issue, create a Virtual Environment(venv). Instructions on how to do that are given in steps to run project below.

## Deployment

### Steps to run the project in MacOS

* Download the zip file and unzip it, the folder structure is Code, project report, Readme file.
* Download data from the dataset link above. Unzip the dataset folder named `defungi` and save that folder in the Code folder, which contains all the .py files.
* The first step is to open a terminal and go to the code folder path `(your_path/code)`.
* The next step is to create a venv for downloading packages, in your terminal run `python3 -m venv venv` followed by `source venv/bin/activate` followed by `pip3 install -r requirements.txt` (make sure before this step you have done the previous one).
* Once all the packages are installed we can start the project by using the `python3 main.py` command in terminal (your_path/code).
* Close the popup images to continue the script, moreover, there will be instruction like `press enter to continue` in your terminal to proceed with the next model or graph. This is kept so that the transition between models does not freeze the popup images. Keep repeating that step till all the models are run.
* Once the project runs, at each modeling step the model architecture, training vs validation loss, training vs validation accuracy, ROC curve, and confusion matrix will be saved in a new folder named results in the code folder.

### Steps to run the project in Windows

* Download the zip file and unzip it, the folder structure is Code, project report, Readme file.
* Download data from the dataset link above. Unzip the dataset folder named `defungi` and save that folder in the Code folder, which contains all the .py files.
* The first step is to open a terminal and go to the code folder path `(your_path/code)`.
* The next step is to create a venv for downloading packages, in your terminal run `python -m venv venv` followed by `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` and `.\venv\Scripts\Activate`.
* Next apply this command `pip install -r requirements.txt` in terminal to download all the packages.
* Once all the packages are installed we can start the project by using the `python main.py` command in terminal (your_path/code).
* Close the popup images to continue the script, moreover, there will be instruction like `press enter to continue` in your terminal to proceed with the next model or graph. This is kept so that the transition between models does not freeze the popup images. Keep repeating that step till all the models are run.
* Once the project runs, at each modeling step the model architecture, training vs validation loss, training vs validation accuracy, ROC curve, and confusion matrix will be saved in a new folder named results in the code folder.

## File Description

**main.py** : if main.py is called it will conduct the data preprocessing and data splitting, followed by modeling and evaluation, so main.py calls data_handler.py , model.py & train_evaluate.py

**data_handler.py** : Defines a class DataHandler that encapsulates data preprocessing steps, data augmentation for class balancing & data splitting

**model.py** : It contains all the model archeticture code for MiniVGG, MiniVGG_var, ResNet50_Var, DesnsNet121_Var, DenseNet121_Var1, which is called by main .py

**train_evaluation.py**: It contains the code for the parameters which are used by each models and also stores values for plotting ROC curves, confusion metrics etc., called by main.py

## Contributions

**Business Understanding, Data Collection**- Charan, Kunal, Vijen

**Data Preprocessing** - Kunal, Vijen, Charan

**Modeling** - Vijen, Charan, Kunal

**Evaluation** - Kunal, Vijen , Charan

**Modular programming, IDE, Github** - Vijen, Kunal, Charan

**PPT and Report** - Charan, Vijen, Kunal
