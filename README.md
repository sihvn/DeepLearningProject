# DeepLearningProject

This project aims to develop a deep learning model for detecting the presence of Pneumothorax in patients. 

# Dataset


The project uses dataset from [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data).

Download the dataset from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737) and save it under the folder **raw_data**.


# To setup environment:

```
pip3 install torch torchvision tensorboard torch-lr-scheduler glob2 pillow pandas
```

# Usage

## Preparing Dataset


The function **get_data_loaders** is located in file **preprocessing.py**. We imported this function into **Trainer.ipynb**.
Run the first two cells in **Trainer.ipynb** to generate the training, validation and test dataset.


## Exploratory Data Analysis


The data is then inspected in eda.ipynb, where we can look at examples of positive and negative cases,
and look at the correlation between selected non-image.


## Custom Model


The class **dense_net** is located in file **custom_densenet.py**. We imported this class into **Trainer.ipynb** in order to train the model.
Run the first four cells in Trainer.ipynb to view the model structure.


## Training the model


Run all cells in **Trainer.ipynb** to train the model.
Some training hyperparameters are exposed in the train helper function that is defined in the **train_chexnet.py** file


## Evaluating Dataset


Run all cells in **evaluation.ipynb**  to get the F1 score on the held-out test dataset, and view examples of some of the errors (False Negative, False Positive)


# Result

  | Learning Rate | Batch Size | Optimizer | Criterion | Epochs | Train F1 Score | Validation F1 Score | Test F1 Score |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 1e-4  | 16  | Adam  | BCELoss  | ?  | ?  | ?  | ?  |

# Project Report

[Report]()

# Project Directory:  

```
├── .venv
├── 121-layer
|   ├── src   
|   |    ├── custom_densenet.py
|   |    ├── preprocessing.py
|   |    ├── train_densenet.py
├── notebooks 
|   ├── evaluation.ipynb
|   ├── trainer.ipynb
├── raw_data
|   ├── archive
|   |   ├── images_001
|   |   ├── etc...
```
