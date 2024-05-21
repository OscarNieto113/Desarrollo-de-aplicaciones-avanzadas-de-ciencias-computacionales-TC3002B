# One Piece Character Classifier

## About Dataset
The dataset used for this project contains images of characters from the anime and manga series One Piece. The dataset is structured with each character's images stored in a separate folder, named after the character. This structure makes it convenient for organizing and processing the dataset for training a model to classify One Piece characters.

## Goal
The goal of this project is to develop an image classifier capable of identifying characters from One Piece. The classifier will be trained on the dataset mentioned above and will be able to classify images into the respective character classes.

## Dataset Preparation
Using separate sets for training, testing and validation is a fundamental practice for the following reasons:

**Model Performance Evaluation:** One of the primary goals of machine learning is to build models that generalize well to unseen data. By splitting your dataset into three subsets, we can assess your model's performance accurately.

**Identify Overfitting:** The test set serves as a benchmark to check for overfitting. If a model performs well on the training data but poorly on the test data, it's a sign of overfitting.

**Hyperparameter Tuning:** To optimize the model's hyperparameters, we need a validation set. You can train multiple models with different hyperparameters and select the one that performs best on the validation set

- **Train (70%)**: This category will be used to train the model and will contain approximately 70% of the images of each character.

- **Validation (15%)**: The validation category is used to evaluate the model's performance during training. It helps in adjusting the model's hyperparameters to prevent overfitting or underfitting. The validation category will contain around 15% of the images.

- **Test (15%)**: The test category is used to evaluate the model's performance on unseen data. It also contains around 15% of the images of each character.

## Data Preprocessing
Data preprocessing is a crucial step in preparing the images for training a classification model.:

- **Resizing Images**: All images are resized to 150x150 pixels to ensure consistent input size for the model, regardless of the original image dimensions.

- **Normalization**: The pixel values of the images are normalized to the range [0, 1] by dividing each pixel value by 255.

- **Data Augmentation**: Data augmentation involves applying various transformations to the existing dataset to create new, diverse training examples. In the context of garbage classification, these transformations typically include rotations, flips, shifts, zooms, and changes in brightness. The goal is to artificially expand the training dataset, providing the model with more varied examples to learn from.

- **Training Dataset Only:** Data augmentation is typically applied to the training dataset. This ensures that the model learns from augmented examples during training but is evaluated on unaltered data during validation and testing.

- **Evaluation on Unaltered Data:** The validation and test datasets remain untouched by data augmentation. This separation is crucial for assessing how well the model generalizes to real-world, unaltered scenarios.


- **Class Mode**: The class mode parameter specifies the format of the labels generated for the data. In this case, since the model is intended to classify various classes, the class parameter is set to categorical.

## How to Download the Dataset
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier?select=Data). Please note that the images in this dataset are not owned by the creator of this project.

## How to use the program
1. Download the dataset
2. Extract all the files
3. Copy and Paste the folder called /Data in to the folder of the repository Artificial Intelligence
4. Uncomment the function setup_and_split_dataset from the code
5. Run the code