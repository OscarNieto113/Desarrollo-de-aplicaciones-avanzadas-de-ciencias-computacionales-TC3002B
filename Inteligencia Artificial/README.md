# One Piece Character Classifier

## About Dataset
The dataset used for this project contains images of characters from the anime and manga series One Piece. The dataset is structured with each character's images stored in a separate folder, named after the character. This structure makes it convenient for organizing and processing the dataset for training a model to classify One Piece characters.

## Goal
The goal of this project is to develop an image classifier capable of identifying characters from One Piece. The classifier will be trained on the dataset mentioned above and will be able to classify images into the respective character classes.

## Dataset Preparation
To prepare the dataset for training, it will be split into three main categories: train, test, and validation. This division ensures that the model is trained on a diverse set of images and evaluated on unseen data to measure its performance accurately.

- **Train (70%)**: This category will be used to train the model and will contain approximately 70% of the images of each character.

- **Validation (15%)**: The validation category is used to evaluate the model's performance during training. It helps in adjusting the model's hyperparameters to prevent overfitting or underfitting. The validation category will contain around 15% of the images.

- **Test (15%)**: The test category is used to evaluate the model's performance on unseen data. It also contains around 15% of the images of each character.

## Data Preprocessing
Data preprocessing is a crucial step in preparing the images for training a classification model. In this project, the following preprocessing steps are applied using TensorFlow's ImageDataGenerator:

- **Resizing Images**: All images are resized to 150x150 pixels to ensure consistent input size for the model, regardless of the original image dimensions.

- **Normalization**: The pixel values of the images are normalized to the range [0, 1] by dividing each pixel value by 255.

- **Data Augmentation**: Data augmentation is applied to increase the diversity of the training dataset. The following transformations are applied to the training set:

  - Rotation: Images can be rotated up to 90 degrees.
  - Width and height shift: Images can be horizontally and vertically shifted up to 20% of their size.
  - Shear: The shear transformation distorts the shape of an image. The shear transformation can vary randomly up to a maximum of approximately 0.2 radians (11.46 degrees).
  - Zoom: Images can be zoomed in by up to 20%.
  - Horizontal flip: Images can be horizontally flipped.

- **Class Mode**: The class mode parameter specifies the format of the labels generated for the data. In this case, since the model is intended to classify various classes, the class parameter is set to categorical.

A data generator is used to generate batches of image data with a batch size of 64 images per batch. For educational purposes, examples of augmented images are saved in the "augmented" folder with the prefix "aug" and in PNG format.

## How to Download the Dataset
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier?select=Data). Please note that the images in this dataset are not owned by the creator of this project.
