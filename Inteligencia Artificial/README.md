# One Piece Character Classifier

## About Dataset
The dataset used for this project contains images of characters from the anime and manga series One Piece. The dataset is structured with each character's images stored in a separate folder, named after the character. This structure makes it convenient for organizing and processing the dataset for training a model to classify One Piece characters.

## Goal
The goal of this project is to develop an image classifier capable of identifying characters from One Piece. The classifier will be trained on the dataset mentioned above and will be able to classify images into the respective character classes.

## Papers utilizados en la implementación:
- [Image Classification Based On CNN: A Survey](https://www.researchgate.net/publication/355800126_Image_Classification_Based_On_CNN_A_Survey)
- [Binary cross entropy with deep learning technique for
Image classification](https://www.researchgate.net/profile/Vamsidhar-Yendapalli/publication/344854379_Binary_cross_entropy_with_deep_learning_technique_for_Image_classification/links/5f93eed692851c14bce1ac68/Binary-cross-entropy-with-deep-learning-technique-for-Image-classification.pdf)


1. Applied Preprocessing

## ImageDataGenerator for Training, Testing, and Validation

Three instances of `ImageDataGenerator` are created: one for training data (`train_datagen`), one for testing data (`test_datagen`), and one for validation data (`validation_datagen`). Each instance applies the following data augmentation transformations to the images:

- **rescale=1./255**: Normalizes pixel values to the range [0, 1] by dividing each pixel value by 255.
- **rotation_range=90**: Randomly rotates images up to 90 degrees.
- **width_shift_range=0.2**: Randomly shifts images horizontally.
- **height_shift_range=0.2**: Randomly shifts images vertically.
- **shear_range=0.2**: Randomly applies shearing transformations.
- **zoom_range=0.2**: Randomly zooms in on images.
- **horizontal_flip=True**: Randomly flips images horizontally.

### Justification
The transformations applied by `ImageDataGenerator` introduce variety into the training, testing, and validation datasets, which helps the model become more robust and generalize better. Normalizing pixel values ensures that the data is in a suitable range for the model, while transformations such as rotation, shifting, shearing, zooming, and flipping introduce variability that can enhance the model's ability to recognize different variations of the anime character images.

#### Image Data Generators

Image data generators are created for training (`train_generator`), testing (`test_generator`), and validation (`validation_generator`) using the corresponding data directories. All generators have a target size of `(150, 150)` to resize the images to 150x150 pixels. A batch size of 32 is specified for each generator.


- **Train (70%)**: This category will be used to train the model and will contain approximately 70% of the images of each character.

- **Validation (15%)**: The validation category is used to evaluate the model's performance during training. It helps in adjusting the model's hyperparameters to prevent overfitting or underfitting. The validation category will contain around 15% of the images.

- **Test (15%)**: The test category is used to evaluate the model's performance on unseen data. It also contains around 15% of the images of each character.

## [Data Preprocessing](https://www.geeksforgeeks.org/data-preprocessing-in-data-mining/)
Data preprocessing is a crucial step in preparing the images for training a classification model.:

- **Resizing Images**: All images are resized to 150x150 pixels to ensure consistent input size for the model, regardless of the original image dimensions.

- **Normalization**: The pixel values of the images are normalized to the range [0, 1] by dividing each pixel value by 255.

## 2. Algorithm Selection

### Character Classification as an Image Classification Problem

Classifying characters from One Piece can be considered an image classification problem, as it involves identifying and labeling the presence of specific characters within images. Image classification tasks are well-suited for Convolutional Neural Networks (CNNs), which are designed to recognize patterns and features in image data.

In this context, the goal is to accurately identify characters such as Luffy, Zoro, Nami, and others from the One Piece series in various images. This requires the model to learn the unique characteristics of each character, including facial features, clothing, and other visual attributes.

### Choosing Between Different Neural Network Models

The choice of neural network model for character classification depends on several factors, including the complexity of the character patterns, the amount of available data, and the requirements for model performance and efficiency.

- **Complexity of Character Patterns**: If the character patterns are distinct and well-defined, a standard CNN might be sufficient to achieve accurate classification. However, if the characters have more complex and varied appearances, deeper and more sophisticated CNN architectures such as ResNet, Inception, or VGG may be more suitable.

- **Amount of Available Data**: CNNs generally perform well with large amounts of training data. In your project, you have a substantial dataset of 31,575 images, which should provide ample data for the model to learn effectively. This quantity of data supports the use of a more complex CNN model if needed.

- **Model Performance and Efficiency**: For applications requiring high accuracy, deeper CNN architectures can be beneficial. If computational efficiency and speed are critical, models like MobileNet or EfficientNet, which are designed to be lightweight and fast, may be considered.

### Justification for Choosing a Convolutional Neural Network (CNN)

The decision to use a Convolutional Neural Network (CNN) for character classification in your project is justified by the following factors:

1. **Distinct Patterns**: The characters from One Piece have distinct visual patterns and features that a CNN can learn to recognize effectively.
2. **Sufficient Data**: With 31,575 images in your dataset, there is enough data to train a CNN to achieve good performance.
3. **Proven Effectiveness**: CNNs have a proven track record in image classification tasks, making them a reliable choice for this project.

Given these considerations, a CNN architecture with several convolutional and pooling layers, followed by dense layers, is expected to perform well in classifying One Piece characters accurately and efficiently.

## How to Download the Dataset
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier?select=Data). Please note that the images in this dataset are not owned by the creator of this project.

## How to use the program
1. Download the dataset
2. Extract all the files
3. Copy and Paste the folder called /Data in to the folder of the repository Artificial Intelligence
4. Uncomment the function setup_and_split_dataset from the code
5. Run the code