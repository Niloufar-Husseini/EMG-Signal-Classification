# EMG-Signal-Classification


## Description
This project involves the classification of electromyographic (EMG) signals from leg muscles into two groups: healthy individuals and individuals with limitations. The dataset contains EMG signals from four channels of leg muscles. The project includes several methods for signal processing and classification using deep learning techniques.

## Methods
### Signal to Image Conversion
Convert EMG signals into images for better feature extraction using CNNs.
### Convolutional Neural Network
Design and train a 2D CNN with dropout layers for classification.
### Regularization
Implement L1 and L2 regularization to improve model generalization.
### Low Pass Filtering
Apply a low pass filter to the signals and compare the performance with unfiltered signals.
### Autoencoder Network
Design and train an autoencoder network on the dataset.
### Transfer Learning
Use transfer learning to enhance the classification network with a pre-trained self-encoder and compare results with other pre-trained networks like ResNet.

## Project Structure
- `convert_signal_to_image.ipynb`: Script to convert EMG signals to images.
- `train_cnn_dropout.py`: Script to train the convolutional neural network with dropout.
- `train_cnn_L2_regularization.py`: Script to train the convolutional neural network with L2 regularization.
- `apply_filter.py`: Script to apply filters to the signals.
- `autoencoder`: Script to use autoencoder model.
- `transfer_learning_1` and `transfer_learning_2`: Scripts to use transfer learning to improve the accuracy of the network
- `models/`: Directory containing trained models.


