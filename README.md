# Brain Tumor Classification

This project involves training a deep learning model to classify brain tumor images into four categories: glioma, meningioma, notumor, and pituitary. The model is built using Convolutional Neural Networks (CNN) in Keras/TensorFlow.

## Setup

1. Clone the repository:



2. Install dependencies:
```
pip install -r requirements.txt

```

3. Download the dataset (brain tumor classification dataset) and place it in the `data/brain_tumor_data/` directory.

## Usage

### Training
Run the following command to start training the model:
```
python main.py

```


### Evaluation
Once the model is trained, it will be evaluated on the test set, and performance metrics such as accuracy and loss will be displayed.

## Model Architecture

The model consists of several convolutional layers followed by fully connected layers. The architecture is designed for image classification tasks, specifically brain tumor classification.

## Future Improvements

- Implement more sophisticated data augmentation techniques.
- Explore transfer learning using pre-trained models like VGG16 or ResNet.
