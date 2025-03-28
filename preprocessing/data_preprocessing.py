import os
import cv2
import numpy as np

def load_data():
    X = []
    Y = []
    image_size = 150
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    for label in labels:
        folder_path = os.path.join('data/brain_tumor_data', label)
        for image_name in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, image_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            X.append(img)
            Y.append(label)

    return np.array(X), np.array(Y)

def preprocess_data(X, Y):
    X = X / 255
    Y = [labels.index(label) for label in Y]
    return X, Y
