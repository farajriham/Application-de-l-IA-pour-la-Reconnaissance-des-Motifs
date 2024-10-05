from model import config
import torch
from torch.utils.data import Dataset
import cv2
import os


class ImageDataset(Dataset):
    # initialize the constructor
    def __init__(self, data, transforms=None):
        self.transforms = transforms
        self.data = data

    def __getitem__(self, index):
        # retrieve annotations from stored list
        filename, x_min, y_min, x_max, y_max, label = self.data[index]

        # get full path of filename
        image_path = os.path.join(config.IMAGES_PATH, label, filename)

        # load the image (in OpenCV format), and grab its dimensions

        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        # scale bounding box coordinates relative to dimensions of input image
        x_min = float(x_min)
        y_min = float(y_min)
        x_max = float(x_max)
        y_max = float(y_max)
        
        x_min /= w
        y_min /= h
        x_max /= w
        y_max /= h

        # normalize label in (0, 1, 2) and convert to tensor
        label = torch.tensor(config.LABELS.index(label))

        # apply image transformations if any
        if self.transforms:
            image = self.transforms(image)

        # return a tuple of the images, labels, and bounding box coordinates
        return image, label, torch.tensor([x_min, y_min, x_max, y_max])

    def __len__(self):
        # return the size of the dataset
        return len(self.data)
