from model.dataset import ImageDataset

from model.network import SimpleDetector as ObjectDetector
from model.network import ResnetObjectDetector
from sklearn.metrics import average_precision_score
from model import config
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as fun
from torch.optim import Adam
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time
import os
import cv2
from PyQt5.QtCore import QLibraryInfo
import numpy as np


def IOU(box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes given as [xmin, ymin, xmax, ymax].
        
        Parameters:
            box1 (list[float]): The [xmin, ymin, xmax, ymax] coordinates of the first box.
            box2 (list[float]): The [xmin, ymin, xmax, ymax] coordinates of the second box.
        
        Returns:
            float: The IoU of the two bounding boxes.
        """
        # Unpack the coordinates
        box1 = [min(box1[0], box1[2]), box1[1], max(box1[0], box1[2]), box1[3]]
        box2 = [min(box2[0], box2[2]), box2[1], max(box2[0], box2[2]), box2[3]]
        
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

        # Calculate intersection coordinates
        intersect_xmin = max(xmin1, xmin2)
        intersect_ymin = max(ymin1, ymin2)
        intersect_xmax = min(xmax1, xmax2)
        intersect_ymax = min(ymax1, ymax2)
        
        # Calculate intersection area
        if (intersect_xmax > intersect_xmin) and (intersect_ymax > intersect_ymin):
            intersection_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
        else:
            intersection_area = 0
        
        # Calculate the area of both bounding boxes
        box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
        box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area != 0 else 0
        
        return iou

if __name__ == '__main__':
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

    # initialize the list of data (images), class labels, target bounding
    # box coordinates, and image paths
    print("**** loading dataset...")
    data = []

    # loop over all CSV files in the annotations directory
    for csv_file in os.listdir(config.ANNOTS_PATH):
        csv_file = os.path.join(config.ANNOTS_PATH, csv_file)
        # loop over CSV file rows (filename, startX, startY, endX, endY, label)
        fichier = open(csv_file).read().strip().split("\n")
        for index, row in enumerate(fichier):
            if index == 0:
                continue
            data.append(row.strip().split(','))

    # randomly partition the data: 80% training, 10% validation, 10% testing
    random.seed(42)
    random.shuffle(data)

    cut_val = int(0.8 * len(data))   # 0.8
    cut_test = int(0.9 * len(data))  # 0.9
    train_data = data[:cut_val]
    val_data = data[cut_val:cut_test]
    test_data = data[cut_test:]

    # create Torch datasets for our training, validation and test data
    train_dataset = ImageDataset(train_data, transforms=config.TRANSFORMS)
    val_dataset = ImageDataset(val_data, transforms=config.TRANSFORMS)
    test_dataset = ImageDataset(test_data, transforms=config.TRANSFORMS)
    print(f"**** {len(train_data)} training, {len(val_data)} validation and "
          f"{len(test_data)} test samples")

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NB_WORKERS,
                              pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NB_WORKERS,
                              pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             num_workers=config.NB_WORKERS,
                             pin_memory=config.PIN_MEMORY)

    # save testing image paths to use for evaluating/testing our object detector
    print("**** saving training, validation and testing split data as CSV...")
    with open(config.TEST_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in test_data]))
    with open(config.VAL_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in val_data]))
    with open(config.TRAIN_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in train_data]))

    # create our custom object detector model and upload to the current device
    print("**** initializing network...")
    # object_detector = ObjectDetector(len(config.LABELS)).to(config.DEVICE)
    object_detector = ObjectDetector(len(config.LABELS)).to(config.DEVICE)

    # initialize the optimizer, compile the model, and show the model summary
    optimizer = Adam(object_detector.parameters(), lr=config.INIT_LR)
    print(object_detector)

    # initialize history variables for future plot
    plots = defaultdict(list)



    # function to compute loss over a batch
    def compute_loss(loader, back_prop=False):
        # initialize the total loss and number of correct predictions
        total_loss, correct , totIOU = 0, 0, 0

        # loop over batches of the training set
        for batch in loader:
            # send the inputs and training annotations to the device
            images, labels, bbox_data = [datum.to(config.DEVICE) for datum in batch]

            # perform a forward pass and calculate the training loss
            predict, bbox_predict = object_detector(images)
            
            #print( "actual real bbox ", bbox_data , " predicted bbox ", bbox_predict)

            # calcul of losses 
            bbox_loss = fun.mse_loss(bbox_predict, bbox_data, reduction="sum") #!!! vous pouvez changer cette métrique si vous voulez !!!
            class_loss = fun.cross_entropy(predict, labels, reduction="sum")
            batch_loss = config.BBOXW * bbox_loss + config.LABELW * class_loss

            if back_prop:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            for i in range (len(bbox_predict)):
                totIOU += IOU(bbox_predict[i],bbox_data[i])

            total_loss += batch_loss
            correct_labels = predict.argmax(1) == labels
            correct += correct_labels.type(torch.float).sum().item()

            
        return total_loss / len(loader.dataset), correct / len(loader.dataset)  , totIOU/len(loader.dataset)

    # loop over epochs
    print("**** training the network...")
    prev_val_acc = None
    prev_val_loss = None
    prev_val_iou = None
    start_time = time.time()
    for e in range(config.NUM_EPOCHS):
        # set model in training mode & backpropagate train loss for all batches
        object_detector.train()

        
        _,_,_ = compute_loss(train_loader, back_prop=True)

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode and compute validation loss
            object_detector.eval()
            train_loss, train_acc , train_iou = compute_loss(train_loader)
            val_loss, val_acc, val_iou = compute_loss(val_loader)
            


        # update our training history
        plots['Training loss'].append(train_loss.cpu())
        plots['Training class accuracy'].append(train_acc)
        plots['Trainig iou'].append(train_iou)

        plots['Validation class accuracy'].append(val_acc)
        plots['Validation loss'].append(val_loss.cpu())
        plots['Validation iou'].append(val_iou)


        # print the model training and validation information
        print(f"**** EPOCH: {e + 1}/{config.NUM_EPOCHS}")
        print(f"Train loss: {train_loss:.8f}, Train accuracy: {train_acc:.8f}, Train iou: {train_iou:.8f}")
        print(f"Val loss: {val_loss:.8f}, Val accuracy: {val_acc:.8f}, Validation iou: {val_iou:.8f}")
       


        if (prev_val_acc is None  ) or (val_acc > prev_val_acc )or (val_acc >= prev_val_acc )or (val_acc == prev_val_acc and val_loss < prev_val_loss)  :
            prev_val_acc = val_acc
            prev_val_loss = val_loss
            prev_val_iou= val_iou
            # serialize the model to disk
            print("**** saving BEST object detector model...")
            # When a network has dropout and / or batchnorm layers
            # one needs to explicitly set the eval mode before saving
            object_detector.eval() # put it en mode evalution 
            torch.save(object_detector, config.BEST_MODEL_PATH) 

    print("**** saving LAST object detector model...")
    object_detector.eval()
    torch.save(object_detector, config.LAST_MODEL_PATH)

    end_time = time.time()
    print(f"**** total time to train the model: {end_time - start_time:.2f}s")

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()

    # Plot training and validation loss
    plt.plot(plots['Training loss'], label='Training Loss')
    plt.plot(plots['Validation loss'], label='Validation Loss')


    # Plot training and validation accuracy
    plt.plot(plots['Training class accuracy'], label='Training Accuracy')
    plt.plot(plots['Validation class accuracy'], label='Validation Accuracy')

  

    # Plot training and validation iou
    
    plt.plot(np.array([tensor.cpu().detach().numpy() for tensor in plots['Trainig iou']]), label='Training IOU')
    plt.plot(np.array([tensor.cpu().detach().numpy() for tensor in plots['Validation iou']]), label='Validation IOU')


    # Add labels and legend
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    # save the training plot
    plt.savefig(config.PLOT_PATH)