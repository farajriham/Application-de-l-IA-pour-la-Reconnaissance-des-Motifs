from model import config
import sys
import os
import torch
import cv2
import numpy
from train import IOU


# load our object detector, set it evaluation mode, and label

if len(sys.argv) < 2:
    print("Please enter the path to the model to be evaluated")
    sys.exit(1)

model_path = sys.argv[1]

print(f"**** loading object detector at {model_path}...")
model = torch.load(model_path).to(config.DEVICE)
model.eval()
print(f"**** object detector loaded")

results_labels = dict()

for mode, csv_file in [['train', config.TRAIN_PATH],
                       ['validation', config.VAL_PATH],
                       ['test', config.TEST_PATH],]:
    data = []
    assert(csv_file.endswith('.csv'))

    print(f"Evaluating {mode} set...")
    # loop over CSV file rows (filename, startX, startY, endX, endY, label)
    for row in open(csv_file).read().strip().split("\n"):
        filename, x1,y1,x2,y2, label = row.split(',')
        filename = os.path.join(config.IMAGES_PATH, label, filename)
        data.append((filename, x1, y1, x2, y2, label))

    print(f"Evaluating {len(data)} samples...")

    # Store all results as well as per class results
    results_labels[mode] = dict()
    results_labels[mode]['all'] = []
    results_labels[mode]['all iou'] = [] 

    for label_str in config.LABELS:
        results_labels[mode][label_str] = []
        results_labels[mode][f'iou_{label_str}'] = []

    # loop over the images that we'll be testing using our bounding box
    # regression model
    for filename, gt_start_x, gt_start_y, gt_end_x, gt_end_y, gt_label in data:
        # load the image, copy it, swap its colors channels, resize it, and
        # bring its channel dimension forward
        image = cv2.imread(filename)
        display = image.copy()
        h, w = display.shape[:2]

        # convert image to PyTorch tensor, normalize it, upload it to the
        # current device, and add a batch dimension
        image = config.TRANSFORMS(image).to(config.DEVICE)
        image = image.unsqueeze(0)

        # predict the bounding box of the object along with the class label
        label_predictions = model(image)
        box_prediction = label_predictions[1]
        # determine the class label with the largest predicted probability
        label_predictions = torch.nn.Softmax(dim=-1)(label_predictions[0])
        most_likely_label = label_predictions.argmax(dim=-1).cpu()
        label = config.LABELS[most_likely_label]

        #denormalize bounding box from (0,1)x(0,1) to (0,w)x(0,h)
        box_prediction[0][0] = int( w * box_prediction[0][0] )
        box_prediction[0][1] = int( h * box_prediction[0][1] ) 
        box_prediction[0][2] = int( w * box_prediction[0][2] ) 
        box_prediction[0][3] = int( h * box_prediction[0][3] )

        # Compare to gt data
        results_labels[mode]['all'].append(label == gt_label)
        results_labels[mode][gt_label].append(label == gt_label)

        #compute cumulated bounding box metrics
        truebox = [int(gt_start_x), int(gt_start_y), int(gt_end_x), int(gt_end_y)]
        iou = IOU(box_prediction[0].cpu().detach().numpy(),truebox)
        results_labels[mode]['all iou'].append(iou)
        results_labels[mode][f'iou_{gt_label}'].append(iou)


        if label != gt_label and iou <0.5 :
            print(f"\tFailure at {filename}")


# Compute per dataset accuracy
for mode in ['train', 'validation', 'test']:
    print(f'\n*** {mode} set accuracy')
    print(f"\tMean accuracy for all labels: "
          f"{numpy.mean(numpy.array(results_labels[mode]['all']))}")
    #display bounding box metrics
    print(f"\tMean iou for all labels: "
          f"{numpy.mean(numpy.array(results_labels[mode]['all iou']))}")

    for label_str in config.LABELS:
        print(f'\n\tMean accuracy for label {label_str}: '
              f'{numpy.mean(numpy.array(results_labels[mode][label_str]))}')
        print(f'\t\t {numpy.sum(results_labels[mode][label_str])} over '
              f'{len(results_labels[mode][label_str])} samples')
        #display bounding box metrics  
        print(f'\n\tMean IoU for label {label_str}: ' +
            f'{numpy.mean(numpy.array(results_labels[mode]["iou_" + label_str]))}')

       
