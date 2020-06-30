
import os
import numpy as np

from yolo_keras.utils import *
from yolo_keras.model import *

classes_path = "yolo_keras/coco_classes.txt"
with open(classes_path) as f:
    class_names = f.readlines()
    class_names = [c.strip() for c in class_names] 
num_classes = len(class_names)
# print(class_names)

# Get the anchor box coordinates for the model
anchors_path = "yolo_keras/yolo_anchors.txt"
with open(anchors_path) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
num_anchors = len(anchors)
# print(anchors)

# Set the expected image size for the model
model_image_size = (416, 416)


