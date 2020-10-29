import os, sys, shutil
import cv2
from PIL import Image
import numpy as np

current_path = os.getcwd()
labels = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting',
          'Normal', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing',
          'Vandalism']

os.makedirs(os.path.join(current_path, "training"), exist_ok=True)
os.makedirs(os.path.join(current_path, "testing"), exist_ok=True)

train_path = os.path.join(current_path, "GT", "UCF_Crimes-Train-Test-Split",
                          "Anomaly_Detection_splits", "Anomaly_Train.txt")
test_path = os.path.join(current_path, "GT", "UCF_Crimes-Train-Test-Split",
                         "Anomaly_Detection_splits", "Anomaly_Test.txt")

with open(train_path, 'r') as f:
    f = f.readlines()
    for line in f:
        file = line.strip().split('/')[-1]
        source = os.path.join(current_path, line.strip())
        target = os.path.join(current_path, "training", file)
        shutil.move(source, target)
        print("Moving training video {} to {}".format(source, target))

with open(test_path, 'r') as f:
    f = f.readlines()
    for line in f:
        file = line.strip().split('/')[-1]
        source = os.path.join(current_path, line.strip())
        target = os.path.join(current_path, "testing", file)
        shutil.move(source, target)
        print("Moving testing video {} to {}".format(source, target))





