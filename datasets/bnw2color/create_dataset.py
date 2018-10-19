import sys
import os
import cv2
import numpy as np
import random

# Assumes that full_dataset is always in a folder named 'full_dataset'

folder = sys.argv[1]
files = os.listdir(folder+"/full_dataset")

files = [file for file in files if file.endswith(".jpg")]
total = len(files)

# Fraction of dataset to be trained and tested on can be changed here 
train_frac = int(0.9*total)
test_frac = int(0.05*total)
eval_frac = total-train_frac-test_frac

random.shuffle(files)

index = 1
for file in files:
	img = cv2.imread(folder+"/full_dataset/"+file)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
	if index <= train_frac:
		cv2.imwrite(folder+"/train/input/"+str(index)+".jpg", gray)
		cv2.imwrite(folder+"/train/output/"+str(index)+".jpg", img)
	elif index <= train_frac+test_frac:
		cv2.imwrite(folder+"/test/input/"+str(index-train_frac)+".jpg", gray)
		cv2.imwrite(folder+"/test/output/"+str(index-train_frac)+".jpg", img)	
	else:
		cv2.imwrite(folder+"/eval/input/"+str(index-train_frac-test_frac)+".jpg", gray)
		cv2.imwrite(folder+"/eval/output/"+str(index-train_frac-test_frac)+".jpg", img)
	index += 1