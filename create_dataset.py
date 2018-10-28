import sys
import os
import cv2
import numpy as np
import random

# Assumes that full_dataset is always in a folder named 'full_dataset'

folder = sys.argv[1]
folder = "datasets/"+folder
task = sys.argv[2]


files = os.listdir(folder+"/full_dataset")

files = [file for file in files if file.endswith(".jpg")]
total = len(files)

# Fraction of dataset to be trained and tested on can be changed here 
train_frac = int(0.9*total)
test_frac = int(0.05*total)
eval_frac = total-train_frac-test_frac

random.shuffle(files)

os.mkdir(folder+"/train_"+task)
os.mkdir(folder+"/test_"+task)
os.mkdir(folder+"/eval_"+task)
os.mkdir(folder+"/train_"+task+"/input/")
os.mkdir(folder+"/train_"+task+"/target/")
os.mkdir(folder+"/test_"+task+"/input/")
os.mkdir(folder+"/test_"+task+"/target/")
os.mkdir(folder+"/test_"+task+"/output/")
os.mkdir(folder+"/eval_"+task+"/input/")
os.mkdir(folder+"/eval_"+task+"/target/")
os.mkdir(folder+"/eval_"+task+"/output/")

index = 1
for file in files:
	img = cv2.imread(folder+"/full_dataset/"+file)

	if task == "bnw2color":
		out = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		out = cv2.cvtColor(out,cv2.COLOR_GRAY2RGB)
	elif task == "deblur":
		out = cv2.blur(img,(5,5))

	if index <= train_frac:
		cv2.imwrite(folder+"/train_"+task+"/input/"+str(index)+".jpg", out)
		cv2.imwrite(folder+"/train_"+task+"/target/"+str(index)+".jpg", img)
	elif index <= train_frac+test_frac:
		cv2.imwrite(folder+"/test_"+task+"/input/"+str(index-train_frac)+".jpg", out)
		cv2.imwrite(folder+"/test_"+task+"/target/"+str(index-train_frac)+".jpg", img)	
	else:
		cv2.imwrite(folder+"/eval_"+task+"/input/"+str(index-train_frac-test_frac)+".jpg", out)
		cv2.imwrite(folder+"/eval_"+task+"/target/"+str(index-train_frac-test_frac)+".jpg", img)
	index += 1