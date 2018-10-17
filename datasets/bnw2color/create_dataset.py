import sys
import os
import cv2
import numpy as np

folder = sys.argv[1]
files = os.listdir(folder)

index = 0
for file in files:
	if file.endswith(".jpg"):
		print(index)
		img = cv2.imread(folder+"/"+file)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
		cv2.imwrite(folder+"/train/input/"+str(index)+".jpg", gray)
		cv2.imwrite(folder+"/train/output/"+str(index)+".jpg", img)
		index += 1