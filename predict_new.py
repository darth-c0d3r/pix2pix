import sys
import torch
import torchvision
import numpy as np
from PIL import Image
import cv2

cuda = 1
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
print(device)

generator = torch.load("saved_models/generator_model.pt").to(device)
index = 9
root_dir = "datasets/bnw2color/Opencountry/train/input/"

with torch.no_grad():
	inp = Image.open(root_dir + str(index) + ".jpg")
	trans = torchvision.transforms.ToTensor()
	tensor = 2.0*(trans(inp)-0.5).to(device)
	tensor = tensor.view(1,tensor.shape[0],tensor.shape[1],tensor.shape[2])
	output = ((generator(tensor)/2.0)+0.5)*256
	tensor = ((tensor/2.0)+0.5)*256

	tensor = tensor.view(tensor.shape[1],tensor.shape[2],tensor.shape[3])
	output = output.view(output.shape[1],output.shape[2],output.shape[3])

	tensor_image = np.zeros([tensor.shape[1], tensor.shape[2], tensor.shape[0]])
	for i in range(0,tensor.shape[1]):
		for j in range(0,tensor.shape[2]):
			tensor_image[i][j][0] = tensor[0][i][j]
			tensor_image[i][j][1] = tensor[1][i][j]
			tensor_image[i][j][2] = tensor[2][i][j]

	output_image = np.zeros([output.shape[1], output.shape[2], output.shape[0]])
	for i in range(0,output.shape[1]):
		for j in range(0,output.shape[2]):
			output_image[i][j][0] = output[0][i][j]
			output_image[i][j][1] = output[1][i][j]
			output_image[i][j][2] = output[2][i][j]

	cv2.imwrite("tensor.jpg", tensor_image)
	cv2.imwrite("output.jpg", output_image)