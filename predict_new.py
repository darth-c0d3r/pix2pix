import sys
import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
import os

cuda = 1
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
print("Device:", device)


folder = sys.argv[1]
task = sys.argv[2]

generator = torch.load("saved_models/generator_model_"+task+".pt").to(device)
generator.eval()
root_dir_input = "datasets/"+folder+"/eval_"+task+"/input/"
root_dir_output = "datasets/"+folder+"/eval_"+task+"/output/"
files = os.listdir(root_dir_input)

with torch.no_grad():
	for file in files:
		inp = Image.open(root_dir_input + file)
		trans = torchvision.transforms.ToTensor()
		tensor = 2.0*(trans(inp)-0.5).to(device)
		tensor = tensor.view(1,tensor.shape[0],tensor.shape[1],tensor.shape[2])
		output = ((generator(tensor)/2.0)+0.5)*255
		tensor = ((tensor/2.0)+0.5)*255

		tensor = tensor.view(tensor.shape[1],tensor.shape[2],tensor.shape[3])
		output = output.view(output.shape[1],output.shape[2],output.shape[3])

		tensor_image = np.zeros([tensor.shape[1], tensor.shape[2], tensor.shape[0]])
		for i in range(0,tensor.shape[1]):
			for j in range(0,tensor.shape[2]):
				tensor_image[i][j][0] = tensor[0][i][j]
				tensor_image[i][j][1] = tensor[1][i][j]
				tensor_image[i][j][2] = tensor[2][i][j]
		
		# trans1 = torchvision.transforms.ToPILImage()
		# tensor_image = np.array(trans1(tensor.cpu()).convert('RGB'))
		# tensor_image = tensor_image[:,:,::-1].copy()

		output_image = np.zeros([output.shape[1], output.shape[2], output.shape[0]])
		for i in range(0,output.shape[1]):
			for j in range(0,output.shape[2]):
				output_image[i][j][0] = output[0][i][j]
				output_image[i][j][1] = output[1][i][j]
				output_image[i][j][2] = output[2][i][j]

		# output_image = np.array(trans1(output.cpu()).convert('RGB'))
		# output_image = output_image[:,:,::-1].copy()

		cv2.imwrite(root_dir_output+file, tensor_image)
		cv2.imwrite(root_dir_output+file, output_image)
		print("Image %s done" % (file))
