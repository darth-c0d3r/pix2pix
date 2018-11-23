import sys
import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

cuda = 1
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
print("Device:", device)


folder = sys.argv[1]
task = sys.argv[2]

generator = torch.load("saved_models/generator_model_"+task+"_3.pt", map_location='cpu').to(device)
generator.eval()
root_dir_input = "datasets/"+folder+"/test_"+task+"/input/"
root_dir_output = "datasets/"+folder+"/test_"+task+"/output/"
files = os.listdir(root_dir_input)

with torch.no_grad():
	for file in files:
		inp = Image.open(root_dir_input + file)
		trans = torchvision.transforms.ToTensor()
		trans1 = torchvision.transforms.ToPILImage()
		tensor = 2.0*(trans(inp)-0.5).to(device)
		tensor = tensor.view(1,tensor.shape[0],tensor.shape[1],tensor.shape[2])
		output = ((generator(tensor)/2.0)+0.5)
		# tensor = ((tensor/2.0)+0.5)
		output = output.view(output.shape[1],output.shape[2],output.shape[3])
		
		output_image = trans1(output)

		# output_image.show()
		output_image.save(root_dir_output+file)
		print("Image %s done" % (file))