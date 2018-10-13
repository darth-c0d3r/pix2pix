import sys
import torch
import torchvision

generator = torch.load(sys.argv[1])
with torch.no_grad():
	tensor = torch.FloatTensor(1,1,5,5).uniform_(-1,0)
	output = generator(tensor)
	print(tensor+output)