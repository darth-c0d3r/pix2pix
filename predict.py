import sys
import torch
import torchvision
import numpy as np

num_iter = 100
total_loss = float(0)

cuda = 1
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
print(device)

generator = torch.load(sys.argv[1]).to(device)

for _ in range(num_iter):
	with torch.no_grad():
		tensor = torch.FloatTensor(1,1,5,5).uniform_(-1,0).to(device)
		output = generator(tensor)
		# print(tensor)
		total_loss += (abs(output+tensor)).sum()/float(np.prod(tensor.size()))

total_loss /= float(num_iter)
print(total_loss.item())