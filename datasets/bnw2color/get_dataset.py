import torch
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyDataset(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.size = len(os.listdir(root_dir+"input/"))

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		inp = Image.open(self.root_dir+"input/"+str(idx+1)+".jpg")
		out = Image.open(self.root_dir+"output/"+str(idx+1)+".jpg")
		trans = torchvision.transforms.ToTensor()
		sample = (2.0*(trans(inp)-0.5),2.0*(trans(out)-0.5))
		return sample


def getDataset(dataset_name):
	train = MyDataset(root_dir=dataset_name+"/train/")
	test = MyDataset(root_dir=dataset_name+"/test/")
	eval = MyDataset(root_dir=dataset_name+"/eval/")

	return {'train': train, 'test': test, 'eval': eval} 
