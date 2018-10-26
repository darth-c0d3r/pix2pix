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
		out = Image.open(self.root_dir+"target/"+str(idx+1)+".jpg")
		trans = torchvision.transforms.ToTensor()
		sample = (2.0*(trans(inp)-0.5),2.0*(trans(out)-0.5))
		return sample


def getDataset(dataset_name, task):
	train = MyDataset(root_dir="datasets/"+dataset_name+"/train_"+task+"/")
	test = MyDataset(root_dir="datasets/"+dataset_name+"/test_"+task+"/")
	eval_ = MyDataset(root_dir="datasets/"+dataset_name+"/eval_"+task+"/")

	return {'train': train, 'test': test, 'eval': eval_} 
