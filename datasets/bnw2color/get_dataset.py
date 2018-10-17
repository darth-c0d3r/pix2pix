import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyDataset(Dataset):
	def __init__(self, root_dir, total):
		self.root_dir = root_dir
		self.size = total

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		inp = Image.open(self.root_dir+"input/"+str(idx)+".jpg")
		out = Image.open(self.root_dir+"output/"+str(idx)+".jpg")
		trans = torchvision.transforms.ToTensor()
		sample = (2.0*(trans(inp)-0.5),2.0*(trans(out)-0.5))
		return sample


def getDataset(dataset_name, total):
	train = MyDataset(root_dir=dataset_name+"/train/", total=total)
	return {'train': train} 
