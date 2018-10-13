import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SampleDataset(Dataset):
	def __init__(self, img_size, total):
		self.img_size = img_size
		self.total = total

	def __len__(self):
		return int(self.total)

	def __getitem__(self, idx):
		data = torch.FloatTensor(3,self.img_size,self.img_size).uniform_(-1,0)
		sample = (data,-data)
		return sample

def getDataset(img_size,train_total,test_total):
	train = SampleDataset(img_size=img_size,total=train_total)
	test = SampleDataset(img_size=img_size,total=test_total)
	return {'train': train, 'eval': test} 
