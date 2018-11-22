import torch
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageOps
import random

def random_crop(inp, out):
	x = random.randint(0,150)
	y = random.randint(0,150)
	inp = inp.crop((x,y,x+100,y+100)).resize((256,256))
	out = out.crop((x,y,x+100,y+100)).resize((256,256))
	return inp, out

class MyDataset(Dataset):
	def __init__(self, root_dir, invert_img, num_random, train):
		self.root_dir = root_dir
		self.size = len(os.listdir(root_dir+"input/"))
		if  train == 1:
			self.size = self.size * (1+invert_img+num_random)
		self.invert_img = invert_img
		self.num_random = num_random
		self.per_img = 1 + invert_img + num_random
		self.train = train

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		idx_ = int(idx/self.per_img)
		inp = Image.open(self.root_dir+"input/"+str(idx_+1)+".jpg")
		out = Image.open(self.root_dir+"target/"+str(idx_+1)+".jpg")

		if self.train == 1:
			if idx % self.per_img == 1:
				if self.invert_img == 1:
					inp = ImageOps.flip(inp)
					out = ImageOps.flip(out)
				else:
					inp, out = random_crop(inp, out)
			elif idx % self.per_img > 1:
				inp, out = random_crop(inp, out)

		trans = torchvision.transforms.ToTensor()
		sample = (2.0*(trans(inp)-0.5),2.0*(trans(out)-0.5))
		return sample


def getDataset(dataset_name, task, invert_img, num_random):
	train = MyDataset("datasets/"+dataset_name+"/train_"+task+"/", invert_img, num_random,1)
	test = MyDataset("datasets/"+dataset_name+"/test_"+task+"/", invert_img, num_random,0)
	eval_ = MyDataset("datasets/"+dataset_name+"/eval_"+task+"/", invert_img, num_random,0)

	return {'train': train, 'test': test, 'eval': eval_} 
