import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision # for data
import generator
import discriminator
import time
import dataset

# hyper-parameters
batch_size = 4
epochs = 200
report_every = 16
conv_gen = [3,32,64] # start with 3 if input image is RGB
conv_dis = [6,32,64] # start with 6 if input image is RGB
size = 256
gen_lambda = 100.0

total_images = 100
image_size = 5

# change gpuid to use GPU
cuda = 0 
gpuid = -1

gen = generator.EncoderDecoderNetwork(conv_gen)
dis = discriminator.DiscriminatorNetwork(conv_dis)

cGAN_loss = nn.BCELoss()
L1_loss = nn.L1Loss()

gen_optimizer = optim.Adagrad(gen.parameters(), lr=0.01)
dis_optimizer = optim.Adagrad(dis.parameters(), lr=0.01)

def train(db):

	gen.train()
	dis.train()

	time_start = time.time()

	for epoch in range(1, epochs+1):

		train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)

		# Update (Train)
		for batch_idx, (data, target) in enumerate(train_loader):

			data, target = Variable(data), Variable(target)

			# train discriminator
			dis.zero_grad()

			dis_result = dis(data, target).squeeze()
			dis_real_loss = cGAN_loss(dis_result, Variable(torch.ones(dis_result.size())))

			gen_result = gen(data)
			dis_result = dis(data, gen_result)
			dis_fake_loss = cGAN_loss(dis_result, Variable(torch.zeros(dis_result.size())))

			dis_train_loss = (dis_real_loss + dis_fake_loss)/2.0
			dis_train_loss.backward()
			dis_optimizer.step()

			# train generator
			gen.zero_grad()

			gen_result = gen(data)
			dis_result = dis(data, gen_result).squeeze()

			gen_train_loss = cGAN_loss(dis_result, Variable(torch.ones(dis_result.size()))) + gen_lambda * L1_loss(gen_result, target)
			gen_train_loss.backward()
			gen_optimizer.step()

			if batch_idx % report_every == 0:
				print('Train Epoch: {} \t GenLoss: {:.6f} \t DisRealLoss: {:.6f}\
				 \t DisFakeLoss: {:.6f} \t DisTotalLoss: {:.6f}'.
					format(epoch, gen_train_loss, dis_real_loss, dis_fake_loss, dis_train_loss))

	torch.save(gen, 'generator_model.pt')
	torch.save(dis, 'discriminator_model.pt')

def main():
	db = dataset.getDataset(image_size,0.8*total_images,0.2*total_images)
	# print(db['train'][0])
	train(db)


if __name__ == '__main__':
	main()	