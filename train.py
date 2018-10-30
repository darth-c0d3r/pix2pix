import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision # for data
import generator
import discriminator
import time
import get_dataset as dataset
import sys

folder = sys.argv[1]
task = sys.argv[2]

# hyper-parameters
batch_size = 4
epochs = 50
report_every = 16
conv_gen = [3,16,32,64,128,256,512] # start with 3 if input image is RGB
conv_dis = [6,16,32,64,128,256] # start with 6 if input image is RGB
size = 256
gen_lambda = 10.0

# GPU related info
cuda = 1
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
print("Device:", device)

# gen = generator.EncoderDecoderNetwork(conv_gen).to(device)
gen = generator.UNetNetwork(conv_gen).to(device)
dis = discriminator.DiscriminatorNetwork(conv_dis).to(device)

cGAN_loss = nn.BCELoss().to(device)
L1_loss = nn.L1Loss().to(device)

gen_optimizer = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
dis_optimizer = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
# gen_optimizer = optim.Adagrad(gen.parameters(), lr=0.001)
# dis_optimizer = optim.Adagrad(dis.parameters(), lr=0.001)

def train(db):

	gen.train()
	dis.train()

	time_start = time.time()

	for epoch in range(1, epochs+1):

		train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)

		# Update (Train)
		for batch_idx, (data, target) in enumerate(train_loader):

			data, target = Variable(data.to(device)), Variable(target.to(device))

			# train discriminator
			dis.zero_grad()

			dis_result = dis(data, target).squeeze()
			dis_real_loss = cGAN_loss(dis_result, Variable(torch.ones(dis_result.size()).to(device)))

			gen_result = gen(data)
			dis_result = dis(data, gen_result)
			dis_fake_loss = cGAN_loss(dis_result, Variable(torch.zeros(dis_result.size()).to(device)))

			dis_train_loss = (dis_real_loss + dis_fake_loss)/2.0
			dis_train_loss.backward()
			dis_optimizer.step()

			# train generator
			gen.zero_grad()

			gen_result = gen(data)
			dis_result = dis(data, gen_result).squeeze()

			gen_train_loss = cGAN_loss(dis_result, Variable(torch.ones(dis_result.size()).to(device))) + gen_lambda * L1_loss(gen_result, target)
			gen_train_loss.backward()
			gen_optimizer.step()

			if batch_idx % report_every == 0:
				print('Train Epoch: {} \t GenLoss: {:.6f} \t DisRealLoss: {:.6f}\
				 \t DisFakeLoss: {:.6f} \t DisTotalLoss: {:.6f}'.
					format(epoch, gen_train_loss, dis_real_loss, dis_fake_loss, dis_train_loss))

	torch.save(gen, 'saved_models/generator_model_'+task+'.pt')
	torch.save(dis, 'saved_models/discriminator_model_'+task+'.pt')

def main():
	db = dataset.getDataset(folder, task)
	train(db)


if __name__ == '__main__':
	main()	