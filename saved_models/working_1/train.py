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
epochs = 100
report_every = 16
conv_gen = [3,16,32,64,128,256,512,512] # start with 3 if input image is RGB
conv_dis = [6,16,32,64,128,256] # start with 6 if input image is RGB
size = 256
gen_lambda = 20.0

# GPU related info
cuda = 1
gpu_id = 0
# device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() and cuda == 1 else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu") # default gpu
print("Device:", device)

# gen = generator.EncoderDecoderNetwork(conv_gen).to(device)
gen = generator.UNetNetwork(conv_gen).to(device)
dis = discriminator.DiscriminatorNetwork(conv_dis).to(device)

gen.normal_init(0,0.02)
dis.normal_init(0,0.02)

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
	batches_done = 0
	best_model = gen_lambda

	for epoch in range(1, epochs+1):

		train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)

		# Update (Train)
		for batch_idx, (data, target) in enumerate(train_loader):

			data, target = Variable(data.to(device)), Variable(target.to(device))

			# train discriminator
			dis.zero_grad()

			dis_result = dis(data, target).squeeze()
			# dis_real_loss = cGAN_loss(dis_result, Variable(torch.ones(dis_result.size()).to(device)))
			dis_real_loss = cGAN_loss(dis_result, Variable((torch.rand(dis_result.size())*(1-0.7) + 0.7).to(device)))

			gen_result = gen(data)
			dis_result = dis(data, gen_result)
			# dis_fake_loss = cGAN_loss(dis_result, Variable(torch.zeros(dis_result.size()).to(device)))
			dis_fake_loss = cGAN_loss(dis_result, Variable((torch.rand(dis_result.size())*0.3).to(device)))
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

			batches_done += 1
			if batches_done % report_every == 0:
				print('Epoch: {} \t GLoss: {:.6f} \t DRLoss: {:.6f} \t DFLoss: {:.6f} \t DTLoss: {:.6f}'.
					format(epoch, gen_train_loss, dis_real_loss, dis_fake_loss, dis_train_loss))

			if epoch > 20 and gen_train_loss < best_model:
			    torch.save(gen, 'saved_models/generator_model_'+task+'_'+str(epoch)+'.pt')
			    torch.save(dis, 'saved_models/discriminator_model_'+task+'_'+str(epoch)+'.pt')
			    best_model = gen_train_loss

	torch.save(gen, 'saved_models/generator_model_'+task+'.pt')
	torch.save(dis, 'saved_models/discriminator_model_'+task+'.pt')

def main():
	db = dataset.getDataset(folder, task)
	train(db)


if __name__ == '__main__':
	main()	