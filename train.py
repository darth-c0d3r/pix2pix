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
import numpy as np
import imageio

folder = sys.argv[1]
task = sys.argv[2]

# hyper-parameters
batch_size = 32
epochs = 200
report_every = 2
conv_gen = [3,16,32,64,128,256,512,512] # start with 3 if input image is RGB
conv_dis = [6,16,32,64,128,256,512] # start with 6 if input image is RGB
size = 256
gen_lambda = 20.0
invert_dataset = 0
num_random_crops = 0

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

	batches_done = 0
	all_frames = [[] for i in range(len(db['eval']))]
	
	for epoch in range(1, epochs+1):

		gen.train()
		dis.train()

		train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)

		# Update (Train)
		for batch_idx, (data, target) in enumerate(train_loader):

			data, target = Variable(data.to(device)), Variable(target.to(device))

			# train discriminator
			dis.zero_grad()

			dis_result = dis(data, target).squeeze()
			# dis_real_loss = cGAN_loss(dis_result, Variable(torch.ones(dis_result.size()).to(device)))
			dis_real_loss = cGAN_loss(dis_result, Variable((torch.rand(dis_result.size())*(1-0.8) + 0.8).to(device)))

			gen_result = gen(data)
			dis_result = dis(data, gen_result)
			# dis_fake_loss = cGAN_loss(dis_result, Variable(torch.zeros(dis_result.size()).to(device)))
			dis_fake_loss = cGAN_loss(dis_result, Variable((torch.rand(dis_result.size())*(1-0.8)).to(device)))
			
			dis_train_loss = (dis_real_loss + dis_fake_loss)/2.0
			dis_train_loss.backward()
			dis_optimizer.step()

			# train generator
			gen.zero_grad()

			gen_result = gen(data)
			dis_result = dis(data, gen_result).squeeze()

			gen_train_cGAN_loss = cGAN_loss(dis_result, Variable(torch.ones(dis_result.size()).to(device)))
			gen_train_L1_loss = gen_lambda * L1_loss(gen_result, target)
			gen_train_loss = gen_train_cGAN_loss + gen_train_L1_loss

			gen_train_loss.backward()
			gen_optimizer.step()

			batches_done += 1
			if batches_done % report_every == 0:
				print('[Train] Epoch: {} \t GcGANLoss: {:.6f} \t GL1Loss: {:.6f} \t GTLoss: {:.6f} \t DRLoss: {:.6f} \t DFLoss: {:.6f} \t DTLoss: {:.6f}'.
					format(epoch, gen_train_cGAN_loss, gen_train_L1_loss, gen_train_loss, dis_real_loss, dis_fake_loss, dis_train_loss))

		gen.eval()
		dis.eval()	

		eval_loader = torch.utils.data.DataLoader(db['eval'],batch_size=batch_size, shuffle=False)

		batch_count = 0
		batch_gen_cGAN_loss = 0
		batch_gen_L1_loss = 0
		batch_gen_eval_loss = 0
		batch_dis_fake_loss = 0
		batch_dis_real_loss = 0
		batch_dis_eval_loss = 0
		
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(eval_loader):

				data, target = Variable(data.to(device)), Variable(target.to(device))

				dis_result = dis(data, target).squeeze()
				dis_real_loss = cGAN_loss(dis_result, Variable(torch.ones(dis_result.size()).to(device)))
				# dis_real_loss = cGAN_loss(dis_result, Variable((torch.rand(dis_result.size())*(1-0.8) + 0.8).to(device)))

				gen_result = gen(data)
				dis_result = dis(data, gen_result)
				dis_fake_loss = cGAN_loss(dis_result, Variable(torch.zeros(dis_result.size()).to(device)))
				# dis_fake_loss = cGAN_loss(dis_result, Variable((torch.rand(dis_result.size())*(1-0.8)).to(device)))
				
				dis_eval_loss = (dis_real_loss + dis_fake_loss)/2.0

				gen_result = gen(data)
				dis_result = dis(data, gen_result).squeeze()

				gen_eval_cGAN_loss = cGAN_loss(dis_result, Variable(torch.ones(dis_result.size()).to(device)))
				gen_eval_L1_loss = gen_lambda * L1_loss(gen_result, target)
				gen_eval_loss = gen_eval_cGAN_loss + gen_eval_L1_loss

				# output_images = gen_result.cpu()

				output_images = ((gen_result.cpu()/2.0)+0.5)

				trans1 = torchvision.transforms.ToPILImage()
				for idx in range(len(output_images)):
					# print(trans1(output_images[idx]).shape)
					all_frames[idx].append(np.asarray(trans1(output_images[idx])))

				batch_count += 1;
				batch_gen_cGAN_loss += gen_eval_cGAN_loss
				batch_gen_L1_loss += gen_eval_L1_loss
				batch_gen_eval_loss += gen_eval_loss
				
				batch_dis_fake_loss += dis_fake_loss
				batch_dis_real_loss += dis_real_loss
				batch_dis_eval_loss += dis_eval_loss
				# if batches_done % report_every == 0:
			print('[Eval] Epoch: {} \t GcGANLoss: {:.6f} \t GL1Loss: {:.6f} \t GTLoss: {:.6f} \t DRLoss: {:.6f} \t DFLoss: {:.6f} \t DTLoss: {:.6f}'.
				format(epoch, batch_gen_cGAN_loss/batch_count, batch_gen_L1_loss/batch_count, batch_gen_eval_loss/batch_count, batch_dis_real_loss/batch_count, batch_dis_fake_loss/batch_count, batch_dis_eval_loss/batch_count))

	for idx in range(len(all_frames)):
		imageio.mimsave('datasets/'+folder+'/eval_'+task+'/output/'+str(idx+1)+'.gif',all_frames[idx],'GIF',duration=0.5)

	torch.save(gen, 'saved_models/generator_model_'+task+'.pt')
	torch.save(dis, 'saved_models/discriminator_model_'+task+'.pt')

def main():
	db = dataset.getDataset(folder, task, invert_dataset, num_random_crops)
	print("Image Shape:",db['train'][0][0].shape)
	print("Training Samples:",len(db['train']))
	print("Evaluation Samples:",len(db['eval']))
	print("Testing Samples:",len(db['test']))
	train(db)


if __name__ == '__main__':
	main()	