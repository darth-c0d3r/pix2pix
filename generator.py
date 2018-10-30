import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class EncoderDecoderNetwork(nn.Module):

	def __init__(self, conv):
		super(EncoderDecoderNetwork, self).__init__()

		# model parameters
		self.kernel_size = 3
		self.padding = 1
		self.dropout = 0.5
		self.leaky_relu_slope = 0.2

		self.batchnorm_layers = nn.ModuleList()
		self.conv_layers = nn.ModuleList()
		self.deconv_layers = nn.ModuleList()

		for i in range(len(conv)-1):
			self.conv_layers.append(nn.Conv2d(conv[i], conv[i+1], kernel_size=self.kernel_size, padding=self.padding))
			self.batchnorm_layers.append(nn.BatchNorm2d(conv[i+1]))

		conv.reverse()
		
		for i in range(len(conv)-1):
			self.deconv_layers.append(nn.ConvTranspose2d(conv[i], conv[i+1], kernel_size=self.kernel_size, padding=self.padding))
			self.batchnorm_layers.append(nn.BatchNorm2d(conv[i+1]))

		# last batchnorm layer is redundant as it is not used


	def forward(self, x):

		batchnorm_index = 0

		# convolutional layers
		for conv_layer in self.conv_layers:
			x = conv_layer(x)
			x = F.leaky_relu(self.batchnorm_layers[batchnorm_index](x), self.leaky_relu_slope)
			batchnorm_index += 1

		# deconvolutional layers
		for deconv_layer in self.deconv_layers[:-1]:
			x = deconv_layer(x) # add drop-out here if needed
			x = F.relu(self.batchnorm_layers[batchnorm_index](x))
			batchnorm_index += 1

		x = torch.tanh(self.deconv_layers[-1](x))

		return x

class UNetNetwork(nn.Module):

	def __init__(self, conv):
		super(UNetNetwork, self).__init__()

		# model parameters
		self.kernel_size = 3
		self.padding = 1
		self.dropout = 0.5
		self.leaky_relu_slope = 0.2

		self.batchnorm_layers = nn.ModuleList()
		self.conv_layers = nn.ModuleList()
		self.deconv_layers = nn.ModuleList()

		for i in range(len(conv)-1):
			self.conv_layers.append(nn.Conv2d(conv[i], conv[i+1], kernel_size=self.kernel_size, padding=self.padding))
			self.batchnorm_layers.append(nn.BatchNorm2d(conv[i+1]))

		conv.reverse()
		
		for i in range(len(conv)-1):
			self.deconv_layers.append(nn.ConvTranspose2d(2*conv[i], conv[i+1], kernel_size=self.kernel_size, padding=self.padding))
			self.batchnorm_layers.append(nn.BatchNorm2d(conv[i+1]))

		# last batchnorm layer is redundant as it is not used
		# another bit of redundancy is that the output of last encoder layer
		# is duplicated and is then forwarded to the first decoder layer
		# this is just for the sake of elegancy of the code and also it is not
		# expected to affect the performance/accuracy of the network

	def forward(self, x):

		batchnorm_index = 0
		encoder_outputs = list()

		# convolutional layers
		for conv_layer in self.conv_layers:
			x = conv_layer(x)
			x = F.leaky_relu(self.batchnorm_layers[batchnorm_index](x), self.leaky_relu_slope)
			encoder_outputs.append(x)
			batchnorm_index += 1

		# deconvolutional layers
		deconv_index = -1
		for deconv_layer in self.deconv_layers[:-1]:
			x = torch.cat([x, encoder_outputs[deconv_index]], 1)
			x = deconv_layer(x)
			if deconv_index >= -4:
			    x = F.dropout(x,0.5) # add drop-out here if needed
			x = F.leaky_relu(self.batchnorm_layers[batchnorm_index](x), self.leaky_relu_slope)
			batchnorm_index += 1
			deconv_index -= 1

		x = torch.cat([x, encoder_outputs[0]], 1)
		x = torch.tanh(self.deconv_layers[-1](x))

		return x