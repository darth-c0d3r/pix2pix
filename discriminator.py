import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class DiscriminatorNetwork(nn.Module):

	def __init__(self, conv):
		super(DiscriminatorNetwork, self).__init__()

		# model parameters
		self.kernel_size = 3
		self.padding = 1
		self.dropout = 0.5
		self.leaky_relu_slope = 0.2

		self.batchnorm_layers = nn.ModuleList()
		self.conv_layers = nn.ModuleList()

		for i in range(len(conv)-1):
			self.conv_layers.append(nn.Conv2d(conv[i], conv[i+1], kernel_size=self.kernel_size, padding=self.padding))
			self.batchnorm_layers.append(nn.BatchNorm2d(conv[i+1]))

		self.output_layer = nn.Conv2d(conv[-1], 1, kernel_size=self.kernel_size, padding=self.padding)

	def forward(self, x, y):
		x = torch.cat([x,y],1)

		batchnorm_index = 0

		# convolutional layers
		for conv_layer in self.conv_layers:
			x = conv_layer(x)
			x = self.batchnorm_layers[batchnorm_index](x)
			# x = F.dropout(x,0.5) # add drop-out here if needed
			x = F.leaky_relu(x, self.leaky_relu_slope)
			batchnorm_index += 1

		x = self.output_layer(x)
		x = torch.sigmoid(x)

		return x