#-------------------------------------------------------------------------------------------------------------------
# depthwise_conv_v2 - Balaji Adithya V
# Define custom torch function approx_depth_conv() and implementation of MBM algo done
# Create lookup table for referring while performing MBM
# ------------------------------------------------------------------------------------------------------------------


import torch
import numpy as np
import torchvision as tv
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import unfold
from torch.utils.cpp_extension import load

#---------------------------------------------- HERE ---------------------------------------------------------------
# This snippet of code extracts row-wise data from reference_mbm.csv that contains the data for determining product
# value based on the 2 inputs. 
#
# Lookup_table created where the inputs themslves act as the pointers needed to locate the MBM product
# (eg) i=25, j=64, then lookup_table[i][j] = MBM(i,j). In tis manner the look up time complexity reduces to O(1)


import csv
import pandas as pd
import time

filename = "reference_mbm.csv"
rows = []

with open(filename, 'r') as file:
	rows = list(csv.reader(file))

	for row in rows[1:]:
		row[0], row[1], row[2] = int(row[0]), int(row[1]), int(row[2])	

lookup_table = np.zeros((256,256))
for row in rows[1:]:
	lookup_table[row[0]][row[1]] = row[2]

#-------------------------------------------------------------------------------------------------------------------
# approx_conv class
# Derived from the torch.autograd.Function class
# Performs in_feature * kernel where (*) represents depthwise-convolution operation
# 
# out_channel = in_channel
#
# in_feature dimensions -> [batch_size, in_channel, in_height, in_width]
# kernel dimensions     -> [out_channel, 1, kh, kw]
# 
# Create patches of input image based on kernel size; Kernel size is also reshaped -> done for efficient computation (parallelizing)
# patches dimensions -> [batch_size, in_channel, out_h*out_w, kh*kw]
# kernel dimensions  -> [in_channel, 1*kh*kw]
# result[b][c] = k[c]*patches[b][c] -> dimensions -> [out_channel, out_h, out_w]; where b denotes current batch, c=current channel
#--------------------------------------------------------------------------------------------------------------------

class depthwise_conv(torch.autograd.Function):
	@staticmethod

	def forward(ctx, in_feature, kernel, out_channel, stride, padding, dilation, groups, bias):
		print("I'm in conv fwd")
		print("input = {}".format(in_feature.shape))
		print("kernel= {}".format(kernel.shape))

		batch_size = in_feature.size(0)
		in_channels = in_feature.size(1)
		orig_h, orig_w = in_feature.size(2), in_feature.size(3)

		tmp = torch.abs(in_feature)
		mx = torch.max(tmp)
		mn = torch.min(tmp)
		mean = torch.mean(tmp)

		print(mx ,mn ,mean)

		#Kernel Dimenstions
		kh, kw = kernel.size(2), kernel.size(3)
		#Strides
		dh, dw = stride, stride

		pad = padding

		#output dimensions
		out_h = (orig_h+2*pad-kh)//dh + 1
		out_w = (orig_w+2*pad-kw)//dw + 1

		img = F.pad(input= in_feature, pad= (pad, pad, pad, pad), mode='constant', value= 0)

		patches = img.unfold(2,kh,dh).unfold(3,kw,dw).reshape(batch_size, in_channels, out_h*out_w, kh*kw)
		result = torch.zeros(batch_size, out_channel, out_h*out_w)
		kernel = kernel.resize(out_channel, -1)

		for b in range(batch_size):
			x = patches[b]
			for c in range(groups):
				start = time.time()
				for i in range(x.size(1)):
					r=0
					for j in range(kernel.size(1)):
						t1 = int((kernel[c][j]*1000).round())
						t2 = int((x[c][i][j]).round())

						if(t1>255) : t1 =255
						if(t1<-255): t1 =-255
						if(t2>255) : t2 =255
						if(t2<-255): t2 =-255

						if((t1>0 and t2>0) or (t1<0 and t2<0)):
							t1=abs(t1)
							t2=abs(t2)
							sign=1
						else:
							t1=abs(t1)
							t2=abs(t2)
							sign=-1

						r += lookup_table[t1][t2]*sign

					result[b][c][i] = r/1000

				end = time.time()
				print("time taken for channel {} in batch {} is {}s".format(c, b, end-start))

		result.reshape(batch_size, out_channel, out_h, out_w)

		if bias is not None:
			result+= bias[None,:,None,None].expand_as(result)

		return result


#-------------------------------------------------------------------------------------------------
# approx_dconv class 
# Derived from nn.Module class
# Used to register the parameters of the conv layer - weight, bias
# Call the depthwise_conv constructor to start depthwise convolution operation
#-------------------------------------------------------------------------------------------------


class approx_dconv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False):
		super(approx_dconv, self).__init__()

		#Initialize misc. variables
		self.in_channels = in_channels  
		self.out_channels = out_channels
		self.stride = stride
		self.dilation = dilation
		self.groups = groups
		self.padding = padding
		self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
		self.bias = self.register_parameter('bias',None)
		print("I'm here")

	def forward(self, x):
		print("I'm in fwd of approx_dconv")
		return depthwise_conv.apply(x, self.weight, self.out_channels, self.stride, self.padding, self.dilation, self.groups, self.bias)
		