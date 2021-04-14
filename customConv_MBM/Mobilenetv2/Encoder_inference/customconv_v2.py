#-------------------------------------------------------------------------------------------------------------------
# customconv_v2 - Balaji Adithya V
# Define custom torch function approx_conv() and implementation of MBM algo done
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

# ------------------------------------------- HERE -----------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------
# approx_conv class
# Derived from the torch.autograd.Function class
# Performs in_feature * kernel where (*) represents convolution operation
# 
# in_feature dimensions -> [batch_size, in_channel, in_height, in_width]
# kernel dimensions     -> [out_channel, in_channel, kh, kw]
# 
# Create patches of input image based on kernel size; Kernel size is also reshaped -> done for efficient computation (parallelizing)
# patches dimensions -> [batch_size, in_channel*kh*kw, out_h*out_w]
# kernel dimensions  -> [out_channel, in_channel*kh*kw]
# result[b] = k*patches[b] -> dimensions -> [out_channel, out_h, out_w]; where b denotes current batch
#--------------------------------------------------------------------------------------------------------------------


class approx_conv(torch.autograd.Function):

	@staticmethod
	#define the forward utility function - does the MBM convolution operation
	#kernel dimensions --> [out_channel, in_channel, kh, kw]

	def forward(ctx, in_feature, kernel, out_channel, padding, bias):
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
		dh, dw = 1, 1

		pad = padding

		#output dimensions
		out_h = (orig_h+2*pad-kh)//dh + 1
		out_w = (orig_w+2*pad-kw)//dw + 1

		img = F.pad(input= in_feature, pad= (pad, pad, pad, pad), mode='constant', value= 0)

		#Image Dimenstions
		h, w = img.size(2), img.size(3)
		# print(h)
		# print(w)
		# print(rows)

		#Creating the patches - over which convolution is done
		patches = img.unfold(2,kh,dh).unfold(3,kw,dw).reshape(batch_size, in_channels*kh*kw, -1)
		result = torch.zeros(batch_size, out_channel, out_h*out_w)
		kernel = kernel.reshape(out_channel, -1)

		print("reshaped img = {}".format(patches.shape))
		print("out result = {}".format(result.shape))
		print("reshaped kernel = {}".format(kernel.shape))
		print(patches)

		#txt = open("approx_log.txt",'a+')

		for b in range(batch_size):
			x = patches[b]
			for i in range(kernel.size(0)):
				start = time.time()
				for j in range(x.size(1)):
					r=0
					for k in range(kernel.size(1)):
					
						t1 = int((kernel[i][k]*1000).round())
						t2 = int((x[k][j]*100).round())
						#print("here i={}, j={}, k={}".format(i, j, k))
						#print("t1={}, t2={}".format(t1,t2))
						
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
							
						r+= lookup_table[t1][t2]*sign
						
					result[b][i][j] = r/100000
					#print("result={}".format(result[b][i][j]))
				end = time.time()
				#txt.write("\n time taken for 300 convolutions in %d / 1104 is %f" %(i, (end-start)))
				print("time_taken for in {} / {} in batch {} = {}".format(i, kernel.size(0), b, end-start))
				
		result = result.reshape(batch_size, out_channel, out_h, out_w)
		#r = result.permute(0,2,3,1)

		if bias is not None:
			#result += bias.unsqueeze(0).expand_as(result)
			result += bias[None,:,None,None].expand_as(result)
		
		#result = r.permute(0,3,1,2)
		#txt.close()
		return result

#-------------------------------------------------------------------------------------------------
# approxconv2d class 
# Derived from nn.Module class
# Used to register the parameters of the conv layer - weight, bias
# Call the approx_conv constructor to start convolution operation
#-------------------------------------------------------------------------------------------------


class approxconv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False): #,bias=None):
		super(approxconv2d, self).__init__()
		
		#Initialize misc. variables
		self.in_channels = in_channels  
		self.out_channels = out_channels
		self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
		self.bias = self.register_parameter('bias',None)
		self.padding = padding

		# if bias:
		# 	self.bias = nn.Parameter(torch.Tensor(out_channels))
		# else:
		# 	self.register_parameter('bias',None)


	def forward(self, x):
		return approx_conv.apply(x, self.weight, self.out_channels, self.padding, self.bias)





#--------------------------------------------------------------------------------------------------------------------------------------

