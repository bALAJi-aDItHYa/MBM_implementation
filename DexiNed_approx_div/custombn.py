import torch
import numpy as np
import torchvision as tv
from torch import nn
import torch.nn.functional as F


import csv
import pandas as pd
import time

filename = "ref_div.csv"
rows = []

with open(filename,'r') as file:
	rows = list(csv.reader(file))

	for row in rows[1:]:
		row[0], row[1], row[2] = int(row[0]), int(row[1]), int(row[2])

lookup_div = np.zeros((256,256))
for row in rows[1:]:
	lookup_div[row[0]][row[1]] = row[2]


def batchnorm2d(X, weight, bias, running_var, running_mean, eps=1e-05):
	batchsize = X.size(0)
	shape = torch.zeros((1,X.size(1), 1, 1))

	weight = weight[None,:,None,None].expand_as(shape)
	bias = bias[None,:,None,None].expand_as(shape)
	running_var = running_var[None,:,None,None].expand_as(shape)
	running_mean = running_mean[None,:,None,None].expand_as(shape)
	print(weight.shape)
	print(bias.shape)
	print(running_mean.shape)
	print(running_var.shape)
	print(X.shape)

	numr = X - running_mean
	denr = torch.sqrt(running_var + eps)
	Y = torch.zeros(X.shape).to(torch.device('cuda'))

	# print("Hey")
	tmp = torch.abs(numr)
	mx = torch.max(tmp)
	mn = torch.min(tmp)
	mean = torch.mean(tmp)
	print(mx ,mn ,mean)

	tmp = torch.abs(denr)
	mx = torch.max(tmp)
	mn = torch.min(tmp)
	mean = torch.mean(tmp)
	print(mx ,mn ,mean)

	print("-----------------------------------------------------------------------------")
	print("\n I am running_mean")
	print(running_mean)
	print('\n')
	print("-----------------------------------------------------------------------------")
	print('\n')
	print(running_var)

	for b in range(batchsize):
		print("In batch {}/ {}".format(b, batchsize))
		for c in range(X.size(1)):
			start = time.time()
			# print("In channel {}/ {}".format(c, X.size(1)))
			for i in range(X.size(2)):
				# print("In h = {}/ {}".format(i, X.size(2)))
				for j in range(X.size(3)):
					# print("In w = {}".format(j))

					t1 = int((numr[b][c][i][j]).round())
					t2 = int((denr[0][c][0][0]).round())

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

					Y[b][c][i][j] = lookup_div[t1][t2]
			end = time.time()
			print("Time taken for channel {}/ {} = {}".format(c,X.size(1),end-start))

	return Y

class custom_bnorm2d(nn.Module):
	def __init__(self, num_features, num_dims=4):
		super(custom_bnorm2d,self).__init__()
		shape = (num_features)

		self.weight = nn.Parameter(torch.ones(shape))
		self.bias = nn.Parameter(torch.zeros(shape))

		self.running_mean = nn.Parameter(torch.zeros(shape))
		self.running_var = nn.Parameter(torch.ones(shape))
		self.num_batches_tracked = nn.Parameter(torch.Tensor(torch.Size([])))

	def forward(self, x):
		y = batchnorm2d(x, self.weight, self.bias, self.running_var, self.running_mean)
		return y