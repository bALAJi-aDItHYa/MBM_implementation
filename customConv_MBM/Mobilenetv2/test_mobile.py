import csv
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import unfold
import tqdm

import time
import numpy as np
from Mobile_model import o_Model

model = o_Model()
model.load_state_dict(torch.load("mobile_model_2.h5"))

# torch.set_printoptions(profile="full")
torch.set_printoptions(profile="default")
x = model.state_dict()
print(x['decoder.conv3.weight'])


# for params in x:
# 	print("name={}, size={}".format(params, x[params].shape))

# in_ = torch.zeros(1,1280,15,20)

# x = nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=1, padding=1, stride=1)

# print(x(in_).shape)
