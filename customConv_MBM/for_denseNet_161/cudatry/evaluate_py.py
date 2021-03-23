#-----------------------------------------------------------------------------------------------------------------------------------
# evaluate_py.py
# The actions of this file are :
# 1) Extract test data from the nyu_test.zip containing the required .npy files
# 2) Select the number of images to be passed for evaluation
# 3) Call to initialize the Approx model and load it with pretrained weights and biases nyu_e10.h5
# 4) Pass the model, rgb images, ground truth and crops (Eigen et al.) for evaluation
# 5) Report time taken for evaluation
#
#
#nyu_e10.h5 - Trained parameters of accurate model on 2500 images, 10 epochs, bs=4
#-----------------------------------------------------------------------------------------------------------------------------------

import os
import glob
import time
import argparse
import torch
from torch import nn
from utils2 import evaluate #Here
from approx_Model import approx_model


# Load test data

print('Loading test data...', end='')
import numpy as np
from zipfile import ZipFile
def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

data = extract_zip('nyu_test.zip')
from io import BytesIO
rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
depth = np.load(BytesIO(data['eigen_test_depth.npy']))
crop = np.load(BytesIO(data['eigen_test_crop.npy']))
print('Test data loaded.\n')

#---------------------for 1 image = 5 hrs ---------------- 
rgb = rgb[0:1]
depth = depth[0:1]

#----------------- For GPU -----------------
ap_model = approx_model()
ap_model.load_state_dict(torch.load('nyu_5000_e15.h5'))

#------------------ For CPU ------------------
#ap_model = approx_model()#.to(torch.device('cpu'))
#ap_model.load_state_dict(torch.load('nyu_e10.h5', map_location=torch.device('cpu')))

#for t in ap_model.state_dict():
#	print(t)

start = time.time()
print('Testing...')

e = evaluate(ap_model, rgb, depth, crop, batch_size=1)


end = time.time()
print('\nTest time', end-start, 's')