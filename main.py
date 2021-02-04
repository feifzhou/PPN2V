#!/usr/bin/env python
# coding: utf-8

# # Supervised CARE Training
# 
# Here we use CARE training using noisy images as input and averaged noisy images as target.

# In[1]:

from option import args
options=args

import torch
# import matplotlib.pyplot as plt
import numpy as np
import sys
# sys.path.append('../../../')
from unet.model import UNet
from pn2v import utils
# from pn2v import histNoiseModel
from pn2v import training
# from tifffile import imread
# See if we can use a GPU
# device=utils.getDevice()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
nGPU = torch.cuda.device_count()
if options.test_only:
    nGPU = 1

def predict(model, dat):
    careRes=[]
    resultImgs=[]
    inputImgs=[]

    # We iterate over all test images.
    for index in range(dataTest.shape[0]):
        
        im=dataTest[index]
        gt=dataTestGT[0] # The ground truth is the same for all images
        
        # We are using tiling to fit the image into memory
        # If you get an error try a smaller patch size (ps)
        careResult = prediction.tiledPredict(im, net ,ps=256, overlap=48,
                                                device=device, noiseModel=None)
        
        
        inputImgs.append(im)

        rangePSNR=np.max(gt)-np.min(gt)
        carePrior=PSNR(gt, careResult,rangePSNR )
        careRes.append(carePrior)

        print ("image:",index)
        print ("PSNR input", PSNR(gt, im, rangePSNR))
        print ("PSNR CARE", carePrior) # Without info from masked pixel
        print ('-----------------------------------')
        
        
    # # We display the results for the last test image       
    # vmi=np.percentile(gt,0.01)
    # vma=np.percentile(gt,99)

    # plt.figure(figsize=(15, 15))
    # plt.subplot(1, 2, 1)
    # plt.title(label='Input Image')
    # plt.imshow(im, vmax=vma, vmin=vmi, cmap='magma')

    # plt.subplot(1, 2, 2)
    # plt.title(label='CARE result')
    # plt.imshow(careResult, vmax=vma, vmin=vmi, cmap='magma')

    # plt.figure(figsize=(15, 15))
    # plt.subplot(1, 2, 1)
    # plt.title(label='Input Image')
    # plt.imshow(im[100:200,150:250], vmax=vma, vmin=vmi, cmap='magma')

    # plt.subplot(1, 2, 2)
    # plt.title(label='CARE result')
    # plt.imshow(careResult[100:200,150:250], vmax=vma, vmin=vmi, cmap='magma')
    # plt.show()

    print("Avg PSNR CARE:", np.mean(np.array(careRes) ), '+-(2SEM)',2*np.std(np.array(careRes) )/np.sqrt(float(len(careRes)) ) )

# Download data
# import os
# import urllib
# import zipfile

# if not os.path.isdir('../../../data/Mouse skull nuclei'):
#     os.mkdir('../../../data/Mouse skull nuclei')

# zipPath="../../../data/Mouse_skull_nuclei.zip"
# if not os.path.exists(zipPath):  
#     data = urllib.request.urlretrieve(' https://owncloud.mpi-cbg.de/index.php/s/31ZiGfcLmJXZk3X/download', zipPath)
#     with zipfile.ZipFile(zipPath, 'r') as zip_ref:
#         zip_ref.extractall("../../../data/Mouse skull nuclei")


# ### Load Data
# #### Ensure ```filename = example2_digital_offset300.tif``` and specify the ```dataname```  

# In[3]:


# path='../../../data/Mouse skull nuclei/'
# fileName='example2_digital_offset300.tif'
# dataName='mouseskullnuclei' # This will be used to name the care model


# #### Noisy Data (Input to network)

# In[4]:

# The CARE network requires only a single output unit per pixel
import os, glob
os.makedirs(options.dir, exist_ok=True)
nameModel=options.jobid

model_file = os.path.join(options.dir, "last.net")
if options.load_model and os.path.exists(model_file):
    net=torch.load(model_file)
    print('Model loaded from', model_file)
else:
    net = UNet(1, depth=options.unet_depth)

if nGPU > 1:
    print("Using", nGPU, "GPUs")
    pmodel = torch.nn.DataParallel(net)
    model = pmodel.module
else:
    pmodel = net
pmodel.to(device)

if options.test_only:
    predict(net, options.data)
    exit()

# data=imread(path+fileName)
# nameModel=dataName+'_care'
data_list = [np.load(f) for f in glob.glob(options.data)]
if options.remove_edge > 0:
    data_list = [d[:, options.remove_edge:-options.remove_edge,options.remove_edge:-options.remove_edge] for d in data_list]


if options.mode == 'supervised':
# #### Ground truth Data (Target of Network)
    if options.GT_from_average:
        dataGT_list = [np.repeat(np.mean(d, axis=0)[np.newaxis,...,np.newaxis], len(d), axis=0) for d in data_list]
        data_list=[d[...,np.newaxis] for d in data_list]
        dataGT = np.concatenate(dataGT_list)
        data = np.concatenate(data_list)
        print("Shape of Raw Noisy Image is ", data.shape, "; Shape of Target Image is ", dataGT.shape)
        data = np.concatenate((data,dataGT),axis=-1)
        # Add the target image as a second channel to `data`
    else:
        data = np.concatenate(data_list)
        assert data.shape[-1] >=2, ValueError('Supervised learning without GT channel')
        print('Supervised learning, assuming GT channel(s) at end.')
else:
    print('Unsupervised learning.')
    data = np.concatenate(data_list)
# rng = np.random.default_rng()
# rng.shuffle(data)
np.random.shuffle(data)
print("Shape of data is ", data.shape)


# plt.figure(figsize=(20, 10))
# plt.subplot(1,2,1)
# plt.imshow(data[0,:,:256,0])
# plt.title('Left')

# plt.subplot(1,2,2)
# plt.imshow(data[0,:,256:,0])
# plt.title('Right')

# plt.show()

# # We now crop away the left portion of the data since this portion will be used later for testing
# data = data[:, :, 256:, :]


# ### Create the Network and Train it
# This can take a while.

# In[8]:




# Split training and validation data.
my_train_data=data[:-options.nvalid]#.copy()
my_val_data=data[-options.nvalid:]#.copy()

# Start training.
trainHist, valHist = training.trainNetwork(net=pmodel, trainData=my_train_data, valData=my_val_data,
                                           postfix=nameModel, directory=options.dir, noiseModel=None,
                                           device=device,
                                           numOfEpochs= options.epochs,
                                           stepsPerEpoch=options.step_per_epoch, 
                                           virtualBatchSize=options.virtualbatch_size,
                                           batchSize=options.minibatch_size,
                                           patchSize=options.patch_size,
                                           learningRate=options.lr, 
                                           supervised=(options.mode=='supervised'))
