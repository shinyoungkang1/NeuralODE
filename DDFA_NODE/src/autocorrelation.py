import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data_normalize(obs_dim, datafilepath):
    datafilepath = 'C:/Users/shiny/Documents/NeuralODE_HumanGait/Humangaitdata.npy'
    data = np.load(datafilepath)
    traj_tot = np.load(datafilepath).reshape(72, 1500, obs_dim)
    traj_tot = traj_tot[:,150:1350,:]
    data = data[:, 300:1200, :]
    data = data.reshape(72, 900, obs_dim)
    noise_std = 0.2

    orig_trajs = np.zeros((data.shape[0],data.shape[1],data.shape[2]))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            trajs = data[i,:,j]
            trajs_tot = traj_tot[i,:,j]
            orig_trajs[i,:,j] = (trajs - trajs_tot.mean()) / trajs_tot.std()
            
    #samp_trajs += npr.randn(*samp_trajs.shape) * noise_std #add noise

    return orig_trajs

samp_trajs = load_data_normalize(6, 'C:/Users/shiny/Documents/NeuralODE_HumanGait/Humangaitdata.npy')