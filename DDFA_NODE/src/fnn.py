import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

nonlinearTseries = importr("nonlinearTseries")

Normalized_Concatenated_Flies_path = 'C:/Users/shiny/Documents/NeuralODE_Fly/Normalized_Concatenated_Flies_Data'
Norm_Concatenated_file_name = os.path.join(Normalized_Concatenated_Flies_path, 'Flies_Concatenated_Norm.npy')
samp_trajs = np.load(Norm_Concatenated_file_name)

good_ind = [18,19, 24, 25, 28,29,30,31, 34, 35, 40, 41]

X = samp_trajs[:,30000:30500,good_ind]

X = X[:,::2,:]

cao_emb_dim = nonlinearTseries.estimateEmbeddingDim(
    data,  # time series
    len(data),  # number of points to use, use entire series
    3,  # time delay
    12,  # max no. of dimension
    0.95,  # threshold value
    0.1,  # max relative change
    True,  # do the plot
    "Computing the embedding dimension",  # main
    "dimension (d)",  # x_label
    "E1(d) & E2(d)",  # y_label
    ro.NULL,  # x_lim
    ro.NULL,  # y_lim
    0  # add a small amount of noise to the original series to avoid the
          # appearance of false neighbours due to discretization errors.
          # This also prevents the method to fail with periodic signals, 0 for no noise
)

embedding_dimension = int(cao_emb_dim[0])