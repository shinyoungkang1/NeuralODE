import numpy as np
import os
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

nonlinearTseries = importr("nonlinearTseries")

def get_autocorr_1_e_time(x, threshold=1/np.e, maxlags=100):
    lags, autocorr, _, _ = plt.acorr(x, maxlags=maxlags)
    plt.close()

    # only look at positive lags
    lags, autocorr = lags[lags>0], autocorr[lags>0]

    # Find the first time autocorrelation goes below the threshold
    first_below_threshold = None
    for lag, ac in zip(lags, autocorr):
        if ac < threshold:
            first_below_threshold = lag
            break
    if first_below_threshold is None:
        print("No 1/e time detected. Increase maxlags parameter!")
        
    return first_below_threshold

def get_best_1_e_time(data):
    trials, timesteps, features = data.shape
    e_times = np.array([[get_autocorr_1_e_time(data[x, :, y]) for x in range(trials)] for y in range(features)])
    return e_times.median()

def get_embedding_dim(x, delay=3, max_dim=12, threshold=0.95, max_rel_change=0.1, plot=True, noise=0.0):
    cao_emb_dim = nonlinearTseries.estimateEmbeddingDim(
        x,  # time series
        len(x),  # number of points to use, use entire series
        delay,  # time delay
        max_dim,  # max no. of dimension
        threshold,  # threshold value
        max_rel_change,  # max relative change
        plot,  # do the plot
        "Computing the embedding dimension",  # main
        "dimension (d)",  # x_label
        "E1(d) & E2(d)",  # y_label
        ro.NULL,  # x_lim
        ro.NULL,  # y_lim
        noise # add a small amount of noise to the original series to avoid the
              # appearance of false neighbours due to discretization errors.
              # This also prevents the method to fail with periodic signals, 0 for no noise
    )
    return int(cao_emb_dim[0])

def takenembedding(data, tau, k):
    data_TE = np.zeros((data.shape[0], data.shape[1]-tau*k, data.shape[2]), dtype = object)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            for t in range(data.shape[1]-tau*k):
                data_TE[i,t,j] = data[i, t:t+tau*k+1, j][::tau][::-1]
                
    data_TE = np.array(data_TE.tolist())
    data_TE = data_TE.reshape(data_TE.shape[0],data_TE.shape[1], data.shape[2]*(k+1))
    
    return data_TE


def embed_data(data, e_time_threshold=1/np.e, maxlags=100, max_dim=12, nn_threshold=0.95, max_rel_change=0.1, plot=True, noise=0.0):
    best_delay = get_best_1_e_time(data, maxlags=maxlags, threshold=e_time_threshold)
    best_dim = get_best_embedding_dim(data, max_dim=max_dim, nn_threshold=0.95, max_rel_change=0.1, plot=True, noise=0.0)
    return takens_embedding(data, best_delay, best_dim)
    
