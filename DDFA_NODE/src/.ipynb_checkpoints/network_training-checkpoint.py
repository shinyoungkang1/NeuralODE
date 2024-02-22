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
from torch.utils.data import DataLoader
from .latent_neural_ode_model import LatentODEfunc, RecognitionRNN, Decoder, plot_graph, save_model, MSELoss, RunningAverageMeter

def train_network(data_train, data_val, device, samp_ts, val_ts, n_itrs, latent_dim, n_hidden, obs_dim, rnn_hidden, dec_hidden, batch_size, lr=0.008):
    func = LatentODEfunc(latent_dim, n_hidden).to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_hidden, batch_size).to(device)
    dec = Decoder(latent_dim, obs_dim, dec_hidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=lr)
    loss_meter = RunningAverageMeter()

    train_losses = []
    val_losses = []

    torch.cuda.empty_cache()

    train_loader = DataLoader(dataset = data_train, batch_size = batch_size, shuffle = True, drop_last = True)
    val_loader = DataLoader(dataset = data_val, batch_size = batch_size, shuffle = True, drop_last = True)

    for itr in range(n_itrs):
        for data in train_loader:
            optimizer.zero_grad()
            h = rec.initHidden().to(device)
            c = rec.initHidden().to(device)
            hn = h[0, :, :]
            cn = c[0, :, :]
            for t in reversed(range(data.size(1))):
                obs = data[:, t, :]
                out, hn, cn = rec.forward(obs, hn, cn)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean   

            # forward in time and solve ode for reconstructions
            pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
            pred_x = dec(pred_z)

            # compute loss
            loss = MSELoss(pred_x, data)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        with torch.no_grad():
            for data_val in val_loader:
                h = torch.zeros(1, 72, rnn_hidden).to(device)
                c = torch.zeros(1, 72, rnn_hidden).to(device)
                hn = h[0, :, :]
                cn = c[0, :, :]

                for t in reversed(range(data_val.size(1))):
                    obs = data_val[:, t, :]
                    out, hn, cn = rec.forward(obs, hn, cn)
                qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                #forward in time and solve ode for reconstructions
                pred_z = odeint(func, z0, val_ts).permute(1, 0, 2)
                pred_x = dec(pred_z)

                
                val_loss = MSELoss(pred_x, data_val)
                val_losses.append(val_loss.item())

        # if ((itr > 1000) and (itr % 15 == 0)):
        #     pass
            # save_model(tau, k, latent_dim, itr)
        if (itr % 50 == 0):
            print(f'Iter: {itr}, running avg mse: {loss.item()}, val_loss: {val_loss.item()}')
    return func, rec, dec, train_losses, val_losses


# def load_model(path='model/ODE_TakenEmbedding_RLONG_rnn2_lstm256_tau18k5_LSTM_lr0.008_latent12_LSTMautoencoder_Dataloader_timestep500_epoch1410.pth')
    
#     checkpoint = torch.load(path)
#     func = LatentODEfunc(latent_dim, nhidden).to(device)
#     rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_size).to(device)
#     dec = Decoder(latent_dim, obs_dim, dec_nhidden).to(device)
#     rec.load_state_dict(checkpoint['encoder_state_dict'])
#     func.load_state_dict(checkpoint['odefunc_state_dict'])
#     dec.load_state_dict(checkpoint['decoder_state_dict'])