import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=8, nhidden=50):
        super(LatentODEfunc, self).__init__()
        #self.tanh = nn.ELU(inplace= True)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return out

class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=8, obs_dim=46, nhidden=50, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        #self.h1o = nn.Linear(obs_dim, latent_dim*4)
        #self.h3o = nn.Linear(latent_dim*4, latent_dim*2)
        #self.lstm = nn.LSTMCell(latent_dim*2, nhidden)
        self.h1o = nn.Linear(obs_dim, 12)
        self.h3o = nn.Linear(12, 6)
        self.lstm = nn.LSTMCell(6, nhidden)
        self.tanh = nn.Tanh()
        self.h2o = nn.Linear(nhidden, latent_dim*2)

    def forward(self, x, h, c):
        xo = self.h1o(x)
        xo = self.tanh(xo)
        xxo = self.h3o(xo)
        hn, cn = self.lstm(xxo, (h,c))
        hn = self.tanh(hn)
        out = self.h2o(hn)
        return out, hn, cn
    

    def initHidden(self):
        return torch.zeros(1, self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=8, obs_dim=46, nhidden=50):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden*2)
        self.fc3 = nn.Linear(nhidden*2, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))

def mseloss(x, mean):
    loss = nn.MSELoss()
    return loss(x, mean)

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

def MSELoss(yhat, y):
    assert type(yhat) == torch.Tensor
    assert type(y) == torch.Tensor
    return torch.mean((yhat - y) ** 2)


def get_args():
    return {'latent_dim': latent_dim,
            'obs_dim': obs_dim,
            'nhidden': nhidden,
            'dec_nhidden' : dec_nhidden,
            'rnn_nhidden': rnn_nhidden,
            'device': device}

def get_state_dicts():
    return {'odefunc_state_dict': func.state_dict(),
            'encoder_state_dict': rec.state_dict(),
            'decoder_state_dict': dec.state_dict()}

def data_get_dict():
    return {
        'samp_trajs_TE': samp_trajs_TE,
        'samp_trajs_val_TE': samp_trajs_val_TE,
        'samp_ts': samp_ts,
    }

def get_losses():
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_losses_k1': val_losses_k1,
        'val_losses_k2': val_losses_k2,
        'val_losses_k3': val_losses_k3,
        'val_losses_k4': val_losses_k4,
        'val_losses_k5': val_losses_k5
    }

def save_model(tau, k, latent_dim, itr):

    save_dict = {
        'model_args': get_args(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'data': data_get_dict(),
        'train_loss': get_losses()
    }
    
    save_dict.update(get_state_dicts())
    
    torch.save(save_dict, 'model/ODE_TakenEmbedding_RLONG_rnn2_lstm{}_tau{}k{}_LSTM_lr0.008_latent{}_LSTMautoencoder_Dataloader_timestep{}_Trial2_epoch{}.pth'.format(rnn_nhidden, tau, k, latent_dim,tot_num, itr))

    
def data_for_plot_graph(gen_index):
    with torch.no_grad():
        # sample from trajectorys' approx. posterior

        ts_pos = np.linspace(0, ts_num*gen_index, num=tot_num*gen_index)
        ts_pos = torch.from_numpy(ts_pos).float().to(device)
    
        h = torch.zeros(1, samp_trajs_TE_test.shape[0], rnn_nhidden).to(device)
        c = torch.zeros(1, samp_trajs_TE_test.shape[0], rnn_nhidden).to(device)
    
        hn = h[0, :, :]
        cn = c[0, :, :]
    
        for t in reversed(range(samp_trajs_TE_test.size(1))):
            obs = samp_trajs_TE_test[:, t, :]
            out, hn, cn = rec.forward(obs, hn, cn)
        qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(func, z0, ts_pos).permute(1, 0, 2) #change time and batch with permute
        pred_x = dec(pred_z)
        
        return pred_x, pred_z
    
def plot_graph(gen_index, times_index, tot_index, dataset_value, deriv_index, pred_x_forgraph, orig_trajs, itr, path):
    with torch.no_grad():
        orig_trajs_forgraph = orig_trajs
        ts_pos_combined = np.linspace(0, ts_num*tot_index, num=tot_num*tot_index) 
        
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(15, 9))
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            ax.scatter(ts_pos_combined[times_index:times_index+tot_num*gen_index], orig_trajs_forgraph[dataset_value,times_index:tot_num*gen_index, i*(k+1)+deriv_index], s = 5)
            ax.plot(ts_pos_combined[times_index:times_index+tot_num*tot_index], pred_x_forgraph[dataset_value, times_index:times_index+tot_num*tot_index, i*(k+1)+deriv_index], 'r')
            ax.set_ylim(-3, 3)

        
        plot_name = 'lstm_datasetnum{}_latent{}_gen{}_deriv{}_epoch{}.png'.format(dataset_value, latent_dim, tot_index, deriv_index, itr)
        save_path = os.path.join(path, plot_name)
        plt.savefig(save_path, dpi=500)
        plt.close()
    