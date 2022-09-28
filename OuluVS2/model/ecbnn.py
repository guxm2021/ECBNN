import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from model.base import BaseModel, temporal_ce_loss, to_tensor, to_np, vote_video_classification_result
from model.net import DeltaEncoder, LinearEncoder


# ======================================================================================================================
class nnSqueeze(nn.Module):
    def __init__(self):
        super(nnSqueeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


class nnUnsqueeze(nn.Module):
    def __init__(self):
        super(nnUnsqueeze, self).__init__()

    def forward(self, x):
        return x[:, :, None, None]

# define the networks for backbone
class Conv3dEncoder(nn.Module):
    """
    Conv3d module is adopted to process video
    """
    def __init__(self, opt, dropout_p=0.5):
        super(Conv3dEncoder, self).__init__()
        channels = opt.cnn_channels
        self.conv1 = nn.Conv3d(1, channels[0], (3, 3, 3), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(channels[0], channels[1], (3, 3, 3), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(channels[1], channels[2], (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)

    def forward(self, x):
        # x: (batch, channel, seqlen, h, w)
        x = self.conv1(x)           # (44, 50) -> (23, 26)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)           # (23, 26) -> (11, 13)

        x = self.conv2(x)           # (11, 13) -> (13, 14)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)           # (13, 14) -> (6, 7)

        x = self.conv3(x)           # (6, 7) -> (6, 7)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)           # (6, 7) -> (3, 3)
        return x  # (B, C, T, H, W)

class Encoder1(nn.Module):
    """
    Linear + Delta + BLSTM as encoder
    Linear as classifier
    """
    def __init__(self, opt):
        super(Encoder1, self).__init__()
        self.device = opt.device
        
        # model architecture for CNN
        self.cnn_encoder = Conv3dEncoder(opt)

        # model architecture for Linear
        self.cnn_output_size = opt.cnn_output_size
        self.shapes = opt.dnn_shapes
        self.nonlinearities = opt.nonlinearities
        self.linear_encoder = LinearEncoder(self.cnn_output_size, self.shapes, self.nonlinearities, residual=False)
        
        # model architecture for Delta
        self.window = opt.window
        self.delta_encoder = DeltaEncoder(self.device, self.window)
        
        # model architecture for LSTM
        self.nz = opt.nz                       # size for hidden feature (domain adaptation)
        self.lstm_input_size = self.shapes[-1]
        self.nc = opt.nc

        self.blstm_encoder = nn.LSTM(
            input_size=self.lstm_input_size*3,
            hidden_size=self.nz,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def init_hidden(self, batch_size, num_layers=1, directions=1):  # blstm so 2 direction
        # h_0=(num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)

        h_0 = torch.randn(num_layers*directions, batch_size, self.nz)
        c_0 = torch.randn(num_layers*directions, batch_size, self.nz)

        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)

        return h_0, c_0

    def forward(self, x, hidden=None):
        # x: (B, C, T, H, W) cnn_encode
        batch = x.shape[0]
        
        x = self.cnn_encoder(x)

        # initialize the hidden state for LSTM
        if hidden == None:
            h_0, c_0 = self.init_hidden(batch)
        else:
            h_0, c_0 = hidden

        # (B, C, T, H, W)->(T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(1, 0, 2).contiguous()

        # linear encode
        x = self.linear_encoder(x)   # (batch, seq_len, lstm_input_size)

        # delta encode
        x = self.delta_encoder(x)    # (batch, seq_len, lstm_input_size * 3)
        
        # z is the feature for domain adaptation
        z1, (h_n, c_n) = self.blstm_encoder(x, (h_0, c_0))      # (batch, seq_len, nz)

        return z1, (h_n, c_n)


class Encoder2(nn.Module):
    """
    BNN + Linear + Delta + BLSTM as encoder
    Linear as classifier
    """
    def __init__(self, opt):
        super(Encoder2, self).__init__()
        self.device = opt.device
        self.n_particle = opt.n_particle
        self.bnn_dim = opt.bnn_dim
        self.theta_dim = opt.theta_dim

        # model architecture for CNN
        self.cnn_encoder = Conv3dEncoder(opt)

        # model architecture for Linear
        self.cnn_output_size = opt.cnn_output_size
        self.shapes = opt.dnn_shapes
        self.nonlinearities = opt.nonlinearities
        self.linear_encoder = LinearEncoder(self.cnn_output_size, self.shapes, self.nonlinearities, residual=False)
        
        # model architecture for Delta
        self.window = opt.window
        self.delta_encoder = DeltaEncoder(self.device, self.window)
        
        # model architecture for LSTM
        self.nz = opt.nz                       # size for hidden feature (domain adaptation)
        self.lstm_input_size = self.shapes[-1]
        self.nc = opt.nc

        self.blstm_encoder = nn.LSTM(
            input_size=self.lstm_input_size*3,
            hidden_size=self.nz,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def init_hidden(self, batch_size, num_layers=1, directions=1):  # blstm so 2 direction
        # h_0=(num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)

        h_0 = torch.randn(num_layers*directions, batch_size*self.n_particle, self.nz)
        c_0 = torch.randn(num_layers*directions, batch_size*self.n_particle, self.nz)

        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)

        return h_0, c_0

    def forward(self, x, particles, hidden=None):
        # x: (B, C, T, H, W) cnn encode
        # particles: (B, T, n_particle, theta_dim)
        x = self.cnn_encoder(x)
        x_feat = x

        batch, channel, frame = x.shape[:3]
        
        # initialize the hidden state for LSTM
        if hidden == None:
            h_0, c_0 = self.init_hidden(batch)
        else:
            h_0, c_0 = hidden
        
        # (B, C, T, H, W)->(B, T, C, H, W)
        x = x.transpose(1, 2).contiguous()
        # (B, T, C, H, W)->(B, T, C, H*W)
        x = x.view(batch, frame, channel, -1)

        # bnn
        bnn = particles.view(batch, frame, self.n_particle, self.bnn_dim, self.bnn_dim)
        x = F.relu(torch.einsum('bfnij, bfcj -> bfnci', bnn, x))      # (batch, frame, n_particle, 96, 9)

        x = x.reshape(batch, frame, self.n_particle, self.cnn_output_size)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch*self.n_particle, frame, self.cnn_output_size) # (batch*n_particle, frame, 864)

        # linear encode
        x = self.linear_encoder(x)   # (batch*n_particle, frame, lstm_input_size)

        # delta encode
        x = self.delta_encoder(x)    # (batch*n_particle, frame, lstm_input_size * 3)
        
        # z2 is the feature for domain adaptation
        z2, (h_n, c_n) = self.blstm_encoder(x, (h_0, c_0))      # (batch*n_particle, frame, nz)

        # reshape
        z2 = z2.view(batch, self.n_particle, frame, self.nz)
        z2 = z2.transpose(1, 2).contiguous()                   # (batch, frame, n_particle, nz)

        return z2, (h_n, c_n), x_feat
    
    def forward_inf(self, x, particles, hidden=None):
        # x: (B, C, T, H, W) cnn encode
        # particles: (B, T, n_particle, theta_dim)

        batch, channel, frame = x.shape[:3]
        
        # initialize the hidden state for LSTM
        if hidden == None:
            h_0, c_0 = self.init_hidden(batch)
        else:
            h_0, c_0 = hidden
        
        # (B, C, T, H, W)->(B, T, C, H, W)
        x = x.transpose(1, 2).contiguous()
        # (B, T, C, H, W)->(B, T, C, H*W)
        x = x.view(batch, frame, channel, -1)

        # bnn
        bnn = particles.view(batch, frame, self.n_particle, self.bnn_dim, self.bnn_dim)
        x = F.relu(torch.einsum('bfnij, bfcj -> bfnci', bnn, x))      # (batch, frame, n_particle, 96, 9)

        x = x.reshape(batch, frame, self.n_particle, self.cnn_output_size)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch*self.n_particle, frame, self.cnn_output_size) # (batch*n_particle, frame, 864)

        # linear encode
        x = self.linear_encoder(x)   # (batch*n_particle, frame, lstm_input_size)

        # delta encode
        x = self.delta_encoder(x)    # (batch*n_particle, frame, lstm_input_size * 3)
        
        # z2 is the feature for domain adaptation
        z2, (h_n, c_n) = self.blstm_encoder(x, (h_0, c_0))      # (batch*n_particle, frame, nz)

        # reshape
        z2 = z2.view(batch, self.n_particle, frame, self.nz)
        z2 = z2.transpose(1, 2).contiguous()                   # (batch, frame, n_particle, nz)

        return z2, (h_n, c_n)
    


# define networks for discriminator
class DiscConv(nn.Module):
    def __init__(self, nin, nout):
        super(DiscConv, self).__init__()
        nh = 512
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnSqueeze(),
            nn.Linear(nh, nout),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 3:
            batch, frame, nz = x.shape
            x = x.reshape(batch*frame, nz, 1, 1)
            x = self.net(x)
            x = x.reshape(batch, frame, -1)
        else:
            batch, frame, n_particle, nz = x.shape
            x = x.reshape(batch*frame*n_particle, nz, 1, 1)
            x = self.net(x)
            x = x.reshape(batch, frame, n_particle, -1)
        return x


class Decoder(nn.Module):
    """
    Decoder is adopted to reconstruct x from particles and computes the importance weights
    """
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.theta_dim = opt.theta_dim
        self.pic_size = opt.pic_size
        self.main = nn.Sequential(
            nn.Linear(self.theta_dim, self.pic_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.pic_size, self.pic_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.pic_size, self.pic_size),
            nn.Tanh()
        )

    def forward(self, x):
        """
        :param x (particles): (batch, frame, n_particle, theta_dim)
        :return: x_recons
        """
        batch, frame, n_particle, channel = x.shape
        x = x.view(batch*frame*n_particle, channel)
        x = self.main(x)
        x = x.view(batch, frame, n_particle, self.pic_size)
        return x

# define network for posterior and prior sampler
class PriorSampler(nn.Module):
    """
    PriorSampler is adopted to compute the prior for x and compute kL divergence
    """
    def __init__(self, opt):
        super(PriorSampler, self).__init__()
        self.nz = opt.nz
        self.device = opt.device
        self.theta_dim = opt.theta_dim
        self.n_particle = opt.n_particle

        self.bottom = nn.Linear(self.nz + self.theta_dim * 2, self.theta_dim)
        self.rnn = nn.GRU(self.theta_dim, self.theta_dim, num_layers=2, batch_first=True)
        self.head1 = nn.Linear(self.theta_dim, self.theta_dim)
        self.head2 = nn.Linear(self.theta_dim, self.theta_dim)
        self.positive_act = nn.Softplus()

    def forward(self, initial_value, z1, hidden=None):  
        # input to this network is z1: (batch, frame , nz)
        # input initial_value: tuple -> mean (batch, theta_dim)  variance > 0 (batch, theta_dim)
        initial_mean, initial_variance = initial_value
        assert initial_mean.dim() == 2 and initial_variance.dim() == 2
        batch, frame = z1.shape[:2]
        
        # concatenate
        initial_mean = initial_mean.unsqueeze(dim=-2).expand(batch, frame, self.theta_dim)
        initial_variance = initial_variance.unsqueeze(dim=-2).expand(batch, frame, self.theta_dim)
        x = torch.cat([z1, initial_mean, initial_variance], dim=-1)

        # first layer
        x = F.relu(self.bottom(x))

        # rnn
        # x = x.transpose(0, 1).contiguous()

        x, hidden_state = self.rnn(x, hidden)
        # x = x.transpose(0, 1).contiguous()                  # (batch, frame, nz)
        
        # proposal
        proposal_mean = self.head1(x)                                 # mean for x prior
        proposal_variance = self.positive_act(self.head2(x))          # variance (delta ** 2, not delta) for x prior

        # sampling
        particles = torch.randn(batch, frame, self.n_particle, self.theta_dim).to(self.device) * \
            torch.sqrt(proposal_variance.unsqueeze(dim=-2)) + proposal_mean.unsqueeze(dim=-2)
        return particles, proposal_mean, proposal_variance, hidden_state


class InitialValueModule(nn.Module):
    """
    IntialValueModule is adopted to generate the initial value for the next batch
    Only use the observations from last batch
    """
    def __init__(self, opt):
        super(InitialValueModule, self).__init__()
        self.nz = opt.nz
        self.theta_dim = opt.theta_dim
        self.rnn = nn.GRU(self.theta_dim, self.theta_dim, num_layers=2, batch_first=True)
        self.bottom = nn.Linear(self.nz, self.theta_dim)
        self.head1 = nn.Linear(self.theta_dim, self.theta_dim)
        self.head2 = nn.Linear(self.theta_dim, self.theta_dim)
        self.positive_act = nn.Softplus()
    
    def forward(self, x, hidden=None):
        # input to this network is z1 (dimension=nh), which is the output of encoder
        # x: (batch, frame, nz)

        # forward mlp to obtain the mean and variance
        x = F.relu(self.bottom(x))                        # (batch, frame, theta_dim)  

        # forward RNN
        # x = x.transpose(0, 1).contiguous()

        x, hidden_state = self.rnn(x, hidden)
        # x = F.relu(x)                                   # whether to use relu() is determined by experimental results
        # x = x.transpose(0, 1).contiguous()              # (batch, frame, theta_dim)
        
        x = x[:, -1, :]
        # generate mean and variance
        mean = self.head1(x)
        variance = self.positive_act(self.head2(x))

        initial_value = (mean, variance)
        return initial_value, hidden_state              # tuple: ((batch, theta_dim), (batch, theta_dim))


class ResLinear(nn.Module):
    """
    Residual network for linear
    """
    def __init__(self, ndim, num_layers):
        super(ResLinear, self).__init__()
        self.layers_list = nn.ModuleList([])
        self.num_layers = num_layers
        for _ in range(self.num_layers):
            self.layers_list.append(nn.Sequential(nn.Linear(ndim, ndim), nn.ReLU(True), nn.Linear(ndim, ndim)))
    
    def forward(self, x):
        for layer in range(self.num_layers):
            res = x
            x = self.layers_list[layer](x)
            x = F.relu(x + res)
        return x


class MeanSampler(nn.Module):
    def __init__(self, ndim, num_layers):
        super(MeanSampler, self).__init__()
        self.bottom1 = nn.Linear(1, ndim)
        self.bottom2 = nn.Linear(ndim, ndim)
        self.join = nn.Linear(ndim * 2, ndim)
        self.resblock = ResLinear(ndim, num_layers) # nn.Sequential(nn.Linear(ndim, ndim), nn.ReLU(True), nn.Linear(ndim, ndim), nn.ReLU(True))# ResLinear(ndim, num_layers)
        self.head = nn.Linear(ndim, ndim)
    
    def forward(self, x, t):
        # x:(batch, frame, ndim)  t: (batch, frame, 1)
        t = F.relu(self.bottom1(t))
        x = F.relu(self.bottom2(x))
        x = torch.cat([t, x], dim=-1)
        x = F.relu(self.join(x))
        x = self.resblock(x)
        mean = self.head(x)
        return mean


class VarianceSampler(nn.Module):
    def __init__(self, ndim, num_layers):
        super(VarianceSampler, self).__init__()
        self.bottom1 = nn.Linear(1, ndim)
        self.bottom2 = nn.Linear(ndim, ndim)
        self.join = nn.Linear(ndim * 2, ndim)
        self.resblock = ResLinear(ndim, num_layers) # nn.Sequential(nn.Linear(ndim, ndim), nn.ReLU(True), nn.Linear(ndim, ndim), nn.ReLU(True))# ResLinear(ndim, num_layers)
        self.head = nn.Sequential(nn.Linear(ndim, ndim), nn.Softplus())
    
    def forward(self, x, t):
        # x:(batch, frame, ndim)  t: (batch, frame, 1)
        t = F.relu(self.bottom1(t))
        x = F.relu(self.bottom2(x))
        x = torch.cat([t, x], dim=-1)
        x = F.relu(self.join(x))
        x = self.resblock(x)
        variance = self.head(x)
        return variance


# WeightSampler serves as the NN3 in the model
class WeightSampler(nn.Module):
    def __init__(self, opt):
        """
        WeightSampler is implemented to sample importance weights
        """
        super(WeightSampler, self).__init__()
        self.device = opt.device
        self.theta_dim = opt.theta_dim
        self.bottom1 = nn.Linear(1, self.theta_dim)
        self.bottom2 = nn.Linear(self.theta_dim, self.theta_dim)
        self.bottom3 = nn.Linear(self.theta_dim, self.theta_dim)
        self.bottom4 = nn.Linear(self.theta_dim, self.theta_dim)

        self.main = nn.Sequential(
            nn.Linear(self.theta_dim * 4,  1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Softplus()
        )

    
    def forward(self, proposal_mean, proposal_variance, particles, t_seq):
        # proposal_mean / proposal_variance: (batch, frame, theta_dim)
        # particles: (batch, frame, n_particle, theta_dim)
        # t_seq: (batch, frame, n_particle, 1) 
        batch, frame, n_particle = particles.shape[:3]
        proposal_mean = proposal_mean.unsqueeze(dim=-2).expand(batch, frame, n_particle, self.theta_dim)
        proposal_variance = proposal_variance.unsqueeze(dim=-2).expand(batch, frame, n_particle, self.theta_dim)
        # t_seq = t_seq.unsqueeze(dim=-2).expand(batch, frame, n_particle, 1)

        t_seq = F.relu(self.bottom1(t_seq))
        particles = F.relu(self.bottom2(particles))
        proposal_mean = F.relu(self.bottom3(proposal_mean))
        proposal_variance = F.relu(self.bottom4(proposal_variance))
        
        x = torch.cat([t_seq, particles, proposal_mean, proposal_variance], dim=-1)
        x = x.view(batch*frame*n_particle, -1)
        weights = self.main(x)
        weights = weights.view(batch, frame, n_particle)
        # weights = weights.squeeze(dim=-1)
        log_weights = torch.log(weights + 1e-10)
        # log_weights = log_weights.squeeze(dim=-1)
        # normalize weights
        weights = F.softmax(weights, dim=-1)  # weights / (weights.sum(dim=-1, keepdim=True) + 1e-5) 
        return weights, log_weights     # (batch, frame, n_particles)


class PosteriorSampler(nn.Module):
    """
    PosteriorSampler is adopted to compute the posterior for x and compute kL divergence
    """
    def __init__(self, opt):
        super(PosteriorSampler, self).__init__()
        # define the network parameters
        self.device = opt.device
        self.theta_dim = opt.theta_dim

        self.n_particle = opt.n_particle               # number of particles
        # define the network used to sampling theta

        # network 1 for mean
        self.network1 = MeanSampler(self.theta_dim, num_layers=3)
        # network 2 for variance
        self.network2 = VarianceSampler(self.theta_dim, num_layers=3)
        # network 3 for weights
        self.network3 = WeightSampler(opt)

    def forward(self, initial_value, t_seq, is_diff=False):
        """
        :param: initial_value: tuple -> mean (batch, theta_dim)  variance > 0 (batch, theta_dim)
        :param: t_seq: (batch, frame, 1) 
        :output: particles: (batch, frame, n_particle, theta_dim)
        :output: output proposal_mean: (batch, frame, theta_dim)
        :output: output proposal_variance: (batch, frame, theta_dim)
        """
        initial_mean, initial_variance = initial_value
        assert initial_mean.dim() == 2 and initial_variance.dim() == 2
        batch, frame = t_seq.shape[:2]
        t_seq.requires_grad_(True)

        # determine proposal mean and proposal variance 
        initial_mean = initial_mean.unsqueeze(dim=-2).expand(batch, frame, self.theta_dim)           # (batch, frame, theta_dim)
        initial_variance = initial_variance.unsqueeze(dim=-2).expand(batch, frame, self.theta_dim)   # (batch, frame, theta_dim)

        proposal_mean = self.network1(initial_mean, t_seq)
        proposal_variance = self.network2(initial_variance, t_seq)

        # sample the particles from proposal mean and proposal std
        particles = torch.randn(batch, frame, self.n_particle, self.theta_dim).to(self.device) * \
            torch.sqrt(proposal_variance.unsqueeze(dim=-2)) + proposal_mean.unsqueeze(dim=-2)
 
        # sample weights
        t_seq_expand = t_seq.unsqueeze(dim=-2).expand(batch, frame, self.n_particle, 1)
        weights, log_weights = self.network3(proposal_mean, proposal_variance, particles, t_seq_expand)

        # compute gradient
        if is_diff:
            # gradient matching
            dlogw_dt = torch.autograd.grad(log_weights, t_seq_expand, torch.ones_like(log_weights), create_graph=True)[0]
            d2logw_dt2 = torch.autograd.grad(dlogw_dt, t_seq_expand, torch.ones_like(dlogw_dt), create_graph=True)[0]
            # gradient: (batch, frame, n_particle, 1)

            dmean_dt = torch.autograd.grad(proposal_mean, t_seq, torch.ones_like(proposal_mean), create_graph=True)[0]
            d2mean_dt2 = torch.autograd.grad(dmean_dt, t_seq, torch.ones_like(dmean_dt), create_graph=True)[0]

            dvar_dt = torch.autograd.grad(proposal_variance, t_seq, torch.ones_like(proposal_variance), create_graph=True)[0]
            d2var_dt2 = torch.autograd.grad(dvar_dt, t_seq, torch.ones_like(dvar_dt), create_graph=True)[0]
            # gradient: (batch, frame, 1)
            
            # pack
            deriv = (d2logw_dt2, dmean_dt, d2mean_dt2, dvar_dt, d2var_dt2)
        else:
            deriv = None

        return particles, weights, proposal_mean, proposal_variance, deriv    


class ECBNN(BaseModel):
    def __init__(self, opt):
        super(ECBNN, self).__init__(opt)
        # parameters for particle filter
        self.n_particle = opt.n_particle
        self.inf_step = opt.inf_step
        self.inf_scale = opt.inf_scale

        # parameters for network architecture
        self.nc = opt.nc
        self.nz = opt.nz
        self.n_frame = opt.n_frame
        
        # lambda for loss terms
        self.lambda_D_dis_frame = opt.lambda_D_dis_frame
        self.lambda_G_dis_frame = opt.lambda_G_dis_frame
        self.lambda_D_dis_inf = opt.lambda_D_dis_inf
        self.lambda_G_dis_inf = opt.lambda_G_dis_inf
        
        self.lambda_G_class = opt.lambda_G_class
        self.lambda_G_recons = opt.lambda_G_recons
        self.lambda_G_kldiv1 = opt.lambda_G_kldiv1
        self.lambda_G_kldiv2 = opt.lambda_G_kldiv2
        self.lambda_G_match = opt.lambda_G_match

        # encoder
        self.encoder1 = Encoder1(opt)
        self.encoder2 = Encoder2(opt)

        # classifier
        self.classifier = nn.Linear(self.nz*2, self.nc) # nn.Sequential(nn.Linear(self.nz*2, self.nc), nn.Softmax(dim=-1))

        # discriminator
        self.discnet1 = DiscConv(nin=opt.nz, nout=1)
        self.discnet2 = DiscConv(nin=opt.nz, nout=1)

        # posterior module
        self.post_particle_net = PosteriorSampler(opt)
        
        # prior module
        self.prior_particle_net = PriorSampler(opt)
        self.prior_decoder = Decoder(opt)

        # initial value 
        self.initial_net = InitialValueModule(opt)
        
        # define the optimizers
        G_parameters = list(self.encoder1.parameters()) + list(self.encoder2.parameters()) + list(self.classifier.parameters()) +\
                       list(self.post_particle_net.parameters()) + list(self.initial_net.parameters()) +\
                       list(self.prior_particle_net.parameters()) + list(self.prior_decoder.parameters())
        D_parameters = list(self.discnet1.parameters()) + list(self.discnet2.parameters())

        self.optimizer_G = torch.optim.Adam(G_parameters, lr=opt.lr_gen)#, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(D_parameters, lr=opt.lr_dis)#, betas=(opt.beta1, 0.999))
        
        # define the learning rate scheduler
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))

        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

        self.loss_names = ['D_dis_frame1', 'D_dis_frame2', 'D_dis_inf', 'D', 'G_class', 'G_dis_frame1', 'G_dis_frame2', 'G_dis_inf',
                           'G_kldiv1', 'G_kldiv2', 'G_recons', 'G_match', 'G']

        # compute interpolation matrix
        self.compute_interpolate()
        
    def backward_G(self):
        self.compute_encoder()
        self.loss_G = self.loss_G_class * self.lambda_G_class + self.loss_G_dis_frame1 * self.lambda_G_dis_frame + self.loss_G_dis_frame2 * self.lambda_G_dis_frame \
                    + self.loss_G_kldiv1 * self.lambda_G_kldiv1 + self.loss_G_recons * self.lambda_G_recons + self.loss_G_kldiv2 * self.lambda_G_kldiv2 \
                    + self.loss_G_match * self.lambda_G_match + self.loss_G_dis_inf * self.lambda_G_dis_inf 
        self.loss_G.backward()
                      
    def backward_D(self):
        self.compute_discriminator()
        self.loss_D = self.loss_D_dis_frame1 * self.lambda_D_dis_frame + self.loss_D_dis_frame2 * self.lambda_D_dis_frame\
                    + self.loss_D_dis_inf * self.lambda_D_dis_inf
        self.loss_D.backward()
    
    def compute_weights(self, x_recon):
        # input: x_recon: (batch, frame, n_particle, 2200)
        # input: self.x_split: (batch, 1, frame, 44, 50)
        batch, frame, n_particle = x_recon.shape[:3]

        norm = (x_recon.view(batch, frame, n_particle, -1) - self.x_split.view(batch, frame, 1, -1)) ** 2
        norm = norm.sum(dim=-1)                                 # norm = norm.mean(dim=-1)
        likelihood = torch.exp(-norm)                           # (batch, frame, n_particle)
        
        # compute the weights
        weights = []
        yi = likelihood[:, [0], :]
        weights.append(yi)
        for i in range(1, frame):
            yi = yi * likelihood[:, [i], :]
            weights.append(yi)
        weights = torch.cat(weights, dim=1)                # (batch, frame, n_particle)

        # normalize the weights
        weights = F.softmax(weights, dim=-1) # weights / (weights.sum(dim=-1, keepdim=True) + 1e-5)
        return weights
    
    def compute_recons(self, x_recon, post_weights):
        # input: x_recon: (batch, frame, n_particle, 2200)
        # input: self.x_seq: (batch, 1, frame, 44, 50)
        batch, frame, n_particle = x_recon.shape[:3]
        norm = torch.abs(x_recon.view(batch, frame, n_particle, -1) - self.x_split.view(batch, frame, 1, -1))
        norm = norm.sum(dim=-1)
        self.loss_G_recons = torch.mean(norm * post_weights)
    
    def compute_kldiv1(self, posterior_mean, posterior_variance, prior_mean, prior_variance):
        # q~N(u1, sigma_1^2): proposal_mean, proposal_variance
        # p~N(u2, sigma_2^2): prior_mean, prior_variance
        # kldiv: log(sigma_2^2/sigma_1^2)/2 + (sigma_1^2+(u1-u2)^2)/2/sigma_2^2
        term1 = torch.log(prior_variance + 1e-10) - torch.log(posterior_variance + 1e-10)           # (batch, frame, theta_dim)
        term2 = (posterior_variance + (posterior_mean - prior_mean) ** 2) / (prior_variance + 1e-5) # (batch, frame, theta_dim)
        loss = (term1 + term2) / 2
        self.loss_G_kldiv1 = loss.mean()
    
    def compute_kldiv2(self, posterior_particles, posterior_weights, prior_particles, prior_weights):
        # q: {weights, particles} discrete distribution
        # p: {weights, particles} discrete distribution
        # weights: (batch, frame, n_particle) particles: (batch, frame, n_particle, theta_dim)

        # fit a gaussian distribution
        posterior_mean = torch.sum(posterior_weights.unsqueeze(dim=-1) * posterior_particles, dim=-2)
        posterior_variance = torch.sum(posterior_weights.unsqueeze(dim=-1) * 
                                      (posterior_particles - posterior_mean.unsqueeze(dim=-2)) ** 2, dim=-2)
        prior_mean = torch.sum(prior_weights.unsqueeze(dim=-1) * prior_particles, dim=-2)
        prior_variance = torch.sum(prior_weights.unsqueeze(dim=-1) *
                                  (prior_particles - prior_mean.unsqueeze(dim=-2)) ** 2, dim=-2)

        # kldiv: log(sigma_2^2/sigma_1^2)/2 + (sigma_1^2+(u1-u2)^2)/2/sigma_2^2
        term1 = torch.log(prior_variance + 1e-10) - torch.log(posterior_variance + 1e-10)           # (batch, frame, theta_dim)
        term2 = (posterior_variance + (posterior_mean - prior_mean) ** 2) / (prior_variance + 1e-5) # (batch, frame, theta_dim)
        loss = (term1 + term2) / 2
        self.loss_G_kldiv2 = loss.mean()
    
    def compute_class(self):
        # input: self.f_split: (batch, frame, nc)  softmax
        # input: self.y_seq: (batch, frame)
        self.loss_G_class = temporal_ce_loss(self.f_split[self.is_source == 1], self.y_seq[self.is_source == 1])
    
    def compute_encoder(self):
        # self.z1: (batch, frame, nh)
        # self.z2: (batch, frame, n_particle, nh)
        # self.weights: (batch, frame, n_particle)
        # firstly compute loss of discnet1
        d_frame = self.discnet1(self.z1)                                # (batch, frame, nh)
        d_frame_target = d_frame[self.is_source == 0]                     
        self.loss_G_dis_frame1 = - torch.log(d_frame_target + 1e-10).mean()

        # secondly compute loss of discnet2
        d_frame = self.discnet2(self.z2)                                # (batch, frame, n_particle, nh)
        d_frame_target = d_frame[self.is_source == 0]
        weights_target = self.weights[self.is_source == 0]            
        loss_G_dis_frame2 = - torch.log(d_frame_target.squeeze(dim=-1) + 1e-10) * weights_target
        self.loss_G_dis_frame2 = loss_G_dis_frame2.sum(-1).mean()

        # finally compute loss of infdiscnet
        d_seq = self.discnet2(self.z2_inf)                     # (batch, inf_step-1, n_particle, 1)
        d_seq_target = d_seq[self.is_source == 0]
        # G(-epsilon)
        weights_target_inf = self.inf_weights[self.is_source == 0]
        loss_G_epsilon1 = - torch.log(d_seq_target.mean(-1) + 1e-10) * weights_target_inf
        # first weighted average of n_particle and then average the batch
        loss_G_epsilon1 = loss_G_epsilon1.sum(-1).mean(0)

        m_seq_inf = torch.arange(-1 + 1 / self.inf_step, 0, 1 / self.inf_step).to(self.device)
        # integrate the steps
        loss_G_epsilon1 = loss_G_epsilon1 * self.inf_scale / (- 2 * m_seq_inf * torch.sqrt(- torch.log(-m_seq_inf)) * self.inf_step)
        loss_G_epsilon1 = loss_G_epsilon1.sum()   
        
        # G(-2*epsilon)
        loss_G_epsilon2 = - torch.log(d_seq_target[:, :-1].mean(-1) + 1e-10) * weights_target_inf[:, :-1]
        loss_G_epsilon2 = loss_G_epsilon2.sum(-1).mean(0)
        loss_G_epsilon2 = loss_G_epsilon2 * self.inf_scale / (- 2 * m_seq_inf[:-1] * torch.sqrt(- torch.log(-m_seq_inf[:-1])) * self.inf_step)
        loss_G_epsilon2 = loss_G_epsilon2.sum()
        
        # G(-epsilon) + |G(-2*epsilon) - G(-*epsilon)|
        self.loss_G_dis_inf = loss_G_epsilon1 + torch.abs(loss_G_epsilon2 - loss_G_epsilon1)
           
    def compute_discriminator(self):
        # self.z1: (batch, frame, nh)
        # self.z2: (batch, frame, n_particle, nh)
        # self.weights: (batch, frame, n_particle)
        # firstly compute loss of framediscnet
        d_frame = self.discnet1(self.z1.detach())                        # (batch, frame, nh)
        d_frame_source = d_frame[self.is_source == 1]
        d_frame_target = d_frame[self.is_source == 0]
        self.loss_D_dis_frame1 = - torch.log(d_frame_source + 1e-10).mean() \
                                 - torch.log(1 - d_frame_target + 1e-10).mean()

        # secondly compute loss of discnet2
        d_frame = self.discnet2(self.z2.detach())                        # (batch, frame, nh)
        d_frame_source = d_frame[self.is_source == 1]
        d_frame_target = d_frame[self.is_source == 0]
        weights_source = self.weights[self.is_source == 1].detach()
        weights_target = self.weights[self.is_source == 0].detach()    
        term1 = - torch.log(d_frame_source.squeeze(dim=-1) + 1e-10) * weights_source 
        term2 = - torch.log(1 - d_frame_target.squeeze(dim=-1) + 1e-10) * weights_target
        self.loss_D_dis_frame2 = term1.sum(-1).mean() + term2.sum(-1).mean()

        # finally compute infinity domain invariance loss
        d_seq = self.discnet2(self.z2_inf.detach())                  # (batch, inf_step-1, n_particle, 1)
        d_seq_source = d_seq[self.is_source == 1]
        d_seq_target = d_seq[self.is_source == 0]
        # G(-epsilon)
        # first weighted average of n_particle and then average the batch
        weights_source_inf = self.inf_weights[self.is_source == 1].detach()
        weights_target_inf = self.inf_weights[self.is_source == 0].detach()
        term1 = - torch.log(d_seq_source.squeeze(dim=-1) + 1e-10) * weights_source_inf
        term2 = - torch.log(1 - d_seq_target.squeeze(dim=-1) + 1e-10) * weights_target_inf
        term1 = term1.sum(-1).mean(0)
        term2 = term2.sum(-1).mean(0)
        loss_G_epsilon1 = term1 + term2

        m_seq_inf = torch.arange(-1 + 1 / self.inf_step, 0, 1 / self.inf_step).to(self.device)
        # integrate the steps
        loss_G_epsilon1 = loss_G_epsilon1 * self.inf_scale / (- 2 * m_seq_inf * torch.sqrt(- torch.log(-m_seq_inf)) * self.inf_step)
        loss_G_epsilon1 = loss_G_epsilon1.sum()

        # G(-2*epsilon)
        term1 = - torch.log(d_seq_source[:, :-1].squeeze(dim=-1) + 1e-10) * weights_source_inf[:, :-1]
        term2 = - torch.log(1 - d_seq_target[:, :-1].squeeze(dim=-1) + 1e-10) * weights_target_inf[:, :-1]
        term1 = term1.sum(-1).mean(0)
        term2 = term2.sum(-1).mean(0)
        loss_G_epsilon2 = term1 + term2

        loss_G_epsilon2 = loss_G_epsilon2 * self.inf_scale / (- 2 * m_seq_inf[:-1] * torch.sqrt(- torch.log(-m_seq_inf[:-1])) * self.inf_step)
        loss_G_epsilon2 = loss_G_epsilon2.sum()
        
        # first weighted average of n_particle
        self.loss_D_dis_inf = loss_G_epsilon1 + torch.abs(loss_G_epsilon2 - loss_G_epsilon1)

    def compute_match(self, d2ht_dt2, d2logw_dt2):
        loss = (d2logw_dt2 - d2ht_dt2) ** 2
        self.loss_G_match = loss.sum(dim=-1).mean()
    
    def compute_interpolate(self):
        # preapre for infinity domain invariance
        m = torch.arange(-1 + 1 / self.inf_step, 0, 1 / self.inf_step).to(self.device)
        # m = -e^(-t^2/a^2)  -> t = a\sqrt(-log(-m))
        t = self.inf_scale *  torch.sqrt(- torch.log(- m))
        assert t.min() >= 1 and t.max() <= self.n_frame
        self.matrix_interpolate = torch.zeros(self.inf_step-1, self.n_frame).to(self.device)

        for i in range(self.inf_step - 1):
            t_s = t[i].item()
            self.matrix_interpolate[i, int(math.floor(t_s))-1] = math.ceil(t_s) - t_s
            self.matrix_interpolate[i, int(math.ceil(t_s))-1] = t_s - math.floor(t_s)

    def forward(self, is_train=False):
        # self.x_split: (batch, 1, frame, 44, 50)
        # self.y_seq: (batch,)
        batch, _, frame = self.x_split.shape[:3]

        # encoder forward() to compute z1
        if self.seq_first:
            self.hidden_encoder1 = None   # hidden state for encoder 1
            self.hidden_encoder2 = None   # hidden state for encoder 2
            self.hidden_inv = None        # hidden state for initial value module

        self.z1, hidden = self.encoder1(self.x_split, self.hidden_encoder1)              # (batch, frame, nz)
        self.hidden_encoder1 = (hidden[0].detach(), hidden[1].detach())
        
        # generate initial value
        if self.seq_first:
            self.x_last = self.z1.detach()
            initial_value, hidden = self.initial_net(self.x_last, self.hidden_inv)
        else:
            initial_value, hidden = self.initial_net(self.x_last, self.hidden_inv)
            self.x_last = self.z1.detach()
        self.hidden_inv = hidden.detach()

        # compute prior for theta
        prior_particles, prior_proposal_mean, prior_proposal_variance, _ = self.prior_particle_net(initial_value, self.z1)

        # decoder forward()
        x_recon = self.prior_decoder(prior_particles)

        # importance weights sampling for prior
        prior_weights = self.compute_weights(x_recon)

        # compute posterior for theta
        t_seq = torch.arange(1, frame + 1)[None, :, None].float().expand(batch, frame, 1).to(self.device)
        if is_train:
            is_diff = True
        else:
            is_diff = False
        post_particles, post_weights, post_proposal_mean, post_proposal_variance, deriv = self.post_particle_net(initial_value, t_seq, is_diff)

        # prob encoder forward()
        self.z2, hidden, x_feat = self.encoder2(self.x_split, post_particles, self.hidden_encoder2)
        self.hidden_encoder2 = (hidden[0].detach(), hidden[1].detach()) # hidden.detach()

        # concatenate z1 and z2 to compute z_prime
        z1_expand = self.z1.unsqueeze(dim=-2).expand(batch, frame, self.n_particle, self.nz)
        z_prime = torch.cat([z1_expand, self.z2], dim=-1)              # (batch, frame, n_particle, 2*nz)

        # classifier forward()
        y_logits = self.classifier(z_prime)

        # marginalize n_particle of logits
        y_pred = torch.sum(post_weights.unsqueeze(dim=-1) * y_logits, dim=-2)     # (batch, frame, nc)
        self.f_split = torch.softmax(y_pred, dim=-1)

        if is_train:
            # compute loss for encoder
            self.compute_class()
            self.compute_recons(x_recon, prior_weights)
            self.compute_kldiv1(posterior_mean=post_proposal_mean, posterior_variance=post_proposal_variance, 
                                prior_mean=prior_proposal_mean, prior_variance=prior_proposal_variance)
            self.compute_kldiv2(posterior_particles=post_particles, posterior_weights=post_weights,
                                prior_particles=prior_particles, prior_weights=prior_weights)
            
            d2logw_dt2, dmean_dt, d2mean_dt2, dvar_dt, d2var_dt2 = deriv
            # compute d2ht_dt2
            term1 = dvar_dt ** 2 * post_proposal_variance - 2 * dmean_dt ** 2 * post_proposal_variance ** 2 - d2var_dt2 * post_proposal_variance
            term1 = term1.unsqueeze(dim=-2)
            term2 = (2 * d2mean_dt2 * post_proposal_variance - 4 * dvar_dt * dmean_dt).unsqueeze(dim=-2) * \
                    (post_particles - post_proposal_mean.unsqueeze(dim=-2)) * post_proposal_variance.unsqueeze(dim=-2)
            term3 = (post_particles - post_proposal_mean.unsqueeze(dim=-2)) ** 2 * (d2var_dt2 * post_proposal_variance - 2 * dvar_dt ** 2).unsqueeze(dim=-2)
            
            term4 = 2 * post_proposal_variance.unsqueeze(dim=-2) ** 3 + 1e-5
            d2ht_dt2 = (term1 + term2 + term3) / term4
            d2ht_dt2 = d2ht_dt2.mean(dim=-1, keepdim=True)   # (batch, frame, particles, 1)

            # gradient matching
            self.compute_match(d2ht_dt2, d2logw_dt2)
            
            self.weights = post_weights

            # infinity domain invariance
            m_seq_inf = torch.arange(-1 + 1 / self.inf_step, 0, 1 / self.inf_step)[None, :, None].float().expand(batch, self.inf_step - 1, 1).to(self.device)
            # m = -e^(-t^2/a^2)  -> t = a\sqrt(-log(-m))
            t_seq_inf = self.inf_scale *  torch.sqrt(- torch.log(- m_seq_inf))

            # interpolate
            x_inf = torch.einsum('ij, bcjhw->bcihw', self.matrix_interpolate, x_feat) # x_feat (B, C, T, H, W)
            # forward infinity
            self.inf_particles, self.inf_weights, _, _, _ = self.post_particle_net(initial_value, t_seq_inf, is_diff=False)
            self.z2_inf, _ = self.encoder2.forward_inf(x_inf, self.inf_particles)
                  
    def optimize_parameters(self):
        self.forward(is_train=True)
        # update the discriminator D
        self.set_requires_grad([self.discnet1, self.discnet2], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update the encoder E and predictor F
        self.set_requires_grad([self.discnet1, self.discnet2], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    
    def test(self, epoch, dataloader, is_eval):
        if not is_eval:
            self.load()
        self.eval()

        self.acc_reset_oulu()
        # all_batch_size = 0
        # all_time = 0.0
        for data in dataloader:
            # set input
            x_seq, y_seq, t_seq, u_seq = [to_tensor(_, self.device) for _ in data]
            # x: (batch, frame=40, 44, 50)
            # y: (batch, 1)  t: (batch, 1)  is_source: (batch, 1)
            x_seq = x_seq[:, :, None, :, :].transpose(1, 2).contiguous()  # (batch, 1, frame=40, 44, 50)
            
            # frame = x_seq.shape[2]

            # split sequence
            x_splits = torch.split(x_seq, self.n_frame, dim=2)  # self.n_frame

            f_seq = []
            
            for i in range(len(x_splits)):
                if i == 0:
                    self.seq_first = True
                else:
                    self.seq_first = False
                # fetch the split
                x_split = x_splits[i]

                self.set_input(input=(x_split, y_seq, t_seq, u_seq))
                self.forward()
                # all_time += t_delta
                
                f_seq.append(self.f_split.detach())
            
            # compute acc for the whole seq
            f_seq = torch.cat(f_seq, dim=1)
            self.g_seq, self.top3_pred = vote_video_classification_result(f_seq, y_seq)
            self.acc_update_oulu()
            
        #     # accumulate dataset
        #     sing_batch_size = x_seq.shape[0]
        #     sing_batch_size_source = self.is_source.sum()
        #     sing_batch_size_target = sing_batch_size - sing_batch_size_source
        #     all_batch_size += sing_batch_size
        #     print(f"Number of batch size: {sing_batch_size}. Number of source batch size: {sing_batch_size_source}. Number of target batch size: {sing_batch_size_target}")
        # print(f'Number of sequences: {all_batch_size}, Total cost time: {all_time}, Average testing time: {round(all_time / 4 / all_batch_size * 100, 7)}')
        # accuracy message
        if is_eval:
            self.acc_msg = f'Eval[{epoch+1}] Oulu: {self.acc_domain} src: {self.acc_source} tgt:{self.acc_target} test_acc:{self.total_acc:.3f} top3_acc:{self.top3_acc:.3f}'
        else:
            self.acc_msg = f'Test[{epoch+1}] Oulu: {self.acc_domain} src: {self.acc_source} tgt:{self.acc_target} test_acc:{self.total_acc:.3f} top3_acc:{self.top3_acc:.3f}'
        
        # print accuracy message
        with open(self.valid_log, 'a') as f:
            f.write(self.acc_msg + "\n")
        print(self.acc_msg)
        return self.acc_target * 100