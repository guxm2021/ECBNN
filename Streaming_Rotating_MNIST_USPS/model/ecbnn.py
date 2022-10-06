import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from model.base import BaseModel, to_np
import time

# ======================================================================================================================
class EncoderRNN(nn.Module):
    def __init__(self, opt):
        super(EncoderRNN, self).__init__()
        # The design of self.conv and self.fc_pred adopts LeNet
        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(20, 50, 5), nn.Dropout2d(p=0.5), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
        )
        self.nz = opt.nz
        self.nc = opt.nc

        self.fc_down = nn.Sequential(
            nn.Linear(50*4*4, self.nz),
            nn.ReLU(inplace=True),
        )
        
        self.rnn = nn.GRU(self.nz, self.nz, num_layers=2, batch_first=True)

        self.fc_rnn = nn.Sequential(
            nn.Linear(self.nz, self.nz),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x, hidden=None):
        """
        :param x: (batch, frame, 1, 28, 28)
        :return: y: (batch, frame, nc) z: (batch, frame, nz) hidden
        """
        batch, frame = x.shape[:2]
        x = x.reshape(batch*frame, *x.shape[2:])
        # extract features
        x = self.conv(x)
        # flatten
        x = x.reshape(batch*frame, -1)
        x = self.fc_down(x)
        # rnn (batch first)
        x = x.reshape(batch, frame, -1)
        # x = x.transpose(0, 1).contiguous()

        x, hidden_state = self.rnn(x, hidden)
        # x = x.transpose(0, 1).contiguous()
        x = x.reshape(batch*frame, -1)
        ###################################
        z1 = self.fc_rnn(x)
        z1 = z1.view(batch, frame, -1)
        ###################################
        return z1, hidden_state         # (batch, frame, nz)


class Classifier(nn.Module):
    """
    Classifier doing binary classification: class 0 v.s. class 1
    """
    def __init__(self, opt):
        super(Classifier, self).__init__()
        nz = opt.nz
        nc = opt.nc
        self.fc_pred = nn.Sequential(
            nn.Linear(2 * nz, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, nc)
        )
    
    def forward(self, x):
        # input x: (batch, frame, n_particle, 2*nz)
        if x.dim() == 4:
            batch, frame, n_particle = x.shape[:3]
            x = x.view(batch*frame*n_particle, -1)
            x_logits =  self.fc_pred(x)
            x_logits = x_logits.view(batch, frame, n_particle, -1)
            x_log_softmax = F.log_softmax(x_logits, dim=-1)
        else:
            batch, frame = x.shape[:2]
            x = x.view(batch*frame, -1)
            x_logits =  self.fc_pred(x)
            x_logits = x_logits.view(batch, frame, -1)
            x_log_softmax = F.log_softmax(x_logits, dim=-1)
        return x_logits, x_log_softmax


class FrameDiscNet(nn.Module):
    """
    Frame-level Discriminator doing binary classification: source v.s. target
    """
    def __init__(self, opt):
        super(FrameDiscNet, self).__init__()
        nz = opt.nz
        self.discnet = nn.Sequential(
            nn.Linear(nz, nz * 2),
            nn.BatchNorm1d(nz * 2),
            nn.ReLU(inplace=True),
            nn.Linear(nz * 2, nz * 2),
            nn.BatchNorm1d(nz * 2),
            nn.ReLU(inplace=True),
            nn.Linear(nz * 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # input x: (batch, frame, nz) or (batch, frame, n_particle, nz)
        if x.dim() == 3:
            batch, frame = x.shape[:2]
            x = x.view(batch*frame, -1)
            x = self.discnet(x)
            x = x.view(batch, frame, -1)
        else:
            batch, frame, n_particle = x.shape[:3]
            x = x.view(batch*frame*n_particle, -1)
            x = self.discnet(x)
            x = x.view(batch, frame, n_particle, -1)
        return x                   # output: sigmoid

# ======================================================================================================================
# Implement posterier modules
class InitialValueModule(nn.Module):
    """
    IntialValueModule is adopted to generate the initial value for the next batch
    Only use the observations from last batch
    """
    def __init__(self, opt):
        super(InitialValueModule, self).__init__()
        self.nz = opt.nz
        self.theta_dim = 16 * 16
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
        self.theta_dim = 16 * 16
        self.bottom1 = nn.Linear(1, 256)
        self.bottom2 = nn.Linear(256, 256)
        self.bottom3 = nn.Linear(256, 256)
        self.bottom4 = nn.Linear(256, 256)

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
        self.theta_dim = 16 * 16

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
            # dlogw_dt = torch.autograd.grad(log_weights, t_seq_expand, torch.ones_like(log_weights), create_graph=True)[0]
            # d2logw_dt2 = torch.autograd.grad(dlogw_dt, t_seq_expand, torch.ones_like(dlogw_dt), create_graph=True)[0]
            dlogw_dt = torch.autograd.grad(log_weights, t_seq_expand, torch.ones_like(log_weights), create_graph=True)[0]
            d2logw_dt2 = torch.autograd.grad(dlogw_dt, t_seq_expand, torch.ones_like(dlogw_dt), create_graph=True)[0]
            # d2logw_dt2 = d2logw_dt2.squeeze(dim=-1)
            # gradient: (batch, frame, n_particle)

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

# ======================================================================================================================
# implement prior modules
class Decoder(nn.Module):
    """
    Decoder is adopted to reconstruct x from particles and computes the importance weights
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(256, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        """
        :param x (particles): (batch, frame, n_particle, 256)
        :return: x_recons
        """
        batch, frame, n_particle, channel = x.shape
        x = x.view(batch*frame*n_particle, channel)
        x = self.main(x)
        x = x.view(batch, frame, n_particle, 784)
        return x


class PriorSampler(nn.Module):
    """
    PriorSampler is adopted to compute the prior for x and compute kL divergence
    """
    def __init__(self, opt):
        super(PriorSampler, self).__init__()
        self.nz = opt.nz
        self.device = opt.device
        self.theta_dim = 16 * 16
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
        x, hidden_state = self.rnn(x, hidden)                         # (batch, frame, nz)
        
        # proposal
        proposal_mean = self.head1(x)                                 # mean for x prior
        proposal_variance = self.positive_act(self.head2(x))          # variance (delta ** 2, not delta) for x prior

        # sampling
        particles = torch.randn(batch, frame, self.n_particle, self.theta_dim).to(self.device) * \
            torch.sqrt(proposal_variance.unsqueeze(dim=-2)) + proposal_mean.unsqueeze(dim=-2)
        return particles, proposal_mean, proposal_variance, hidden_state


class ProbEncoder(nn.Module):
    """
    ProbEncoder implements the feature extraction using particles
    """
    def __init__(self, opt):
        super(ProbEncoder, self).__init__()
        # network parameters for probencoder
        self.nz = opt.nz
        
        self.n_particle = opt.n_particle
        
        # convolution
        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(20, 50, 5), nn.Dropout2d(p=0.5), nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
        )
        
        # bnn
        self.theta_dim = 16 * 16
        
        # linear
        self.fc_down = nn.Sequential(
            nn.Linear(50*4*4, self.nz),
            nn.ReLU(inplace=True),
        )
        
        self.rnn = nn.GRU(self.nz, self.nz, num_layers=2, batch_first=True)
        self.fc_rnn = nn.Sequential(
            nn.Linear(self.nz, self.nz),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x, particles, hidden=None):
        """
        :param x: (batch, frame, 1, 28, 28)
        :param particles: (batch, frame, n_particle, theta_dim)
        :return: z2
        """
        batch, frame = x.shape[:2]             
        x = x.reshape(batch*frame, *x.shape[2:])  # (batch*frame, 1, 28, 28)
        # conv network
        x = self.conv(x)                          # (batch*frame, 50, 4, 4)

        # bnn
        x = x.view(batch, frame, 50, 16)
        x_inter = x
        bnn = particles.view(batch, frame, self.n_particle, 16, 16)
        x = F.relu(torch.einsum('bfnij, bfcj -> bfnci', bnn, x)) # (batch, frame, n_particle, 50, 16)
        x = x.reshape(batch, frame, self.n_particle, -1)
        
        # fc_down
        x = self.fc_down(x)                                     # (batch, frame, n_particle, nz)
        # rnn
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch*self.n_particle, frame, self.nz)
        # x = x.transpose(0, 1).contiguous()

        x, hidden_state = self.rnn(x, hidden)
        # x = x.transpose(0, 1).contiguous()
        x = x.view(batch, self.n_particle, frame, self.nz)
        x = x.transpose(1, 2).contiguous()
        z2 = self.fc_rnn(x)                    
        return z2, hidden_state, x_inter

    def forward_inf(self, x, particles, hidden=None):
        """
        :param x: (batch, frame, 50, 16)
        :param particles: (batch, frame, n_particle, theta_dim)
        :return: z2
        """
        batch, frame = x.shape[:2]             
        # bnn
        x = x.view(batch, frame, 50, 16)
        bnn = particles.view(batch, frame, self.n_particle, 16, 16)
        x = F.relu(torch.einsum('bfnij, bfcj -> bfnci', bnn, x)) # (batch, frame, n_particle, 50, 16)
        x = x.reshape(batch, frame, self.n_particle, -1)
        
        # fc_down
        x = self.fc_down(x)                                     # (batch, frame, n_particle, nz)
        # rnn
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch*self.n_particle, frame, self.nz)
        # x = x.transpose(0, 1).contiguous()

        x, hidden_state = self.rnn(x, hidden)
        # x = x.transpose(0, 1).contiguous()
        x = x.view(batch, self.n_particle, frame, self.nz)
        x = x.transpose(1, 2).contiguous()
        z2 = self.fc_rnn(x)                    
        return z2, hidden_state                             

# ======================================================================================================================
class ECBNN(BaseModel):
    def __init__(self, opt):
        super(ECBNN, self).__init__(opt)
        # encoder
        self.encoder1 = EncoderRNN(opt)
        self.encoder2 = ProbEncoder(opt)

        # classifier
        self.classifier = Classifier(opt)

        # discriminator
        self.discnet1 = FrameDiscNet(opt)
        self.discnet2 = FrameDiscNet(opt)

        # posterior module
        self.post_particle_net = PosteriorSampler(opt)
        
        # prior module
        self.prior_particle_net = PriorSampler(opt)
        self.prior_decoder = Decoder()

        # initial value 
        self.initial_net = InitialValueModule(opt)
        
        # define the optimizers
        G_parameters = list(self.encoder1.parameters()) + list(self.encoder2.parameters()) + list(self.classifier.parameters()) +\
                       list(self.post_particle_net.parameters()) + list(self.initial_net.parameters()) +\
                       list(self.prior_particle_net.parameters()) + list(self.prior_decoder.parameters())
        D_parameters = list(self.discnet1.parameters()) + list(self.discnet2.parameters())

        self.optimizer_G = torch.optim.Adam(G_parameters, lr=opt.lr_gen, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(D_parameters, lr=opt.lr_dis, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        
        # define the learning rate scheduler
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))

        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

        self.loss_names = ['D_dis_frame1', 'D_dis_frame2', 'D_dis_inf', 'D', 'G_class', 'G_dis_frame1', 'G_dis_frame2', 'G_dis_inf',
                           'G_kldiv1', 'G_kldiv2', 'G_recons', 'G_match', 'G']

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
        # input: x_recon: (batch, frame, n_particle, 1, 28, 28)
        # input: self.x_seq: (batch, frame, 1, 28, 28)
        batch, frame, n_particle = x_recon.shape[:3]

        norm = (x_recon.view(batch, frame, n_particle, -1) - self.x_seq.view(batch, frame, 1, -1)) ** 2
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
        # input: x_recon: (batch, frame, n_particle, 784)
        # input: self.x_seq: (batch, frame, 1, 28, 28)
        batch, frame, n_particle = x_recon.shape[:3]
        norm = torch.abs(x_recon.view(batch, frame, n_particle, -1) - self.x_seq.view(batch, frame, 1, -1))
        norm = norm.sum(dim=-1)                                 # norm = norm.mean(dim=-1)
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
    
    def compute_class(self, y_pred):
        # input: y_pred: (batch, frame, nc)  log_softmax
        # input: self.y_seq: (batch, frame)
        y_pred_source = y_pred[self.domain_mask == 1]
        y_seq_source = self.y_seq[self.domain_mask == 1]
        self.loss_G_class = F.nll_loss(y_pred_source.view(-1, self.nc), y_seq_source.view(-1))
    
    def compute_encoder(self):
        # self.z1: (batch, frame, nh)
        # self.z2: (batch, frame, n_particle, nh)
        # self.weights: (batch, frame, n_particle)
        # firstly compute loss of discnet1
        d_frame = self.discnet1(self.z1)                                # (batch, frame, nh)
        d_frame_target = d_frame[self.domain_mask == 0]                     
        self.loss_G_dis_frame1 = - torch.log(d_frame_target + 1e-10).mean()

        # secondly compute loss of discnet2
        d_frame = self.discnet2(self.z2)                                # (batch, frame, n_particle, nh)
        d_frame_target = d_frame[self.domain_mask == 0]
        weights_target = self.weights[self.domain_mask == 0]            
        loss_G_dis_frame2 = - torch.log(d_frame_target.squeeze(dim=-1) + 1e-10) * weights_target
        self.loss_G_dis_frame2 = loss_G_dis_frame2.sum(-1).mean()

        # finally compute loss of infdiscnet
        d_seq = self.discnet2(self.z2_inf)                     # (batch, inf_step-1, n_particle, 1)
        d_seq_target = d_seq[self.domain_mask == 0]
        # G(-epsilon)
        weights_target_inf = self.inf_weights[self.domain_mask == 0]
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
        d_frame_source = d_frame[self.domain_mask == 1]
        d_frame_target = d_frame[self.domain_mask == 0]
        self.loss_D_dis_frame1 = - torch.log(d_frame_source + 1e-10).mean() \
                                 - torch.log(1 - d_frame_target + 1e-10).mean()

        # secondly compute loss of discnet2
        d_frame = self.discnet2(self.z2.detach())                        # (batch, frame, nh)
        d_frame_source = d_frame[self.domain_mask == 1]
        d_frame_target = d_frame[self.domain_mask == 0]
        weights_source = self.weights[self.domain_mask == 1].detach()
        weights_target = self.weights[self.domain_mask == 0].detach()    
        term1 = - torch.log(d_frame_source.squeeze(dim=-1) + 1e-10) * weights_source 
        term2 = - torch.log(1 - d_frame_target.squeeze(dim=-1) + 1e-10) * weights_target
        self.loss_D_dis_frame2 = term1.sum(-1).mean() + term2.sum(-1).mean()

        # finally compute infinity domain invariance loss
        d_seq = self.discnet2(self.z2_inf.detach())                  # (batch, inf_step-1, n_particle, 1)
        d_seq_source = d_seq[self.domain_mask == 1]
        d_seq_target = d_seq[self.domain_mask == 0]
        # G(-epsilon)
        # first weighted average of n_particle and then average the batch
        weights_source_inf = self.inf_weights[self.domain_mask == 1].detach()
        weights_target_inf = self.inf_weights[self.domain_mask == 0].detach()
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
        # self.x_seq: (batch, frame, 1, 28, 28)
        # self.y_seq: (batch, frame)
        batch, frame = self.y_seq.shape

        # encoder forward() to compute z1
        if self.seq_first:
            self.hidden_encoder1 = None   # hidden state for encoder 1
            self.hidden_encoder2 = None   # hidden state for encoder 2 (prob encoder)
            self.hidden_inv = None        # hidden state for initial value module
            self.hidden_inf = None        # hidden state for infinity module
        self.z1, hidden = self.encoder1(self.x_seq, self.hidden_encoder1)              # (batch, frame, nz)
        self.hidden_encoder1 = hidden.detach()
        
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
        # self.z2, hidden = self.encoder2(self.x_seq, post_particles, self.hidden_encoder2)
        self.z2, hidden, x_inter = self.encoder2(self.x_seq, post_particles, self.hidden_encoder2)
        self.hidden_encoder2 = hidden.detach()

        # concatenate z1 and z2 to compute z_prime
        z1_expand = self.z1.unsqueeze(dim=-2).expand(batch, frame, self.n_particle, self.nz)
        z_prime = torch.cat([z1_expand, self.z2], dim=-1)              # (batch, frame, n_particle, 2*nz)

        # classifier forward()
        y_logits, _ = self.classifier(z_prime)

        # marginalize n_particle of logits
        y_pred = torch.sum(post_weights.unsqueeze(dim=-1) * y_logits, dim=-2)   # (batch, frame, nc)
        y_log_softmax = F.log_softmax(y_pred, dim=-1)

        # prediction task
        g_seq = y_log_softmax.detach()
        self.g_seq = torch.argmax(g_seq, dim=-1)

        if is_train:
            # t1 = time.time()
            # compute loss for encoder
            self.compute_class(y_log_softmax)
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
            
            self.weights = post_weights # post_weights

            # infinity domain invariance
            m_seq_inf = torch.arange(-1 + 1 / self.inf_step, 0, 1 / self.inf_step)[None, :, None].float().expand(batch, self.inf_step - 1, 1).to(self.device)
            # m = -e^(-t^2/a^2)  -> t = a\sqrt(-log(-m))
            t_seq_inf = self.inf_scale *  torch.sqrt(- torch.log(- m_seq_inf))
            # interpolate
            x_inf = torch.einsum('ij, bjmn->bimn', self.matrix_interpolate, x_inter)
            # forward infinity
            self.inf_particles, self.inf_weights, _, _, _ = self.post_particle_net(initial_value, t_seq_inf, is_diff=False)
            self.z2_inf, hidden = self.encoder2.forward_inf(x_inf, self.inf_particles, self.hidden_inf)
            #self.hidden_inf = hidden.detach()
                  
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
    