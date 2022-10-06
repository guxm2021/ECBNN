import os
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

# ======================================================================================================================
# define functions
def to_np(x):
    return x.detach().cpu().numpy()

class LeNetRNN(nn.Module):
    """
    Adapted LeNet + RNN 
    """
    def __init__(self, opt):
        super(LeNetRNN, self).__init__()
        # The design of self.conv and self.fc_pred adopts LeNet
        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(20, 50, 5), nn.Dropout2d(p=0.5), nn.MaxPool2d(2, 2), nn.ReLU(),
        )
        self.nz = opt.nz
        self.nc = opt.nc

        self.fc_down = nn.Sequential(
            nn.Linear(50*4*4, self.nz),
            nn.ReLU(),
        )

        self.fc_pred = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.nc)
        )

        self.rnn = nn.GRU(self.nz, self.nz, num_layers=2)
        self.fc_rnn = nn.Sequential(
            nn.Linear(self.nz, self.nz),
            nn.ReLU(),
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
        x = x.transpose(0, 1).contiguous()
        if hidden == None:
            x, hidden_state = self.rnn(x)
        else:
            x, hidden_state = self.rnn(x, hidden)
        x = x.transpose(0, 1).contiguous()
        x = x.reshape(batch*frame, -1)
        # handle temporal features
        z = self.fc_rnn(x)
        # classifier
        y = self.fc_pred(z)
        y = F.log_softmax(y, dim=-1)
        # reshape the output
        z = z.view(batch, frame, -1)
        y = y.view(batch, frame, -1)
        return y, z, hidden_state


class LeNet(nn.Module):
    """
    Adapted LeNet
    """
    def __init__(self, opt):
        super(LeNet, self).__init__()
        # The design of self.conv and self.fc_pred adopts LeNet
        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(20, 50, 5), nn.Dropout2d(p=0.5), nn.MaxPool2d(2, 2), nn.ReLU(),
        )
        self.nz = opt.nz
        self.nc = opt.nc

        self.fc_down = nn.Sequential(
            nn.Linear(50*4*4, self.nz),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.fc_pred = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.nc)
        )
    
    def forward(self, x):
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
        z = self.fc_down(x)
        # pred
        y = self.fc_pred(z)
        y = F.log_softmax(y, dim=-1)
        # reshape the output
        z = z.view(batch, frame, -1)
        y = y.view(batch, frame, -1)
        return y, z


class DiscFc(nn.Module):
    def __init__(self, nin, nh, nout):
        super(DiscFc, self).__init__()
        # nin: self.nz
        # nout: dimension of output
        self.discnet = nn.Sequential(
            nn.Linear(nin, nh),
            nn.BatchNorm1d(nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.BatchNorm1d(nh),
            nn.ReLU(),
            nn.Linear(nh, nout),
        )
    
    def forward(self, x):
        """
        :param x: (batch, frame, nz)
        :return: y: (batch, frame, nout)
        """
        batch, frame = x.shape[:2]
        x = x.view(batch*frame, -1)
        x = self.discnet(x)
        x = x.view(batch, frame, -1)
        return x   # output: logits
        
# ======================================================================================================================
# define base model
class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt

        self.train_log = opt.outf + '/train.log'
        self.valid_log = opt.outf + '/valid.log'
        self.model_path = opt.outf + '/model.pth'

        # write opt
        with open(self.train_log, 'a') as f:
            f.write(str(self.opt))
            f.write("\n")
        
        with open(self.valid_log, 'a') as f:
            f.write(str(self.opt))
            f.write("\n")

        # other parameters
        self.n_frame = opt.n_frame
        self.device = opt.device

        # evaluate 
        self.best_score = 0.0
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def check_save(self, score):
        if score > self.best_score:
            self.best_score = score
            self.save()
            print('Get better performance, save the model!!!')

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        try:
            print('load model from {}'.format(self.model_path))
            self.load_state_dict(torch.load(self.model_path))
            print('done!')
        except:
            print('failed!')

    def set_train_input(self, x_source_seq, y_source_seq, x_target_seq, y_target_seq):
        # frame = 90
        # x_source_seq: (source_batch, frame, 1, 28, 28) y_source_seq: (source_batch, frame)
        # x_target_seq: (target_batch, frame, 1, 28, 28) y_target_seq: (target_batch, frame)
        source_batch = x_source_seq.shape[0]
        target_batch = x_target_seq.shape[0]

        x_source_seq = x_source_seq[:, 0:40].to(self.device)        # source domain: 0~40 frames of MNIST
        y_source_seq = y_source_seq[:, 0:40].to(self.device)
        x_target_seq = x_target_seq[:, 50:90].to(self.device)       # target domain: 50~90 frames of USPS
        y_target_seq = y_target_seq[:, 50:90].to(self.device)
        
        # concatenate data
        x_seq = torch.cat([x_source_seq, x_target_seq], dim=0)      
        y_seq = torch.cat([y_source_seq, y_target_seq], dim=0)

        mask_list = [1] * source_batch + [0] * target_batch         # incovenience of fetching data from source and target domain 
        domain_mask = torch.IntTensor(mask_list)
        return x_seq, y_seq, domain_mask

    def set_test_input(self, x_source_seq, y_source_seq, x_target_seq, y_target_seq, shift):
        # frame = 90, 0<=shift<=50
        # x_source_seq: (source_batch, frame, 1, 28, 28) y_source_seq: (source_batch, frame)
        # x_target_seq: (target_batch, frame, 1, 28, 28) y_target_seq: (target_batch, frame)
        source_batch = x_source_seq.shape[0]
        target_batch = x_target_seq.shape[0]
        
        # source domain: 0~40 frames of MNIST, 40~90-shift frames gurantees the code functions well
        x_source_seq = x_source_seq[:, 0:90-shift].to(self.device)
        y_source_seq = y_source_seq[:, 0:90-shift].to(self.device)
        # target domain: 50~90 frames of USPS, shift~50 frames are regarded as the history
        x_target_seq = x_target_seq[:, shift:90].to(self.device) 
        y_target_seq = y_target_seq[:, shift:90].to(self.device)
        
        # concatenate data
        x_seq = torch.cat([x_source_seq, x_target_seq], dim=0)
        y_seq = torch.cat([y_source_seq, y_target_seq], dim=0)

        mask_list = [1] * source_batch + [0] * target_batch         # incovenience of fetching data from source and target domain 
        domain_mask = torch.IntTensor(mask_list)
        return x_seq, y_seq, domain_mask
    
    def forward(self):
        # self.x_seq: (batch, frame, 1, 28, 28)
        # self.y_seq: (batch, frame)
        # batch, frame = self.y_seq.shape
        if self.seq_first:
            self.hidden = None
        # encoder and classifier
        self.f_seq, self.e_seq, hidden = self.netE(self.x_seq, self.hidden)  
        # e_seq: (batch, frame, nz)
        # f_seq: (batch, frame, nc)

        self.hidden = hidden.detach()  #(hidden[0].detach(), hidden[1].detach())  # hidden.detach() 
        
        # prediction
        self.g_seq = torch.argmax(self.f_seq.detach(), dim=-1)      # (batch, frame)
    
    def backward_G(self):
        raise NotImplementedError

    def backward_D(self):
        raise NotImplementedError
    
    def optimize_parameters(self):
        raise NotImplementedError

    def learn(self, epoch, source_dataloader, target_dataloader):
        self.train()

        loss_curve = {
            loss: []
            for loss in self.loss_names
        }

        acc_source = []
        acc_target = []
        steps = 0
        
        for (x_source_seq, y_source_seq), (x_target_seq, y_target_seq) in zip(source_dataloader, target_dataloader):
            # set input
            x_seq, y_seq, domain_mask = self.set_train_input(x_source_seq, y_source_seq, x_target_seq, y_target_seq)
            self.domain_mask = domain_mask

            # split sequence
            x_seq_splits = torch.split(x_seq, self.n_frame, dim=1)
            y_seq_splits = torch.split(y_seq, self.n_frame, dim=1)
            
            g_seq = []
            # forward
            for i in range(len(x_seq_splits)):
                # set batch index
                if  i == 0:
                    self.seq_first = True
                else:
                    self.seq_first = False

                # single split
                self.x_seq = x_seq_splits[i]
                self.y_seq = y_seq_splits[i]
                
                # optimize parameters
                self.optimize_parameters()

                # append predictions
                g_seq.append(self.g_seq)

                # record loss
                for loss in self.loss_names:
                    loss_curve[loss].append(getattr(self, 'loss_' + loss).detach().item())
                
                steps += 1
            
            # concatenate prediction
            g_seq = torch.cat(g_seq, dim=1)

            # append accuracy
            acc_source.append(g_seq[self.domain_mask == 1].eq(y_seq[self.domain_mask == 1]).to(torch.float))
            acc_target.append(g_seq[self.domain_mask == 0].eq(y_seq[self.domain_mask == 0]).to(torch.float))

            if steps % 40 == 0:
                loss_msg = '[Train][epoch: {}, steps: {}] Loss:'.format(epoch, steps)
                for loss in self.loss_names:
                    loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))
                print(loss_msg)
                with open(self.train_log, 'a') as f:
                    f.write(loss_msg + "\n")
        
        # compute accuracy for source and target
        acc_source = to_np(torch.cat(acc_source, dim=0)).mean()
        acc_target = to_np(torch.cat(acc_target, dim=0)).mean()

        acc_msg = '[Train][{}] Accuracy: total average in source domain {:.1f}, in target domain {:.1f}'.format(
            epoch, acc_source * 100, acc_target * 100)

        loss_msg = '[Train][{}] Loss:'.format(epoch)
        for loss in self.loss_names:
            loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))

        with open(self.train_log, 'a') as f:
            f.write(loss_msg + "\n" + acc_msg + "\n")
        
        # update learning rate
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
        
        # print
        print(loss_msg)
        print(acc_msg)
            
    def test(self, epoch, source_dataloader, target_dataloader, is_eval, shift=0):
        if not is_eval:
            self.load()
        self.eval()
        
        acc_source = []
        acc_target = []

        for (x_source_seq, y_source_seq), (x_target_seq, y_target_seq) in zip(source_dataloader, target_dataloader):
            # set input
            x_seq, y_seq, domain_mask = self.set_test_input(x_source_seq, y_source_seq, x_target_seq, y_target_seq, shift)
            self.domain_mask = domain_mask

            # split sequence
            x_seq_splits = torch.split(x_seq, self.n_frame, dim=1)
            y_seq_splits = torch.split(y_seq, self.n_frame, dim=1)
            
            g_seq = []
            # forward
            for i in range(len(x_seq_splits)):
                # set batch index
                if  i == 0:
                    self.seq_first = True
                else:
                    self.seq_first = False

                # single split
                self.x_seq = x_seq_splits[i]
                self.y_seq = y_seq_splits[i]
                
                # optimize parameters
                self.forward()

                # append predictions
                g_seq.append(self.g_seq)

            # concatenate prediction
            g_seq = torch.cat(g_seq, dim=1)

            # append accuracy
            acc_source.append(g_seq[self.domain_mask == 1, 0:40].eq(y_seq[self.domain_mask == 1, 0:40]).to(torch.float))
            acc_target.append(g_seq[self.domain_mask == 0, -40:].eq(y_seq[self.domain_mask == 0, -40:]).to(torch.float))
        
        # compute accuracy for source and target
        acc_source = to_np(torch.cat(acc_source, dim=0)).mean()
        acc_target = to_np(torch.cat(acc_target, dim=0)).mean()

        if is_eval:
            acc_msg = '[Valid][{}] Accuracy: total average in source domain {:.1f}, in target domain {:.1f}'.format(
            epoch, acc_source * 100, acc_target * 100)
        else:
            acc_msg = '[Test] Accuracy: total average in source domain {:.1f}, in target domain {:.1f}'.format(
            acc_source * 100, acc_target * 100)
        
        with open(self.valid_log, 'a') as f:
            f.write(acc_msg + "\n")
        
        # print
        print(acc_msg)
        return acc_target * 100

# ======================================================================================================================
class SO(BaseModel):
    """
    Source Only Model
    """
    def __init__(self, opt):
        super(SO, self).__init__(opt)
        self.netE = LeNetRNN(opt)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr_gen, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G]
        self.loss_names = ['E_pred']
        self.nc = opt.nc

    def backward_G(self):
        f_seq_source = self.f_seq[self.domain_mask == 1] # f_seq_source: (batch, frame, nc)
        y_seq_source = self.y_seq[self.domain_mask == 1]
        self.loss_E_pred = F.nll_loss(f_seq_source.view(-1, self.nc), y_seq_source.view(-1))
        self.loss_E = self.loss_E_pred
        self.loss_E.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

# ======================================================================================================================
class TO(BaseModel):
    """
    Target Only Model
    """
    def __init__(self, opt):
        super(TO, self).__init__(opt)
        self.netE = LeNetRNN(opt)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr_gen, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G]
        self.loss_names = ['E_pred']
        self.nc = opt.nc

    def backward_G(self):
        f_seq_target = self.f_seq[self.domain_mask == 0]   # f_seq_target: (batch, frame, nc)
        y_seq_target = self.y_seq[self.domain_mask == 0]
        self.loss_E_pred = F.nll_loss(f_seq_target.view(-1, self.nc), y_seq_target.view(-1))
        self.loss_E = self.loss_E_pred
        self.loss_E.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
# ======================================================================================================================
