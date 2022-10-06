import os
import math
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from progressbar import ProgressBar
from model.net import LipNet
from core.tools.utils import temporal_ce_loss, vote_video_classification_result


# ======================================================================================================================
# define utility functions
def to_np(x):
    return x.detach().cpu().numpy()

def to_tensor(x, device='cuda'):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x

# ======================================================================================================================
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
        self.domain_threshold = opt.domain_threshold

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
    
    def acc_reset_oulu(self):
        self.hit_domain, self.cnt_domain = np.zeros(5), np.zeros(5)
        self.acc_source, self.acc_target = 0, 0
        self.cnt_source, self.cnt_target = 0, 0
        self.hit_source, self.hit_target = 0, 0
        self.total_samples, self.total_hit, self.total_acc = 0, 0, 0.0
        self.top3_samples, self.top3_hit, self.top3_acc = 0, 0, 0.0

    def acc_update_oulu(self):
        Y = to_np(self.y_seq)
        G = to_np(self.g_seq)
        T = to_np(self.domain)
        T = (T * 90).astype(np.int32)
        T[T <= 15] = 0
        T[(T > 15) & (T <= 37.5)] = 1
        T[(T > 37.5) & (T <= 52.5)] = 2
        T[(T > 52.5) & (T <= 75)] = 3
        T[T > 75] = 4
        TOP3 = to_np(self.top3_pred)
        hit = (Y == G).astype(np.float32)

        hit_top3 = np.array([Y[i] in TOP3[i] for i in range(len(Y))]).astype(np.int32)

        self.total_samples += hit.shape[0]
        self.total_hit += hit.sum().astype(np.int)
        self.total_acc = self.total_hit/self.total_samples

        self.top3_samples += hit_top3.shape[0]
        self.top3_hit += hit_top3.sum().astype(np.int)
        self.top3_acc = self.top3_hit/self.top3_samples

        is_s = to_np(self.is_source)

        for i in range(5):
            self.hit_domain[i] += hit[T == i].sum()
            self.cnt_domain[i] += (T == i).sum()
        self.acc_domain = self.hit_domain / (self.cnt_domain + 1e-10)
        self.acc_source, self.acc_target = self.acc_domain[0], self.acc_domain[1:].mean(
        )
        self.acc_domain = np.round(self.acc_domain, decimals=3)
        self.acc_source = np.round(self.acc_source, decimals=3)
        self.acc_target = np.round(self.acc_target, decimals=3)

        self.cnt_source += (is_s == 1).sum()
        self.cnt_target += (is_s == 0).sum()

        self.hit_source += (hit[is_s == 1]).sum()
        self.hit_target += (hit[is_s == 0]).sum()

    def set_input(self, input):
        self.x_split, self.y_seq, self.t_seq, self.domain = input
        self.domain = self.domain[:, 0]
        self.is_source = (self.domain <= self.domain_threshold).to(torch.float)
    
    def forward(self):
        # self.x_seq: (batch, 1, frame, 44, 50)
        # self.y_seq: (batch,)
        if self.seq_first:
            self.hidden = None
        self.f_split, self.e_split, hidden = self.netE(self.x_split, self.hidden)
        self.hidden = (hidden[0].detach(), hidden[1].detach())

    def learn(self, epoch, dataloader):
        self.epoch = epoch
        self.train()

        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        self.acc_reset_oulu()

        bar = ProgressBar()

        for data in bar(dataloader):
            # set input
            x_seq, y_seq, t_seq, u_seq = [to_tensor(_, self.device) for _ in data]
            # x: (batch, frame=40, 44, 50)
            # y: (batch, 1)  t: (batch, 1)  is_source: (batch, 1)
            x_seq = x_seq[:, :, None, :, :].transpose(1, 2).contiguous()  # (batch, 1, frame=40, 44, 50)
            
            # split sequence
            x_splits = torch.split(x_seq, self.n_frame, dim=2)

            f_seq = []

            for i in range(len(x_splits)):
                if i == 0:
                    self.seq_first = True
                else:
                    self.seq_first = False
                # fetch the split
                x_split = x_splits[i]

                self.set_input(input=(x_split, y_seq, t_seq, u_seq))
                self.optimize_parameters()

                for loss in self.loss_names:
                    loss_curve[loss].append(getattr(self, 'loss_' + loss).detach().item())
                
                f_seq.append(self.f_split.detach())

            # compute acc for the whole seq
            f_seq = torch.cat(f_seq, dim=1)
            self.g_seq, self.top3_pred = vote_video_classification_result(f_seq, y_seq)
            self.acc_update_oulu()
        
        # loss message
        self.loss_msg = '[Train][{}] Loss:'.format(epoch+1)
        for loss in self.loss_names:
            self.loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))
        
        # accuracy message
        self.acc_msg = '[Train][{}] Acc: source {:.3f} ({}/{}) target {:.3f} ({}/{}) total_acc {:.3f} ({}/{})'.format(
            epoch+1, self.acc_source, self.hit_source, self.cnt_source,
            self.acc_target, self.hit_target, self.cnt_target,
            self.total_acc, self.total_hit, self.total_samples)
        
        # print log
        print(self.loss_msg)
        print(self.acc_msg)
        
        with open(self.train_log, 'a') as f:
            f.write(self.loss_msg + "\n" + self.acc_msg + "\n")

        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()


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
                # t1 = time.time()
                self.forward()
                # t2 = time.time()
                # t_delta = t2 - t1
                # all_time += t_delta
                # print(f'Inference testing time: {round(t_delta, 5)}')
                
                f_seq.append(self.f_split.detach())
            
            # compute acc for the whole seq
            f_seq = torch.cat(f_seq, dim=1)
            self.g_seq, self.top3_pred = vote_video_classification_result(f_seq, y_seq)
            self.acc_update_oulu()
            
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

# ======================================================================================================================

class SO(BaseModel):
    """
    Source Only
    """
    def __init__(self, opt):
        super(SO, self).__init__(opt)
        self.netE = LipNet(opt)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr_gen) #, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G]
        self.loss_names = ['E_pred']
        self.nc = opt.nc

    def backward_G(self):
        self.loss_E_pred = temporal_ce_loss(self.f_split[self.is_source == 1], self.y_seq[self.is_source == 1])
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
    Target Only
    """
    def __init__(self, opt):
        super(TO, self).__init__(opt)
        self.netE = LipNet(opt)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr_gen) #, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G]
        self.loss_names = ['E_pred']
        self.nc = opt.nc

    def backward_G(self):
        self.loss_E_pred = temporal_ce_loss(self.f_split[self.is_source == 0], self.y_seq[self.is_source == 0])
        self.loss_E = self.loss_E_pred
        self.loss_E.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

# ======================================================================================================================
