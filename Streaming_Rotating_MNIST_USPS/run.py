import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'

import time
import random
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import dataset
from core.config.opt_dict import get_opt
from core.dataset.datasets import MovRotateDigits, DataLoader
from model.pool import get_model
import argparse


def run(train=False, shift=50):
    # set dataset
    source_train_dataset = MovRotateDigits(dataset="MNIST", split="train")
    source_valid_dataset = MovRotateDigits(dataset="MNIST", split="valid")
    source_test_dataset = MovRotateDigits(dataset="MNIST", split="test")
    target_train_dataset = MovRotateDigits(dataset="USPS", split="train")
    target_valid_dataset = MovRotateDigits(dataset="USPS", split="valid")
    target_test_dataset = MovRotateDigits(dataset="USPS", split="test")

    # set dataloader
    source_train_dataloader = DataLoader(
        dataset = source_train_dataset,
        shuffle = True,
        batch_size = opt.source_batch,
        num_workers = 4,
        pin_memory=True,
    )

    source_valid_dataloader = DataLoader(
        dataset = source_valid_dataset,
        shuffle = False,
        batch_size = opt.source_batch,
        num_workers = 4,
        pin_memory=True,
    )

    source_test_dataloader = DataLoader(
        dataset = source_test_dataset,
        shuffle = False,
        batch_size = opt.source_batch,
        num_workers = 4,
        pin_memory=True,
    )

    target_train_dataloader = DataLoader(
        dataset = target_train_dataset,
        shuffle = True,
        batch_size = opt.target_batch,
        num_workers = 4,
        pin_memory=True,
    )

    target_valid_dataloader = DataLoader(
        dataset = target_valid_dataset,
        shuffle = False,
        batch_size = opt.target_batch,
        num_workers = 4,
        pin_memory=True,
    )

    target_test_dataloader = DataLoader(
        dataset = target_test_dataset,
        shuffle = False,
        batch_size = opt.target_batch,
        num_workers = 4,
        pin_memory=True,
    )

    # build the model
    modelClass = get_model(opt.model)
    model = modelClass(opt)
    model.to(opt.device)

    if train:
        for epoch in range(opt.num_epoch):
            if opt.meta_learning:
                model.learn_stage1(epoch, source_train_dataloader, target_train_dataloader)
                score_target = model.test(epoch, source_valid_dataloader, target_valid_dataloader, is_eval=True, shift=50)
                model.check_save(score_target)
            else:
                model.learn(epoch, source_train_dataloader, target_train_dataloader)
                score_target = model.test(epoch, source_valid_dataloader, target_valid_dataloader, is_eval=True, shift=50)
                model.check_save(score_target)
        
        if opt.meta_learning:
            # meta-testing
            model.load()
            for epoch in range(opt.num_epoch_test):
                model.learn_stage2(epoch, source_train_dataloader, target_train_dataloader)
                score_target = model.test(epoch, source_valid_dataloader, target_valid_dataloader, is_eval=True, shift=50)
                model.check_save(score_target)

    epoch = opt.num_epoch
    score_target = model.test(epoch, source_test_dataloader, target_test_dataloader, is_eval=False, shift=shift)
    return score_target

if __name__ == "__main__":
    # define the opt
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="SO_RNN", help="the name of model")
    parser.add_argument("--gpu", type=int, default=0, help="the num of gpu, chosen from [0, 1, 2, 3]")
    args = parser.parse_args()
    opt = get_opt(model_name=args.model_name)
    
    seed = opt.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run(train=True, shift=50)