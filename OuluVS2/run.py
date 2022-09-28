import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
import time
import argparse
import torch
import random
import numpy as np
from core.config.opt_dict import get_opt
from core.dataset.datasets import OuluLipReading, DataLoader
from model.pool import get_model


def run(train=False):
    # set dataset
    train_dataset = OuluLipReading('./data/unipadding/train.pt', select_angles={0, 30, 45, 60, 90})
    valid_dataset = OuluLipReading('./data/unipadding/valid.pt', select_angles={0, 30, 45, 60, 90})
    test_dataset = OuluLipReading('./data/unipadding/test.pt', select_angles={0, 30, 45, 60, 90})

    train_dataloader = DataLoader(
        dataset = train_dataset,
        shuffle = True,
        batch_size = opt.batch_size,
        num_workers = 4,
    )
    valid_dataloader = DataLoader(
        dataset = valid_dataset,
        shuffle = False,
        batch_size = opt.batch_size,
        num_workers = 4,
    )
    test_dataloader = DataLoader(
        dataset = test_dataset,
        shuffle = False,
        batch_size = opt.batch_size,
        num_workers = 4,
    )

    # build the model
    modelClass = get_model(opt.model)
    model = modelClass(opt)
    model.to(opt.device)

    # train
    t0 = time.time()
    epoch = 0
    if train:
        # Single Step Domain Adaptation
        for epoch in range(opt.num_epoch):
            model.learn(epoch, train_dataloader)
            if (epoch + 1) % 10 == 0:
                acc_target = model.test(epoch, test_dataloader, is_eval=True)
                model.check_save(acc_target)
            t1 = time.time()
            print(f"model {opt.model} spends time {round((t1 - t0) / (epoch + 1), 2)} per seconds")
    
    acc_target = model.test(epoch, test_dataloader, is_eval=False)
    return acc_target
    
if __name__ == "__main__":
    # define the opt
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="NEEDLE", help="the name of model")
    parser.add_argument("--gpu", type=int, default=0, help="the num of gpu, chosen from [0, 1, 2, 3]")
    parser.add_argument("--seed", type=int, default=1, help='random seed for reproducibility')
    parser.add_argument("--lambda_gan", type=float, default=0.2, help='lambda gan hyper-parameter, chosen from [0.2, 0.5, 1.0, 2.0, 5.0]')
    parser.add_argument("--lr", type=float, default=2e-4, help='learning rate')
    parser.add_argument("--match", type=float, default=5e-2, help="gradient matching loss term")
    parser.add_argument("--inf", type=float, default=5e-2, help="infinity domain invariance loss term")
    args = parser.parse_args()
    opt = get_opt(model_name=args.model_name, seed=args.seed, lr=args.lr, lambda_gan=args.lambda_gan, match=args.match, inf=args.inf)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run(train=True)