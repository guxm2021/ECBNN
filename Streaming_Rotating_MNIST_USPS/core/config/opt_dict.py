from easydict import EasyDict
import os
import time

def get_opt(model_name):
    # set experiment configs
    opt = EasyDict()
    opt.model = model_name
    # choose run on which device ["cuda", "cpu"]
    opt.device = "cuda"
    # set random seed
    opt.seed = 2333
    
    opt.nz = 256              # size of features
    opt.nc = 10               # number of class
    # training parameteres
    opt.num_epoch = 50
    opt.source_batch = 50
    opt.target_batch = 50
    opt.lr_gen = 5e-5
    opt.lr_dis = 5e-5
    opt.weight_decay = 5e-4
    opt.beta1 = 0.9

    opt.meta_learning = False    # use meta learning framework

    if opt.model == "ECBNN":
        opt.n_step = 10           # number of integration for ode solver
        opt.inf_step = 10         # number of steps for infinite domain invariance 
        opt.inf_scale = 6         # m = -e^{-t^2/scale^2}
        opt.n_particle = 12       # number of paticles: 12
        # define the loss weights
        opt.lambda_D_dis_frame = 1.0
        opt.lambda_G_class = 1.0
        opt.lambda_G_dis_frame = 0.5
        opt.lambda_G_recons = 0.01 
        # infinite domain invariance
        opt.lambda_D_dis_inf = 1.0
        opt.lambda_G_dis_inf = 1.0
        # gradient matching
        opt.lambda_G_kldiv1 = 1.1
        opt.lambda_G_kldiv2 = 1.1
        opt.lambda_G_match = 2e-4

    opt.n_frame = 10  # number of frames in a batch
    opt.lambda_gan = 1.0
    # experiment folder
    opt.exp = 'Streaming_Rotating_MNIST_USPS_' + opt.model
    opt.outf = './dump/' + opt.exp
    os.system('mkdir -p ' + opt.outf)
    print('Training result will be saved in ', opt.outf)

    # dataset info
    opt.dim_domain = 1

    return opt

