from easydict import EasyDict
import os

def get_opt(model_name, seed, lr, lambda_gan, match=1e-5, inf=0.1):
    # set experiment configs
    opt = EasyDict()
    opt.model = model_name
    # choose run on which device ["cuda", "cpu"]
    opt.device = "cuda"
    # set random seed
    opt.seed = 1
    # network architecture
    opt.dnn_shapes = [2000, 1000, 500, 50]
    opt.nonlinearities = ["rectify", "rectify", "rectify", "linear"]
    opt.pic_size = 2200   # (44, 50)
    opt.window = 3
    opt.nc = 10
    opt.nz = 450
    opt.cnn_channels = [32, 64, 96]
    opt.cnn_output_size = 864

    # training parameteres
    opt.num_epoch = 100
    opt.batch_size = 100
    # lr = 2e-4
    opt.lr_gen = lr
    opt.lr_dis = lr
    opt.lr = lr
    opt.weight_decay = 5e-4
    opt.beta1 = 0.9
    
    opt.meta_learning = False # use meta learning framework

    if opt.model == "ECBNN":
        opt.bnn_dim = 9
        opt.theta_dim = 81
        opt.nz = 450
        opt.n_step = 10     # number of integration for ode solver
        opt.inf_step = 10   # number of steps for infinite domain invariance 
        opt.inf_scale = 6   # m = -e^{-t/scale}
        # define the sampling
        opt.n_particle = 12       # number of paticles: 12
        # define the loss weights
        opt.lambda_D_dis_frame = 1.0
        opt.lambda_G_class = 1.0
        opt.lambda_G_dis_frame = 1.0
        opt.lambda_G_recons = 0.01
        # infinite domain invariance
        opt.lambda_D_dis_inf = 0.05
        opt.lambda_G_dis_inf = 0.05

        opt.lambda_G_kldiv1 = 1.1
        opt.lambda_G_kldiv2 = 1.1
        opt.lambda_G_match = 0.05

    opt.n_frame = 10 # number of frames in a batch
    opt.lambda_gan = lambda_gan
    
    # experimental folder
    opt.exp = '_LipReading_' + opt.model
    opt.outf = './dump_Sept/' + 'seed' + str(opt.seed) + '/' + opt.exp
    os.system('mkdir -p ' + opt.outf)
    print('Traning result will be saved in ', opt.outf)

    # dataset info
    opt.dim_domain = 1
    opt.domain_threshold = 0.1

    return opt

