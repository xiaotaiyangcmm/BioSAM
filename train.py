import torch
import argparse
import numpy as np
import os,sys,csv
os.chdir('/data/caimiaomiao/BISSEG/BISSG/BBBC/code')
sys.path.append(os.getcwd())
from solver import Solver
from model.unet2d_residual import ResidualUNet2D
from model.Unet_EGNN import unet_egnn
from utils.utils import log_args
from data.dataset import BBBC
from torch.utils.data import DataLoader
from utils.logger import Log
import os
import torch
import torch.distributed as dist 
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0,help='node rank for distributed training')
parser.add_argument("-b", "--batch_size",type=int,default=1)
parser.add_argument("-g", "--gpu_nums",type=int,default=1)
parser.add_argument("-e", "--epochs",type=int,default=200)
parser.add_argument("-r", "--lr",type=float,default=1e-3)
parser.add_argument("-p", "--lr_patience",type=int,default=30)
parser.add_argument("-n", "--network",type=str,default="unet_egnn(3,[16,32,64,128,256],3,args)")
parser.add_argument("-t", "--loss_type",type=str,default="BCE_loss")
parser.add_argument("-d", "--data_dir",type=str,default="/data/caimiaomiao/miccai2024/dataset/BBBC")
parser.add_argument("-l", "--logs_dir",type=str,default="./log")
parser.add_argument("-c", "--ckps_dir",type=str,default="./ckp")
# parser.add_argument("-s", "--resample",type=tuple,default=(1, 0.25, 0.25),help="resample rate:(z,h,w)")
parser.add_argument("-w", "--weight_rate",type=list,default=[10,1])
parser.add_argument("-x", "--resume",type=bool,default=False)
parser.add_argument("-y", "--resume_path",type=str,default="./ckp/checkpoint-epoch175.pth")
# parser.add_argument("-z", "--tolerate_shape",type=tuple,default=(192, 384, 384))

#spixel

parser.add_argument('--train_img_height', '-t_imgH', default=256,  type=int, help='img height')
parser.add_argument('--train_img_width', '-t_imgW', default=256, type=int, help='img width')
parser.add_argument('--input_img_height', '-v_imgH', default=256,  type=int, help='img height')
parser.add_argument('--input_img_width', '-v_imgW', default=256, type=int, help='img width')

#embedding
parser.add_argument("-a", "--alpha",type=int,default=1)
parser.add_argument("-be", "--beta",type=int,default=1)
parser.add_argument("-ga", "--gama",type=int,default=0.001)
#EGNN

#loss rate
parser.add_argument("-ls", "--loss_spixel",type=int,default=5) # for affinity
parser.add_argument("-le", "--loss_embedding",type=int,default=1)
parser.add_argument("-lb", "--loss_binary",type=int,default=100)
parser.add_argument("-lg", "--loss_gnn",type=int,default=10)
args = parser.parse_args()

SEED = 123

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
log = Log()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    #DDP
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )

    device = torch.device(f'cuda:{args.local_rank}')

    gpus = args.gpu_nums
    model = eval(args.network)

    criterion = args.loss_type
    metric = "dc_score"
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr


    trainset = BBBC(dir=args.data_dir,mode="train",size=args.train_img_height)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    valset = BBBC(dir=args.data_dir,mode="validation",size=args.input_img_height)
    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=False,sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    val_loader = DataLoader(valset,batch_size=batch_size,shuffle=False,sampler=val_sampler)

    logs_dir = args.logs_dir
    patience = args.lr_patience
    checkpoint_dir = args.ckps_dir
    # scale = args.resample
    weight = args.weight_rate
    resume = args.resume
    resume_path = args.resume_path
    # tolerate_shape = args.tolerate_shape

    #embedding
    alpha = args.alpha


    beta = args.beta
    gama = args.gama
    #spixel
    le = args.loss_embedding
    ls = args.loss_spixel
    lb = args.loss_binary

    log_args(args, log)

    solver = Solver(gpus=gpus,model=model,criterion=criterion,metric=metric,batch_size=batch_size,
                    epochs=epochs,lr=lr,trainset=trainset,valset=valset,train_loader=train_loader,
                    val_loader=val_loader,logs_dir=logs_dir,patience=patience,
                    checkpoint_dir=checkpoint_dir,weight=weight, resume=resume,resume_path=resume_path,
                    log=log,args = args)

    solver.train()
    # cleanup()
