import os
import numpy as np
from scipy.stats import wishart 
import matplotlib.pyplot as plt
from tool import calc_ari, visualize_gmm
from sklearn.metrics import cohen_kappa_score
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset
import argparse
#from custom_data import CustomDataset
from PIL import Image

parser = argparse.ArgumentParser(description='Symbol emergence based on VAE+GMM Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='B', help='input batch size for training')
parser.add_argument('--vae-iter', type=int, default=100, metavar='V', help='number of VAE iteration')
parser.add_argument('--mh-iter', type=int, default=100, metavar='M', help='number of M-H mgmm iteration')
parser.add_argument('--category', type=int, default=28, metavar='K', help='number of category for GMM module')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

############################## Making directory ##############################

file_name = "debug"; model_dir = "./model"; dir_name = "./model/"+file_name# debugフォルダに保存される
graphA_dir = "./model/"+file_name+"/graphA"; graphB_dir = "./model/"+file_name+"/graphB" # 各種グラフの保存先
pth_dir = "./model/"+file_name+"/pth";npy_dir = "./model/"+file_name+"/npy"
reconA_dir = model_dir+"/"+file_name+"/reconA/graph_dist"; reconB_dir = model_dir+"/"+file_name+"/reconB/graph_dist"
log_dir = model_dir+"/"+file_name+"/log"; result_dir = model_dir+"/"+file_name+"/result"
if not os.path.exists(model_dir):   os.mkdir(model_dir)
if not os.path.exists(dir_name):    os.mkdir(dir_name)
if not os.path.exists(pth_dir):    os.mkdir(pth_dir)
if not os.path.exists(graphA_dir):   os.mkdir(graphA_dir)
if not os.path.exists(graphB_dir):   os.mkdir(graphB_dir)
if not os.path.exists(npy_dir):    os.mkdir(npy_dir)
if not os.path.exists(reconA_dir):    os.mkdir(reconA_dir)
if not os.path.exists(reconB_dir):    os.mkdir(reconB_dir)
if not os.path.exists(log_dir):    os.mkdir(log_dir)
if not os.path.exists(result_dir):    os.mkdir(result_dir)


############################## Prepareing Dataset ##############################

# MNIST左右回転設定
angle_a = 0 # 回転角度
angle_b = 45 # 回転角度
trans_ang1 = transforms.Compose([transforms.RandomRotation(degrees=(-angle_a,-angle_a)), transforms.ToTensor()]) # -angle度回転設定
trans_ang2 = transforms.Compose([transforms.RandomRotation(degrees=(angle_b,angle_b)), transforms.ToTensor()]) # angle度回転設定
# データセット定義
trainval_dataset1 = datasets.MNIST('./../data', train=True, transform=trans_ang1, download=False) # Agent A用 MNIST
trainval_dataset2 = datasets.MNIST('./../data', train=True, transform=trans_ang2, download=False) # Agent B用 MNIST
n_samples = len(trainval_dataset1)
D = int(n_samples * (1/6)) # データ総数
subset1_indices1 = list(range(0, D)); subset2_indices1 = list(range(D, n_samples)) 
subset1_indices2 = list(range(0, D)); subset2_indices2 = list(range(D, n_samples)) 
train_dataset1 = Subset(trainval_dataset1, subset1_indices1); val_dataset1 = Subset(trainval_dataset1, subset2_indices1)
train_dataset2 = Subset(trainval_dataset2, subset1_indices1); val_dataset2 = Subset(trainval_dataset2, subset2_indices2)
train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=False) # train_loader for agent A
train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=False) # train_loader for agent B
all_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=D, shuffle=False) # データセット総数分のローダ
all_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=D, shuffle=False) # データセット総数分のローダ

import cnn_vae_module_mnist

def get_concat_h_multi_resize(dir_name, agent, resample=Image.BICUBIC):
    im0 = Image.open(dir_name+'/recon'+agent+'/manual_0.png');im1 = Image.open(dir_name+'/recon'+agent+'/manual_1.png')
    im2 = Image.open(dir_name+'/recon'+agent+'/manual_2.png');im3 = Image.open(dir_name+'/recon'+agent+'/manual_3.png')
    im4 = Image.open(dir_name+'/recon'+agent+'/manual_4.png');im5 = Image.open(dir_name+'/recon'+agent+'/manual_5.png')
    im6 = Image.open(dir_name+'/recon'+agent+'/manual_6.png');im7 = Image.open(dir_name+'/recon'+agent+'/manual_7.png')
    im8 = Image.open(dir_name+'/recon'+agent+'/manual_8.png');im9 = Image.open(dir_name+'/recon'+agent+'/manual_9.png')
    
    im_list = [im0, im1, im2, im3, im4, im5, im6, im7, im8, im9]
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    dst.save(dir_name+'/recon'+agent+'/concat.png')

def decode_from_mgmm(load_iteration, sigma, K, decode_k, sample_num, manual, dir_name):
    for i in range(K):
        sample_d = visualize_gmm(iteration=load_iteration, # load iteration model 
                                sigma=sigma,
                                K=K, 
                                decode_k=i, 
                                sample_num=sample_num, 
                                manual=manual, 
                                model_dir=dir_name, agent="A")
        cnn_vae_module_mnist.decode(iteration=load_iteration, decode_k=i, sample_num=sample_num, 
                          sample_d=sample_d, manual=manual, model_dir=dir_name, agent="A")
        sample_d = visualize_gmm(iteration=load_iteration, # load iteration model 
                                sigma=sigma,
                                K=K, 
                                decode_k=i, 
                                sample_num=sample_num, 
                                manual=manual, 
                                model_dir=dir_name, agent="B")
        cnn_vae_module_mnist.decode(iteration=load_iteration, decode_k=i, sample_num=sample_num, 
                          sample_d=sample_d, manual=manual, model_dir=dir_name, agent="B")

def main():
    load_iteration = 0
    decode_from_mgmm(load_iteration=load_iteration, sigma=0, K=10, decode_k=None, sample_num=1, manual=True, dir_name=dir_name)
    get_concat_h_multi_resize(dir_name = dir_name, agent="A"); get_concat_h_multi_resize(dir_name = dir_name, agent="B")

    #cnn_vae_module_mnist.plot_latent(iteration=load_iteration, all_loader=all_loader1, model_dir=dir_name, agent="A") # plot latent space of VAE on Agent A
    #cnn_vae_module_mnist.plot_latent(iteration=load_iteration, all_loader=all_loader2, model_dir=dir_name, agent="B") # plot latent space of VAE on Agent B

if __name__=="__main__":
    main()
