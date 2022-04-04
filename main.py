import os
import numpy as np
from scipy.stats import wishart, multivariate_normal
import matplotlib.pyplot as plt
from tool import calc_ari,cmx
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.cluster import adjusted_rand_score as ari
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset
import argparse
from tool import visualize_gmm


parser = argparse.ArgumentParser(description='Symbol emergence based on VAE+GMM Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='B', help='input batch size for training')
parser.add_argument('--vae-iter', type=int, default=50, metavar='V', help='number of VAE iteration')
parser.add_argument('--mh-iter', type=int, default=50, metavar='M', help='number of M-H mgmm iteration')
parser.add_argument('--category', type=int, default=10, metavar='K', help='number of category for GMM module')
parser.add_argument('--mode', type=int, default=-1, metavar='M', help='0:All reject, 1:ALL accept')
parser.add_argument('--debug', type=bool, default=False, metavar='D', help='Debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print("CUDA",args.cuda)
if args.debug is True: args.vae_iter=2; args.mh_iter=2


############################## Making directory ##############################
file_name = "debug"; model_dir = "./model"; dir_name = "./model/"+file_name
graphA_dir = "./model/"+file_name+"/graphA"; graphB_dir = "./model/"+file_name+"/graphB"
pth_dir = "./model/"+file_name+"/pth";npy_dir = "./model/"+file_name+"/npy"
reconA_dir = model_dir+"/"+file_name+"/reconA"; reconB_dir = model_dir+"/"+file_name+"/reconB"
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

############################## Prepareing Dataset #############################
# MNIST
print("Dataset : MNIST")
angle_a = 0 # MNIST's angle for Agent A
angle_b = 45 # MNIST's angle for Agent B
trans_ang1 = transforms.Compose([transforms.RandomRotation(degrees=(angle_a, angle_a)), transforms.ToTensor()])
trans_ang2 = transforms.Compose([transforms.RandomRotation(degrees=(angle_b, angle_b)), transforms.ToTensor()]) 
# Define datasets
trainval_dataset1 = datasets.MNIST('./../data', train=True, transform=trans_ang1, download=True) # Dataset for Agent A
trainval_dataset2 = datasets.MNIST('./../data', train=True, transform=trans_ang2, download=True) # Dataset for Agent B
n_samples = len(trainval_dataset1)
D = int(n_samples * (1/6)) # Total data
subset1_indices1 = list(range(0, D)); subset2_indices1 = list(range(D, n_samples)) 
subset1_indices2 = list(range(0, D)); subset2_indices2 = list(range(D, n_samples)) 
train_dataset1 = Subset(trainval_dataset1, subset1_indices1); val_dataset1 = Subset(trainval_dataset1, subset2_indices1)
train_dataset2 = Subset(trainval_dataset2, subset1_indices1); val_dataset2 = Subset(trainval_dataset2, subset2_indices2)
train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=False) # train_loader for agent A
train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=False) # train_loader for agent B
all_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=D, shuffle=False)
all_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=D, shuffle=False)

print(f"Total data:{D}, Category:{args.category}")
print(f"VAE_iter:{args.vae_iter}, Batch_size:{args.batch_size}")
print(f"MH_iter:{args.mh_iter}, MH_mode:{args.mode}(-1:Com 0:No-com 1:All accept)") 

import cnn_vae_module_mnist

mutual_iteration = 1
mu_d_A = np.zeros((D)); var_d_A = np.zeros((D)) 
mu_d_B = np.zeros((D)); var_d_B = np.zeros((D))
for it in range(mutual_iteration):
    print(f"------------------Mutual learning session {it} begins------------------")
    ############################## Training VAE ##############################
    c_nd_A, label, loss_list = cnn_vae_module_mnist.train(
        iteration=it, # Current iteration
        gmm_mu=torch.from_numpy(mu_d_A), gmm_var=torch.from_numpy(var_d_A), # mu and var estimated by Multimodal-GMM
        epoch=args.vae_iter, 
        train_loader=train_loader1, batch_size=args.batch_size, all_loader=all_loader1,
        model_dir=dir_name, agent="A"
    )
    # VAE module on Agent B
    c_nd_B, label, loss_list = cnn_vae_module_mnist.train(
        iteration=it, # Current iteration
        gmm_mu=torch.from_numpy(mu_d_B), gmm_var=torch.from_numpy(var_d_B), # mu and var estimated by Multimodal-GMM
        epoch=args.vae_iter, 
        train_loader=train_loader2, batch_size=args.batch_size, all_loader=all_loader2,
        model_dir=dir_name, agent="B"
    )

    # Plot latent space
    #cnn_vae_module_mnist.plot_latent(iteration=it, all_loader=all_loader1, model_dir=dir_name, agent="A") # plot latent space of VAE on Agent A
    #cnn_vae_module_mnist.plot_latent(iteration=it, all_loader=all_loader2, model_dir=dir_name, agent="B") # plot latent space of VAE on Agent B

    K = args.category # number of category
    z_truth_n = label # true label
    dim = len(c_nd_A[0]) # number of dimentions of VAE

    ############################## Initializing parameters ##############################
    # Set hyperparameters
    beta = 1.0; m_d_A = np.repeat(0.0, dim); m_d_B = np.repeat(0.0, dim) # Hyperparameters for \mu^A, \mu^B
    w_dd_A = np.identity(dim) * 0.1; w_dd_B = np.identity(dim) * 0.1 # Hyperparameters for \Lambda^A, \Lambda^B
    nu = dim

    # Initializing \mu, \Lambda
    mu_kd_A = np.empty((K, dim)); lambda_kdd_A = np.empty((K, dim, dim))
    mu_kd_B = np.empty((K, dim)); lambda_kdd_B = np.empty((K, dim, dim))
    for k in range(K):
        lambda_kdd_A[k] = wishart.rvs(df=nu, scale=w_dd_A, size=1); lambda_kdd_B[k] = wishart.rvs(df=nu, scale=w_dd_B, size=1)
        mu_kd_A[k] = np.random.multivariate_normal(mean=m_d_A, cov=np.linalg.inv(beta * lambda_kdd_A[k])).flatten()
        mu_kd_B[k] = np.random.multivariate_normal(mean=m_d_B, cov=np.linalg.inv(beta * lambda_kdd_B[k])).flatten()

    # Initializing unsampled \w
    w_dk_A = np.random.multinomial(1, [1/K]*K, size=D); w_dk_B = np.random.multinomial(1, [1/K]*K, size=D)

    # Initializing learning parameters
    beta_hat_k_A = np.zeros(K) ;beta_hat_k_B = np.zeros(K)
    m_hat_kd_A = np.zeros((K, dim)); m_hat_kd_B = np.zeros((K, dim))
    w_hat_kdd_A = np.zeros((K, dim, dim)); w_hat_kdd_B = np.zeros((K, dim, dim))
    nu_hat_k_A = np.zeros(K); nu_hat_k_B = np.zeros(K)
    tmp_eta_nB = np.zeros((K, D)); eta_dkB = np.zeros((D, K))
    tmp_eta_nA = np.zeros((K, D)); eta_dkA = np.zeros((D, K))
    cat_liks_A = np.zeros(D); cat_liks_B = np.zeros(D)
    mu_d_A = np.zeros((D,dim)); var_d_A = np.zeros((D,dim)) 
    mu_d_B = np.zeros((D,dim)); var_d_B = np.zeros((D,dim))

    iteration = args.mh_iter
    ARI_A = np.zeros((iteration)); ARI_B = np.zeros((iteration)); concidence = np.zeros((iteration))
    accept_count_AtoB = np.zeros((iteration)); accept_count_BtoA = np.zeros((iteration)) # Number of acceptation


    ############################## M-H algorithm ##############################
    print(f"M-H algorithm Start({it}): Epoch:{iteration}")
    for i in range(iteration):
        pred_label_A = []; pred_label_B = []
        count_AtoB = count_BtoA = 0 
        """~~~~~~~~~~~~~~~~~~~~~~~~~~~~Speaker:A -> Listener:B~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
        w_dk = np.random.multinomial(1, [1/K]*K, size=D);
        for k in range(K): 
            tmp_eta_nA[k] = np.diag(-0.5 * (c_nd_A - mu_kd_A[k]).dot(lambda_kdd_A[k]).dot((c_nd_A - mu_kd_A[k]).T)).copy() 
            tmp_eta_nA[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_A[k]) + 1e-7)
            eta_dkA[:, k] = np.exp(tmp_eta_nA[k])
        eta_dkA /= np.sum(eta_dkA, axis=1, keepdims=True) 

        for d in range(D): 
            # sampling w^A
            w_dk_A[d] = np.random.multinomial(n=1, pvals=eta_dkA[d], size=1).flatten() 
            
            if args.mode == 0:
                pred_label_A.append(np.argmax(w_dk_A[d]))
            elif args.mode == 1:
                w_dk[d] = w_dk_A[d]
                count_AtoB = count_AtoB + 1
                pred_label_B.append(np.argmax(w_dk[d])) 
            else:
                cat_liks_A[d] = multivariate_normal.pdf(c_nd_B[d], 
                                mean=mu_kd_B[np.argmax(w_dk_A[d])], 
                                cov=np.linalg.inv(lambda_kdd_B[np.argmax(w_dk_A[d])]),
                                )
                cat_liks_B[d] = multivariate_normal.pdf(c_nd_B[d], 
                                mean=mu_kd_B[np.argmax(w_dk_B[d])], 
                                cov=np.linalg.inv(lambda_kdd_B[np.argmax(w_dk_B[d])]),
                                )
                judge_r = cat_liks_A[d] / cat_liks_B[d]
                judge_r = min(1, judge_r) 
                rand_u = np.random.rand() 
                if judge_r >= rand_u:
                    w_dk[d] = w_dk_A[d]
                    count_AtoB = count_AtoB + 1 
                else: 
                    w_dk[d] = w_dk_B[d]
                pred_label_B.append(np.argmax(w_dk[d])) 
        if args.mode == -1 or args.mode == 1:
            for k in range(K):
                beta_hat_k_B[k] = np.sum(w_dk[:, k]) + beta; m_hat_kd_B[k] = np.sum(w_dk[:, k] * c_nd_B.T, axis=1)
                m_hat_kd_B[k] += beta * m_d_B; m_hat_kd_B[k] /= beta_hat_k_B[k]
                tmp_w_dd_B = np.dot((w_dk[:, k] * c_nd_B.T), c_nd_B)
                tmp_w_dd_B += beta * np.dot(m_d_B.reshape(dim, 1), m_d_B.reshape(1, dim))
                tmp_w_dd_B -= beta_hat_k_B[k] * np.dot(m_hat_kd_B[k].reshape(dim, 1), m_hat_kd_B[k].reshape(1, dim))
                tmp_w_dd_B += np.linalg.inv(w_dd_B)
                w_hat_kdd_B[k] = np.linalg.inv(tmp_w_dd_B)
                nu_hat_k_B[k] = np.sum(w_dk[:, k]) + nu
                
                # sampling \lambda^B and \mu^B
                lambda_kdd_B[k] = wishart.rvs(size=1, df=nu_hat_k_B[k], scale=w_hat_kdd_B[k])
                mu_kd_B[k] = np.random.multivariate_normal(mean=m_hat_kd_B[k], cov=np.linalg.inv(beta_hat_k_B[k] * lambda_kdd_B[k]), size=1).flatten()
        if args.mode == 0:# No com
            for k in range(K):
                beta_hat_k_A[k] = np.sum(w_dk_A[:, k]) + beta; m_hat_kd_A[k] = np.sum(w_dk_A[:, k] * c_nd_A.T, axis=1)
                m_hat_kd_A[k] += beta * m_d_A; m_hat_kd_A[k] /= beta_hat_k_A[k]
                tmp_w_dd_A = np.dot((w_dk_A[:, k] * c_nd_A.T), c_nd_A)
                tmp_w_dd_A += beta * np.dot(m_d_A.reshape(dim, 1), m_d_A.reshape(1, dim))
                tmp_w_dd_A -= beta_hat_k_A[k] * np.dot(m_hat_kd_A[k].reshape(dim, 1), m_hat_kd_A[k].reshape(1, dim))
                tmp_w_dd_A += np.linalg.inv(w_dd_A)
                w_hat_kdd_A[k] = np.linalg.inv(tmp_w_dd_A)
                nu_hat_k_A[k] = np.sum(w_dk_A[:, k]) + nu
                
                # sampling \lambda^A and \mu^A
                lambda_kdd_A[k] = wishart.rvs(size=1, df=nu_hat_k_A[k], scale=w_hat_kdd_A[k])
                mu_kd_A[k] = np.random.multivariate_normal(mean=m_hat_kd_A[k], cov=np.linalg.inv(beta_hat_k_A[k] * lambda_kdd_A[k]), size=1).flatten()

        """~~~~~~~~~~~~~~~~~~~~~~~~~~~~Speaker:B -> Litener:A~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
        w_dk = np.random.multinomial(1, [1/K]*K, size=D);
        for k in range(K): 
            tmp_eta_nB[k] = np.diag(-0.5 * (c_nd_B - mu_kd_B[k]).dot(lambda_kdd_B[k]).dot((c_nd_B - mu_kd_B[k]).T)).copy() 
            tmp_eta_nB[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_B[k]) + 1e-7)
            eta_dkB[:, k] = np.exp(tmp_eta_nB[k])
        eta_dkB /= np.sum(eta_dkB, axis=1, keepdims=True) 

        for d in range(D):
            # sampling w^B
            w_dk_B[d] = np.random.multinomial(n=1, pvals=eta_dkB[d], size=1).flatten() 
            
            if args.mode == 0:
                pred_label_B.append(np.argmax(w_dk_B[d])) 
            elif args.mode == 1:
                w_dk[d] = w_dk_B[d]
                count_BtoA = count_BtoA + 1 
                pred_label_A.append(np.argmax(w_dk[d])) 
            else:
                cat_liks_B[d] = multivariate_normal.pdf(c_nd_A[d], 
                                mean=mu_kd_A[np.argmax(w_dk_B[d])], 
                                cov=np.linalg.inv(lambda_kdd_A[np.argmax(w_dk_B[d])]),
                                )
                cat_liks_A[d] = multivariate_normal.pdf(c_nd_A[d], 
                                mean=mu_kd_A[np.argmax(w_dk_A[d])], 
                                cov=np.linalg.inv(lambda_kdd_A[np.argmax(w_dk_A[d])]),
                                )
                judge_r = cat_liks_B[d] / cat_liks_A[d] 
                judge_r = min(1, judge_r) 
                rand_u = np.random.rand() 
                if judge_r >= rand_u: 
                    w_dk[d] = w_dk_B[d]
                    count_BtoA = count_BtoA + 1 
                else: 
                    w_dk[d] = w_dk_A[d]
                pred_label_A.append(np.argmax(w_dk[d])) 
        
        if args.mode == -1 or args.mode == 1:
            for k in range(K):
                beta_hat_k_A[k] = np.sum(w_dk[:, k]) + beta; m_hat_kd_A[k] = np.sum(w_dk[:, k] * c_nd_A.T, axis=1)
                m_hat_kd_A[k] += beta * m_d_A; m_hat_kd_A[k] /= beta_hat_k_A[k]
                tmp_w_dd_A = np.dot((w_dk[:, k] * c_nd_A.T), c_nd_A)
                tmp_w_dd_A += beta * np.dot(m_d_A.reshape(dim, 1), m_d_A.reshape(1, dim))
                tmp_w_dd_A -= beta_hat_k_A[k] * np.dot(m_hat_kd_A[k].reshape(dim, 1), m_hat_kd_A[k].reshape(1, dim))
                tmp_w_dd_A += np.linalg.inv(w_dd_A)
                w_hat_kdd_A[k] = np.linalg.inv(tmp_w_dd_A)
                nu_hat_k_A[k] = np.sum(w_dk[:, k]) + nu
                
                # sampling \lambda^A and \mu^A
                lambda_kdd_A[k] = wishart.rvs(size=1, df=nu_hat_k_A[k], scale=w_hat_kdd_A[k])
                mu_kd_A[k] = np.random.multivariate_normal(mean=m_hat_kd_A[k], cov=np.linalg.inv(beta_hat_k_A[k] * lambda_kdd_A[k]), size=1).flatten()

        if args.mode == 0:# No com
            for k in range(K):
                beta_hat_k_B[k] = np.sum(w_dk_B[:, k]) + beta; m_hat_kd_B[k] = np.sum(w_dk_B[:, k] * c_nd_B.T, axis=1)
                m_hat_kd_B[k] += beta * m_d_B; m_hat_kd_B[k] /= beta_hat_k_B[k]
                tmp_w_dd_B = np.dot((w_dk_B[:, k] * c_nd_B.T), c_nd_B)
                tmp_w_dd_B += beta * np.dot(m_d_B.reshape(dim, 1), m_d_B.reshape(1, dim))
                tmp_w_dd_B -= beta_hat_k_B[k] * np.dot(m_hat_kd_B[k].reshape(dim, 1), m_hat_kd_B[k].reshape(1, dim))
                tmp_w_dd_B += np.linalg.inv(w_dd_B)
                w_hat_kdd_B[k] = np.linalg.inv(tmp_w_dd_B)
                nu_hat_k_B[k] = np.sum(w_dk_B[:, k]) + nu
                
                # sampling \lambda^A and \mu^A
                lambda_kdd_B[k] = wishart.rvs(size=1, df=nu_hat_k_B[k], scale=w_hat_kdd_B[k])
                mu_kd_B[k] = np.random.multivariate_normal(mean=m_hat_kd_B[k], cov=np.linalg.inv(beta_hat_k_B[k] * lambda_kdd_B[k]), size=1).flatten()

        ############################## Evaluation ##############################
        _, result_a = calc_ari(pred_label_A, z_truth_n)
        _, result_b = calc_ari(pred_label_B, z_truth_n)
        # Kappa conncidence
        concidence[i] = np.round(cohen_kappa_score(pred_label_A,pred_label_B),3)
        # ARI
        ARI_A[i] = np.round(ari(z_truth_n, result_a),3); ARI_B[i] = np.round(ari(z_truth_n,result_b),3)
        # Number of acceptance 
        accept_count_AtoB[i] = count_AtoB; accept_count_BtoA[i] = count_BtoA
        
        if i == 0 or (i+1) % 10 == 0 or i == (iteration-1): 
            print(f"=> Epoch: {i+1}, ARI_A: {ARI_A[i]}, ARI_B: {ARI_B[i]}, Kappa:{concidence[i]}, A2B:{int(accept_count_AtoB[i])}, B2A:{int(accept_count_BtoA[i])}")
        for d in range(D):
            mu_d_A[d] = mu_kd_A[np.argmax(w_dk[d])]
            var_d_A[d] = np.diag(np.linalg.inv(lambda_kdd_A[np.argmax(w_dk[d])]))
            mu_d_B[d] = mu_kd_B[np.argmax(w_dk[d])]
            var_d_B[d] = np.diag(np.linalg.inv(lambda_kdd_B[np.argmax(w_dk[d])]))

    np.save(npy_dir+'/muA_'+str(it)+'.npy', mu_kd_A); np.save(npy_dir+'/muB_'+str(it)+'.npy', mu_kd_B)
    np.save(npy_dir+'/lambdaA_'+str(it)+'.npy', lambda_kdd_A); np.save(npy_dir+'/lambdaB_'+str(it)+'.npy', lambda_kdd_B)    
    np.savetxt(log_dir+"/ariA"+str(it)+".txt", ARI_B, fmt ='%.3f'); np.savetxt(log_dir+"/ariB"+str(it)+".txt", ARI_B, fmt ='%.2f'); np.savetxt(log_dir+"/cappa"+str(it)+".txt", concidence, fmt ='%.2f')

    ############################## Plot ##############################
    # acceptance
    plt.figure()
    #plt.ylim(0,)
    plt.plot(range(0,iteration), accept_count_AtoB, marker="None", label="Accept_num:AtoB")
    plt.plot(range(0,iteration), accept_count_BtoA, marker="None", label="Accept_num:BtoA")
    plt.xlabel('iteration');plt.ylabel('Number of acceptation')
    plt.ylim(0,D)
    plt.legend()
    plt.savefig(result_dir+'/accept'+str(it)+'.png')
    #plt.show()
    plt.close()

    
    # concidence
    plt.figure()
    plt.plot(range(0,iteration), concidence, marker="None")
    plt.xlabel('iteration'); plt.ylabel('Concidence')
    plt.ylim(0,1)
    plt.title('k')
    plt.savefig(result_dir+"/conf"+str(it)+".png")
    #plt.show()
    plt.close()

    # ARI
    plt.figure()
    plt.plot(range(0,iteration), ARI_A, marker="None",label="ARI_A")
    plt.plot(range(0,iteration), ARI_B, marker="None",label="ARI_B")
    plt.xlabel('iteration'); plt.ylabel('ARI')
    plt.ylim(0,1)
    plt.legend()
    plt.title('ARI')
    plt.savefig(result_dir+"/ari"+str(it)+".png")
    #plt.show()
    plt.close()

    cmx(iteration=it, y_true=z_truth_n, y_pred=result_a, agent="A", save_dir=result_dir)
    cmx(iteration=it, y_true=z_truth_n, y_pred=result_b, agent="B", save_dir=result_dir)
    print(f"Iteration:{it} Done:max_ARI_A: {max(ARI_A)}, max_ARI_B: {max(ARI_B)}, max_Kappa:{max(concidence)}")
