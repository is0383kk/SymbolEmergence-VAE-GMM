import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from random import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix



def sample(iteration, z_dim, mu_gmm, lambda_gmm, sigma, sample_num, sample_k, model_dir="./vae_gmm"):
    """
    iteration:サンプルするモデルのイテレーション
    z_dim:VAEの潜在変数の次元数（＝GMMの観測の次元数）
    mu_gmm:GMMの推定した平均パラメータ
    lambda_gmm:GMMの推定した精度行列パラメータ
    sample_num:サンプル数
    sample_k:サンプルする際のK
    """
    sigma_kdd = sigma * np.identity(z_dim, dtype=float) # 対角成分がsigmaでそれ以外は0の分散共分散行列
    # サンプリングデータを生成
    #x_nd = np.random.multivariate_normal(mean=mu_gmm[k], cov=lambda_gmm[k], size=sample_num)
    manual_sample = np.random.multivariate_normal(mean=mu_gmm[sample_k], cov=sigma_kdd, size=sample_num)
    random_sample = np.random.multivariate_normal(mean=mu_gmm[sample_k], cov=0.2*np.linalg.inv(lambda_gmm[sample_k]), size=sample_num)

    return manual_sample, random_sample

def visualize_gmm(iteration, sigma, K, decode_k, sample_num, manual, model_dir, agent):
    mu_gmm_kd, lambda_gmm_kdd, pi_gmm_k = get_param(iteration=iteration, model_dir=model_dir, agent=agent)
    sample_d, random_sample = sample(iteration=iteration, z_dim=12, 
                                     mu_gmm=mu_gmm_kd, lambda_gmm=lambda_gmm_kdd, sigma=sigma, 
                                     sample_k=decode_k, sample_num=sample_num, model_dir=model_dir
                                    )
    if manual != True: sample_d = random_sample 

    mu_gmm2d_kd = np.zeros((K,2)) # mu 2次元化
    lambda_gmm2d_kdd = np.zeros((K,2,2)) # lambda 2次元化
    for k in range(K):
        mu_gmm2d_kd[k] = mu_gmm_kd[k][:2]
        for dim1 in range(2):
            for dim2 in range(2):
                lambda_gmm2d_kdd[k][dim1][dim2] = lambda_gmm_kdd[k][dim1][dim2]
    # 作図用のx軸のxの値を作成
    x_1_line = np.linspace(
    np.min(mu_gmm_kd[:, 0] - 0.5 * np.sqrt(lambda_gmm_kdd[:, 0, 0])), 
    np.max(mu_gmm_kd[:, 0] + 0.5 * np.sqrt(lambda_gmm_kdd[:, 0, 0])), 
    num=900
    )
    # 作図用のy軸のxの値を作成
    x_2_line = np.linspace(
    np.min(mu_gmm_kd[:, 1] - 0.5 * np.sqrt(lambda_gmm_kdd[:, 1, 1])), 
    np.max(mu_gmm_kd[:, 1] + 0.5 * np.sqrt(lambda_gmm_kdd[:, 1, 1])), 
    num=900
    )
    x_1_grid, x_2_grid = np.meshgrid(x_1_line, x_2_line);x_point = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
    x_dim = x_1_grid.shape

    res_density_k = 0
    tmp_density_k = multivariate_normal.pdf(x=x_point, mean=mu_gmm_kd[decode_k][:2], cov=np.linalg.inv(lambda_gmm2d_kdd[decode_k]))
    res_density_k += tmp_density_k * pi_gmm_k[0]

    annotations_x = [str(n) for n in list(range(sample_num))]
    annotations_k = ["K=0","K=1","K=2","K=3","K=4","K=5","K=6","K=7","K=8","K=9"]

    plt.figure(figsize=(12, 9))
    plt.scatter(x=sample_d[:, 0], y=sample_d[:, 1], label='cluster:' + str(k + 1)) # 観測データ
    plt.scatter(x=mu_gmm2d_kd[:, 0], y=mu_gmm2d_kd[:, 1], color='red', s=100, marker='x') # 事後分布の平均
    plt.contour(x_1_grid, x_2_grid, res_density_k.reshape(x_dim), alpha=0.5, linestyles='dashed') # K=0の等高線
    #plt.contour(x_1_grid, x_2_grid, true_model.reshape(x_dim), linestyles='--') # 真の分布
    plt.suptitle('Gaussian Mixture Model', fontsize=20)
    #if manual != True:
    for i, label in enumerate(annotations_x[:K]):
        plt.annotate(label, (sample_d[i][0], sample_d[i][1]))
    for i, label in enumerate(annotations_k[:K]):
        if i != decode_k: plt.annotate(label, (mu_gmm2d_kd[i][0], mu_gmm2d_kd[i][1]),color="r")
    plt.title('Number of sample='+str(len(sample_d))+', K='+str(decode_k))
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.colorbar()
    #if manual != True:
    #    plt.savefig(model_dir+'/recon'+agent+'/graph_dist/rgausek_'+str(decode_k)+'.png')
    #else:
    #    plt.savefig(model_dir+'/recon'+agent+'/graph_dist/mgausek_'+str(decode_k)+'.png')
    plt.close()

    return sample_d

def get_param(iteration, model_dir, agent):
    mu_gmm_kd = np.load(model_dir+"/npy/mu"+agent+"_"+str(iteration)+".npy") # load mean param from .npy
    lambda_gmm_kdd = np.load(model_dir+"/npy/lambda"+agent+"_"+str(iteration)+".npy") # load lambda param from .npy
    #pi_gmm_k = np.load(model_dir+"/pi_"+str(iteration)+".npy") # GMMの混合比
    pi_gmm_k = np.full(10,0.1)

    return mu_gmm_kd, lambda_gmm_kdd, pi_gmm_k

def visualize_ls(iteration, z, labels, save_dir, agent):
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']
    #colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf','silver','rosybrown','tan','gold','indigo']
    #colors = ["red", "green", "blue", "orange", "purple"]
    points_pca = PCA(n_components=2, random_state=0).fit_transform(z)
    points_tsne = TSNE(n_components=2, random_state=0).fit_transform(z)
    
    # TSNE
    plt.figure(figsize=(10,10))
    for p, l in zip(points_tsne, labels):
        plt.title("TSNE", fontsize=24)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l),c=colors[l],s=100)
    plt.savefig(save_dir+'/graph'+agent+'/tsne_'+agent+str(iteration)+'.png')
    plt.close()

    # PCA
    plt.figure(figsize=(10,10))
    for p, l in zip(points_pca, labels):
        plt.title("PCA", fontsize=24)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l),c=colors[l],s=100)
    plt.savefig(save_dir+'/graph'+agent+'/pca_'+agent+str(iteration)+'.png')
    plt.close()

def cmx(iteration, y_true, y_pred, agent, save_dir):
    labels = sorted(list(set(y_true)))
    cmx = confusion_matrix(y_true, y_pred, labels=labels)
    #cmd = ConfusionMatrixDisplay(cmx,display_labels=None)
    #cmd.plot()
    df_cmx = pd.DataFrame(cmx, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, annot=False)
    plt.savefig(save_dir+"/cm_"+agent+str(iteration)+".png")
    #plt.show()

def calc_ari( results, correct ):
    K = np.max(results)+1     # Number of category
    D = len(results)          # Number of data points
    max_ari = 0               # Max ari
    changed = True
    while changed:
        changed = False
        for i in range(K):
            for j in range(K):
                tmp_result = np.zeros( D )

                for n in range(D):
                    if results[n]==i: tmp_result[n]=j
                    elif results[n]==j: tmp_result[n]=i
                    else: tmp_result[n] = results[n]

                # Caluculate ari
                ari = (tmp_result==correct).sum()/float(D)

                if ari > max_ari:
                    max_ari = ari
                    results = tmp_result
                    changed = True

    return max_ari, results


def softmax(arr, axis=0):
    arr = np.rollaxis(arr, axis)
    vmax = arr.max(axis=0)
    out = np.exp(arr-vmax) / np.sum(np.exp(arr-vmax), axis=0)

    return out