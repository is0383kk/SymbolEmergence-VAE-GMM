import argparse
import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal, wishart
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.cluster import adjusted_rand_score as ari
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

import cnn_vae_module_mnist
from tool import calc_ari, cmx


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Symbol emergence based on VAE+GMM Example"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="B",
        help="input batch size for training",
    )
    parser.add_argument(
        "--vae-iter", type=int, default=50, metavar="V", help="number of VAE iteration"
    )
    parser.add_argument(
        "--mh-iter",
        type=int,
        default=50,
        metavar="M",
        help="number of M-H mgmm iteration",
    )
    parser.add_argument(
        "--category",
        type=int,
        default=10,
        metavar="K",
        help="number of category for GMM module",
    )
    parser.add_argument(
        "--mode", type=int, default=-1, metavar="M", help="0:All reject, 1:ALL accept"
    )
    parser.add_argument(
        "--debug", type=bool, default=False, metavar="D", help="Debug mode"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=2, metavar="S", help="random seed")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.debug:
        args.vae_iter = 2
        args.mh_iter = 2

    return args


def setup_directories(file_name: str = "debug") -> Dict[str, str]:
    """Create necessary directories for saving results."""
    model_dir = "./model"
    dir_name = f"./model/{file_name}"

    directories = {
        "model_dir": model_dir,
        "dir_name": dir_name,
        "graphA_dir": f"{dir_name}/graphA",
        "graphB_dir": f"{dir_name}/graphB",
        "pth_dir": f"{dir_name}/pth",
        "npy_dir": f"{dir_name}/npy",
        "reconA_dir": f"{dir_name}/reconA",
        "reconB_dir": f"{dir_name}/reconB",
        "log_dir": f"{dir_name}/log",
        "result_dir": f"{dir_name}/result",
    }

    for dir_path in directories.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    return directories


def prepare_datasets(batch_size: int) -> Tuple[Any, Any, Any, Any, int, np.ndarray]:
    """Prepare MNIST datasets for both agents with different rotation angles."""
    print("Dataset : MNIST")

    # Different rotation angles for each agent
    angle_a = 0  # MNIST's angle for Agent A
    angle_b = 45  # MNIST's angle for Agent B

    trans_ang1 = transforms.Compose(
        [transforms.RandomRotation(degrees=(angle_a, angle_a)), transforms.ToTensor()]
    )
    trans_ang2 = transforms.Compose(
        [transforms.RandomRotation(degrees=(angle_b, angle_b)), transforms.ToTensor()]
    )

    # Define datasets
    trainval_dataset1 = datasets.MNIST(
        "./../data", train=True, transform=trans_ang1, download=True
    )  # Dataset for Agent A
    trainval_dataset2 = datasets.MNIST(
        "./../data", train=True, transform=trans_ang2, download=True
    )  # Dataset for Agent B

    n_samples = len(trainval_dataset1)
    D = int(n_samples * (1 / 6))  # Total data

    subset1_indices1 = list(range(0, D))
    subset2_indices1 = list(range(D, n_samples))
    subset1_indices2 = list(range(0, D))
    subset2_indices2 = list(range(D, n_samples))

    train_dataset1 = Subset(trainval_dataset1, subset1_indices1)
    val_dataset1 = Subset(trainval_dataset1, subset2_indices1)
    train_dataset2 = Subset(trainval_dataset2, subset1_indices1)
    val_dataset2 = Subset(trainval_dataset2, subset2_indices2)

    train_loader1 = torch.utils.data.DataLoader(
        train_dataset1, batch_size=batch_size, shuffle=False
    )  # train_loader for agent A
    train_loader2 = torch.utils.data.DataLoader(
        train_dataset2, batch_size=batch_size, shuffle=False
    )  # train_loader for agent B
    all_loader1 = torch.utils.data.DataLoader(
        train_dataset1, batch_size=D, shuffle=False
    )
    all_loader2 = torch.utils.data.DataLoader(
        train_dataset2, batch_size=D, shuffle=False
    )

    return train_loader1, train_loader2, all_loader1, all_loader2, D, subset1_indices1


def train_vae_agents(
    args: argparse.Namespace,
    train_loader1,
    train_loader2,
    all_loader1,
    all_loader2,
    mu_d_A: np.ndarray,
    var_d_A: np.ndarray,
    mu_d_B: np.ndarray,
    var_d_B: np.ndarray,
    dir_name: str,
    iteration: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train VAE for both agents."""
    print(
        f"------------------Mutual learning session {iteration} begins------------------"
    )

    # Train VAE module on Agent A
    c_nd_A, label, loss_list = cnn_vae_module_mnist.train(
        iteration=iteration,
        gmm_mu=torch.from_numpy(mu_d_A),
        gmm_var=torch.from_numpy(var_d_A),
        epoch=args.vae_iter,
        train_loader=train_loader1,
        batch_size=args.batch_size,
        all_loader=all_loader1,
        model_dir=dir_name,
        agent="A",
    )

    # Train VAE module on Agent B
    c_nd_B, label, loss_list = cnn_vae_module_mnist.train(
        iteration=iteration,
        gmm_mu=torch.from_numpy(mu_d_B),
        gmm_var=torch.from_numpy(var_d_B),
        epoch=args.vae_iter,
        train_loader=train_loader2,
        batch_size=args.batch_size,
        all_loader=all_loader2,
        model_dir=dir_name,
        agent="B",
    )

    return c_nd_A, c_nd_B, label


def initialize_gmm_parameters(K: int, dim: int, D: int) -> Dict[str, np.ndarray]:
    """Initialize GMM parameters for both agents."""
    # Set hyperparameters
    beta = 1.0
    m_d_A = np.repeat(0.0, dim)
    m_d_B = np.repeat(0.0, dim)
    w_dd_A = np.identity(dim) * 0.1
    w_dd_B = np.identity(dim) * 0.1
    nu = dim

    # Initialize μ, Λ
    mu_kd_A = np.empty((K, dim))
    lambda_kdd_A = np.empty((K, dim, dim))
    mu_kd_B = np.empty((K, dim))
    lambda_kdd_B = np.empty((K, dim, dim))

    for k in range(K):
        lambda_kdd_A[k] = wishart.rvs(df=nu, scale=w_dd_A, size=1)
        lambda_kdd_B[k] = wishart.rvs(df=nu, scale=w_dd_B, size=1)
        mu_kd_A[k] = np.random.multivariate_normal(
            mean=m_d_A, cov=np.linalg.inv(beta * lambda_kdd_A[k])
        ).flatten()
        mu_kd_B[k] = np.random.multivariate_normal(
            mean=m_d_B, cov=np.linalg.inv(beta * lambda_kdd_B[k])
        ).flatten()

    # Initialize unsampled w
    w_dk_A = np.random.multinomial(1, [1 / K] * K, size=D)
    w_dk_B = np.random.multinomial(1, [1 / K] * K, size=D)

    return {
        "beta": beta,
        "m_d_A": m_d_A,
        "m_d_B": m_d_B,
        "w_dd_A": w_dd_A,
        "w_dd_B": w_dd_B,
        "nu": nu,
        "mu_kd_A": mu_kd_A,
        "lambda_kdd_A": lambda_kdd_A,
        "mu_kd_B": mu_kd_B,
        "lambda_kdd_B": lambda_kdd_B,
        "w_dk_A": w_dk_A,
        "w_dk_B": w_dk_B,
    }


def compute_eta_matrix(
    c_nd: np.ndarray, mu_kd: np.ndarray, lambda_kdd: np.ndarray, K: int, D: int
) -> np.ndarray:
    """Compute eta matrix for probability calculations."""
    tmp_eta_n = np.zeros((K, D))
    eta_dk = np.zeros((D, K))

    for k in range(K):
        tmp_eta_n[k] = np.diag(
            -0.5 * (c_nd - mu_kd[k]).dot(lambda_kdd[k]).dot((c_nd - mu_kd[k]).T)
        ).copy()
        tmp_eta_n[k] += 0.5 * np.log(np.linalg.det(lambda_kdd[k]) + 1e-7)
        eta_dk[:, k] = np.exp(tmp_eta_n[k])

    eta_dk /= np.sum(eta_dk, axis=1, keepdims=True)
    return eta_dk


def metropolis_hastings_step(
    speaker_data: np.ndarray,
    listener_data: np.ndarray,
    speaker_mu: np.ndarray,
    speaker_lambda: np.ndarray,
    listener_mu: np.ndarray,
    listener_lambda: np.ndarray,
    w_dk_speaker: np.ndarray,
    w_dk_listener: np.ndarray,
    args: argparse.Namespace,
    K: int,
    D: int,
) -> Tuple[np.ndarray, int, list]:
    """Perform one step of Metropolis-Hastings algorithm."""
    w_dk = np.random.multinomial(1, [1 / K] * K, size=D)
    count_accept = 0
    pred_labels = []

    # Compute eta matrix for speaker
    eta_dk_speaker = compute_eta_matrix(speaker_data, speaker_mu, speaker_lambda, K, D)

    cat_liks_speaker = np.zeros(D)
    cat_liks_listener = np.zeros(D)

    for d in range(D):
        # Sample w for speaker
        w_dk_speaker[d] = np.random.multinomial(
            n=1, pvals=eta_dk_speaker[d], size=1
        ).flatten()

        if args.mode == 0:  # No communication
            pred_labels.append(np.argmax(w_dk_speaker[d]))
        elif args.mode == 1:  # All accept
            w_dk[d] = w_dk_speaker[d]
            count_accept += 1
            pred_labels.append(np.argmax(w_dk[d]))
        else:  # Normal Metropolis-Hastings
            cat_liks_speaker[d] = multivariate_normal.pdf(
                listener_data[d],
                mean=listener_mu[np.argmax(w_dk_speaker[d])],
                cov=np.linalg.inv(listener_lambda[np.argmax(w_dk_speaker[d])]),
            )
            cat_liks_listener[d] = multivariate_normal.pdf(
                listener_data[d],
                mean=listener_mu[np.argmax(w_dk_listener[d])],
                cov=np.linalg.inv(listener_lambda[np.argmax(w_dk_listener[d])]),
            )
            judge_r = cat_liks_speaker[d] / cat_liks_listener[d]
            judge_r = min(1, judge_r)
            rand_u = np.random.rand()

            if judge_r >= rand_u:
                w_dk[d] = w_dk_speaker[d]
                count_accept += 1
            else:
                w_dk[d] = w_dk_listener[d]
            pred_labels.append(np.argmax(w_dk[d]))

    return w_dk, count_accept, pred_labels


def update_gmm_parameters(
    w_dk: np.ndarray,
    c_nd: np.ndarray,
    params: Dict[str, Any],
    agent_suffix: str,
    K: int,
    dim: int,
) -> None:
    """Update GMM parameters for one agent."""
    beta = params["beta"]
    m_d = params[f"m_d_{agent_suffix}"]
    w_dd = params[f"w_dd_{agent_suffix}"]
    nu = params["nu"]
    mu_kd = params[f"mu_kd_{agent_suffix}"]
    lambda_kdd = params[f"lambda_kdd_{agent_suffix}"]

    beta_hat_k = np.zeros(K)
    m_hat_kd = np.zeros((K, dim))
    w_hat_kdd = np.zeros((K, dim, dim))
    nu_hat_k = np.zeros(K)

    for k in range(K):
        beta_hat_k[k] = np.sum(w_dk[:, k]) + beta
        m_hat_kd[k] = np.sum(w_dk[:, k] * c_nd.T, axis=1)
        m_hat_kd[k] += beta * m_d
        m_hat_kd[k] /= beta_hat_k[k]

        tmp_w_dd = np.dot((w_dk[:, k] * c_nd.T), c_nd)
        tmp_w_dd += beta * np.dot(m_d.reshape(dim, 1), m_d.reshape(1, dim))
        tmp_w_dd -= beta_hat_k[k] * np.dot(
            m_hat_kd[k].reshape(dim, 1), m_hat_kd[k].reshape(1, dim)
        )
        tmp_w_dd += np.linalg.inv(w_dd)
        w_hat_kdd[k] = np.linalg.inv(tmp_w_dd)
        nu_hat_k[k] = np.sum(w_dk[:, k]) + nu

        # Sample λ and μ
        lambda_kdd[k] = wishart.rvs(size=1, df=nu_hat_k[k], scale=w_hat_kdd[k])
        mu_kd[k] = np.random.multivariate_normal(
            mean=m_hat_kd[k],
            cov=np.linalg.inv(beta_hat_k[k] * lambda_kdd[k]),
            size=1,
        ).flatten()


def run_metropolis_hastings(
    args: argparse.Namespace,
    c_nd_A: np.ndarray,
    c_nd_B: np.ndarray,
    params: Dict[str, Any],
    K: int,
    D: int,
    dim: int,
    z_truth_n: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the complete Metropolis-Hastings algorithm."""
    iteration = args.mh_iter
    ARI_A = np.zeros(iteration)
    ARI_B = np.zeros(iteration)
    concidence = np.zeros(iteration)
    accept_count_AtoB = np.zeros(iteration)
    accept_count_BtoA = np.zeros(iteration)

    print(f"M-H algorithm Start: Epoch:{iteration}")

    for i in range(iteration):
        pred_label_A = []
        pred_label_B = []

        # Speaker: A -> Listener: B
        w_dk, count_AtoB, pred_label_B_temp = metropolis_hastings_step(
            c_nd_A,
            c_nd_B,
            params["mu_kd_A"],
            params["lambda_kdd_A"],
            params["mu_kd_B"],
            params["lambda_kdd_B"],
            params["w_dk_A"],
            params["w_dk_B"],
            args,
            K,
            D,
        )

        if args.mode != 0:
            pred_label_B.extend(pred_label_B_temp)

        # Update parameters for listener B
        if args.mode == -1 or args.mode == 1:
            update_gmm_parameters(w_dk, c_nd_B, params, "B", K, dim)

        if args.mode == 0:  # No communication mode - update speaker A
            update_gmm_parameters(params["w_dk_A"], c_nd_A, params, "A", K, dim)

        # Speaker: B -> Listener: A
        w_dk, count_BtoA, pred_label_A_temp = metropolis_hastings_step(
            c_nd_B,
            c_nd_A,
            params["mu_kd_B"],
            params["lambda_kdd_B"],
            params["mu_kd_A"],
            params["lambda_kdd_A"],
            params["w_dk_B"],
            params["w_dk_A"],
            args,
            K,
            D,
        )

        if args.mode != 0:
            pred_label_A.extend(pred_label_A_temp)

        # Update parameters for listener A
        if args.mode == -1 or args.mode == 1:
            update_gmm_parameters(w_dk, c_nd_A, params, "A", K, dim)

        if args.mode == 0:  # No communication mode - update speaker B
            update_gmm_parameters(params["w_dk_B"], c_nd_B, params, "B", K, dim)

        # Fill pred_label_A and pred_label_B for mode 0
        if args.mode == 0:
            for d in range(D):
                pred_label_A.append(np.argmax(params["w_dk_A"][d]))
                pred_label_B.append(np.argmax(params["w_dk_B"][d]))

        # Evaluation
        _, result_a = calc_ari(pred_label_A, z_truth_n)
        _, result_b = calc_ari(pred_label_B, z_truth_n)

        concidence[i] = np.round(cohen_kappa_score(pred_label_A, pred_label_B), 3)
        ARI_A[i] = np.round(ari(z_truth_n, result_a), 3)
        ARI_B[i] = np.round(ari(z_truth_n, result_b), 3)
        accept_count_AtoB[i] = count_AtoB
        accept_count_BtoA[i] = count_BtoA

        if i == 0 or (i + 1) % 10 == 0 or i == (iteration - 1):
            print(
                f"=> Epoch: {i+1}, ARI_A: {ARI_A[i]}, ARI_B: {ARI_B[i]}, "
                f"Kappa:{concidence[i]}, A2B:{int(accept_count_AtoB[i])}, "
                f"B2A:{int(accept_count_BtoA[i])}"
            )

        # Update mu_d and var_d for next iteration
        mu_d_A = np.zeros((D, dim))
        var_d_A = np.zeros((D, dim))
        mu_d_B = np.zeros((D, dim))
        var_d_B = np.zeros((D, dim))

        for d in range(D):
            mu_d_A[d] = params["mu_kd_A"][np.argmax(w_dk[d])]
            var_d_A[d] = np.diag(
                np.linalg.inv(params["lambda_kdd_A"][np.argmax(w_dk[d])])
            )
            mu_d_B[d] = params["mu_kd_B"][np.argmax(w_dk[d])]
            var_d_B[d] = np.diag(
                np.linalg.inv(params["lambda_kdd_B"][np.argmax(w_dk[d])])
            )

    return ARI_A, ARI_B, concidence, accept_count_AtoB, accept_count_BtoA


def save_results(
    params: Dict[str, Any],
    ARI_A: np.ndarray,
    ARI_B: np.ndarray,
    concidence: np.ndarray,
    directories: Dict[str, str],
    iteration: int,
) -> None:
    """Save model parameters and results."""
    npy_dir = directories["npy_dir"]
    log_dir = directories["log_dir"]

    np.save(f"{npy_dir}/muA_{iteration}.npy", params["mu_kd_A"])
    np.save(f"{npy_dir}/muB_{iteration}.npy", params["mu_kd_B"])
    np.save(f"{npy_dir}/lambdaA_{iteration}.npy", params["lambda_kdd_A"])
    np.save(f"{npy_dir}/lambdaB_{iteration}.npy", params["lambda_kdd_B"])
    np.savetxt(f"{log_dir}/ariA{iteration}.txt", ARI_A, fmt="%.3f")
    np.savetxt(f"{log_dir}/ariB{iteration}.txt", ARI_B, fmt="%.2f")
    np.savetxt(f"{log_dir}/cappa{iteration}.txt", concidence, fmt="%.2f")


def create_plots(
    ARI_A: np.ndarray,
    ARI_B: np.ndarray,
    concidence: np.ndarray,
    accept_count_AtoB: np.ndarray,
    accept_count_BtoA: np.ndarray,
    directories: Dict[str, str],
    iteration_num: int,
    total_iterations: int,
    D: int,
) -> None:
    """Create evaluation plots."""
    result_dir = directories["result_dir"]

    # Acceptance plot
    plt.figure()
    plt.plot(
        range(0, total_iterations),
        accept_count_AtoB,
        marker="None",
        label="Accept_num:AtoB",
    )
    plt.plot(
        range(0, total_iterations),
        accept_count_BtoA,
        marker="None",
        label="Accept_num:BtoA",
    )
    plt.xlabel("iteration")
    plt.ylabel("Number of acceptation")
    plt.ylim(0, D)
    plt.legend()
    plt.savefig(f"{result_dir}/accept{iteration_num}.png")
    plt.close()

    # Concidence plot
    plt.figure()
    plt.plot(range(0, total_iterations), concidence, marker="None")
    plt.xlabel("iteration")
    plt.ylabel("Concidence")
    plt.ylim(0, 1)
    plt.title("k")
    plt.savefig(f"{result_dir}/conf{iteration_num}.png")
    plt.close()

    # ARI plot
    plt.figure()
    plt.plot(range(0, total_iterations), ARI_A, marker="None", label="ARI_A")
    plt.plot(range(0, total_iterations), ARI_B, marker="None", label="ARI_B")
    plt.xlabel("iteration")
    plt.ylabel("ARI")
    plt.ylim(0, 1)
    plt.legend()
    plt.title("ARI")
    plt.savefig(f"{result_dir}/ari{iteration_num}.png")
    plt.close()


def main():
    """Main function orchestrating the entire training process."""
    # Parse arguments and setup
    args = parse_arguments()
    device = torch.device("cuda" if args.cuda else "cpu")
    print("CUDA", args.cuda)

    # Setup directories
    directories = setup_directories()

    # Prepare datasets
    train_loader1, train_loader2, all_loader1, all_loader2, D, _ = prepare_datasets(
        args.batch_size
    )

    print(f"Total data:{D}, Category:{args.category}")
    print(f"VAE_iter:{args.vae_iter}, Batch_size:{args.batch_size}")
    print(f"MH_iter:{args.mh_iter}, MH_mode:{args.mode}(-1:Com 0:No-com 1:All accept)")

    # Initialize variables for mutual learning
    mutual_iteration = 1
    mu_d_A = np.zeros(D)
    var_d_A = np.zeros(D)
    mu_d_B = np.zeros(D)
    var_d_B = np.zeros(D)

    for it in range(mutual_iteration):
        # Train VAE agents
        c_nd_A, c_nd_B, label = train_vae_agents(
            args,
            train_loader1,
            train_loader2,
            all_loader1,
            all_loader2,
            mu_d_A,
            var_d_A,
            mu_d_B,
            var_d_B,
            directories["dir_name"],
            it,
        )

        K = args.category
        z_truth_n = label
        dim = len(c_nd_A[0])

        # Initialize GMM parameters
        params = initialize_gmm_parameters(K, dim, D)

        # Run Metropolis-Hastings algorithm
        (
            ARI_A,
            ARI_B,
            concidence,
            accept_count_AtoB,
            accept_count_BtoA,
        ) = run_metropolis_hastings(args, c_nd_A, c_nd_B, params, K, D, dim, z_truth_n)

        # Save results
        save_results(params, ARI_A, ARI_B, concidence, directories, it)

        # Create plots
        create_plots(
            ARI_A,
            ARI_B,
            concidence,
            accept_count_AtoB,
            accept_count_BtoA,
            directories,
            it,
            args.mh_iter,
            D,
        )

        # Create confusion matrices
        _, result_a = calc_ari(
            [np.argmax(params["w_dk_A"][d]) for d in range(D)], z_truth_n
        )
        _, result_b = calc_ari(
            [np.argmax(params["w_dk_B"][d]) for d in range(D)], z_truth_n
        )
        cmx(
            iteration=it,
            y_true=z_truth_n,
            y_pred=result_a,
            agent="A",
            save_dir=directories["result_dir"],
        )
        cmx(
            iteration=it,
            y_true=z_truth_n,
            y_pred=result_b,
            agent="B",
            save_dir=directories["result_dir"],
        )

        print(
            f"Iteration:{it} Done:max_ARI_A: {max(ARI_A)}, max_ARI_B: {max(ARI_B)}, max_Kappa:{max(concidence)}"
        )


if __name__ == "__main__":
    main()
