import numpy as np
import pandas as pd
from typing import Dict, Tuple
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def get_global_nondominated_indices(
    methods_data_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, list]]:
    """
    Find Non-dominated points over the combined solution sets of all algorithms.
    Returns front_comb and a dictionary containing the non-dominated indices for each algorithm.
    """
    F_comb_list = []

    for alg, F in methods_data_dict.items():
        if F.shape[0] > 0:
            F_comb_list.append(F)

    if not F_comb_list:
        return np.empty(0), {alg: [] for alg in methods_data_dict.keys()}

    F_comb_arr = np.vstack(F_comb_list)
    front_comb = NonDominatedSorting().do(F_comb_arr)[0]

    start_idx = 0
    nd_indices_dict = {}

    for alg, F in methods_data_dict.items():
        l = F.shape[0]
        idx_nd = [
            idx - start_idx for idx in front_comb if start_idx <= idx < start_idx + l
        ]
        nd_indices_dict[alg] = idx_nd
        start_idx += l

    return front_comb, nd_indices_dict


def compute_bins(X):
    """Compute bin edges for discretization (cube root rule)."""
    bins_list = []
    n_samples = X.shape[0]
    n_bins = int(np.ceil(np.log2(n_samples) / 2))
    for i in range(X.shape[1]):
        feature = X[:, i]
        bins = np.linspace(np.min(feature), np.max(feature), n_bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf
        bins_list.append(bins)
    return bins_list

def transform_to_discrete(X, bins_list):
    """Discretize X using provided bins for each feature."""
    X_binned = np.zeros_like(X, dtype=int)
    for i in range(X.shape[1]):
        X_binned[:, i] = np.digitize(X[:, i], bins_list[i]) - 1
    return X_binned

def discretize_data(X, bins):
    """Discretize X using a single set of bins for all features."""
    return np.digitize(X, bins) - 1

def calculate_entropy_discrete(labels):
    # Calculate entropy for discrete labels
    """H(X) = -sum p(x) log p(x)"""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-12))

def calculate_mi_discrete(x, y):
    """I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    h_x = calculate_entropy_discrete(x)
    h_y = calculate_entropy_discrete(y)

    _, x_idx = np.unique(x, return_inverse=True)
    _, y_idx = np.unique(y, return_inverse=True)

    max_y = np.max(y_idx) + 1

    joint_labels = x_idx * max_y + y_idx

    h_xy = calculate_entropy_discrete(joint_labels)
    return max(0, h_x + h_y - h_xy)

def calculate_joint_mi_three_vars(f1, f2, y):
    """Compute I(F1, F2; y) efficiently."""
    _, f1_idx = np.unique(f1, return_inverse=True)
    _, f2_idx = np.unique(f2, return_inverse=True)

    max_f2 = np.max(f2_idx) + 1
    joint_f1f2 = f1_idx * max_f2 + f2_idx

    return calculate_mi_discrete(joint_f1f2, y)

def calculate_rho(X_discrete, y):
    """
    Compute rho_i = I(Fi; y) for each feature.
    X_discrete: (n x p) discretized data
    y: (n x 1) label vector
    """
    y = np.asarray(y).flatten()
    n_features = X_discrete.shape[1]
    rho = np.zeros((n_features, 1))
    for i in range(n_features):
        rho[i, 0] = calculate_mi_discrete(X_discrete[:, i], y)
    return rho

def calculate_Q(X_discrete, y, rho_scores):
    """
    Compute Q matrix for optimization.
    X_discrete: discretized data
    rho_scores: vector from calculate_rho
    """
    n_features = X_discrete.shape[1]
    S = np.zeros((n_features, n_features))

    H = [calculate_entropy_discrete(X_discrete[:, i]) for i in range(n_features)]
    rho_flat = rho_scores.flatten()

    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                S[i, j] = 1.0
                continue

            I_FiFj_y = calculate_joint_mi_three_vars(
                X_discrete[:, i], X_discrete[:, j], y
            )
            I_multi = rho_flat[i] + rho_flat[j] - I_FiFj_y

            denom = H[i] + H[j]
            s_ij = np.maximum(0, I_multi / np.maximum(1e-12, denom)) 
            S[i, j] = S[j, i] = s_ij

    eigenvalues = np.linalg.eigvalsh(S)
    lambda_min = np.min(eigenvalues)

    delta = -np.minimum(0, lambda_min)
    Q = S + delta * np.eye(n_features) + 1e-6

    return Q


class computePoints:
    # Identify nondominated points in the output set of each iteration
    @staticmethod
    def find_nondominated_points(f_opt):
        n = len(f_opt)

        X_E = set(range(n))  

        eliminated = set()
        for i in range(n - 1):
            for j in range(i + 1, n):
                if np.all(f_opt[i] <= f_opt[j]) and np.any(f_opt[i] < f_opt[j]):
                    X_E.discard(j)
                    eliminated.add(j)
                elif np.all(f_opt[j] <= f_opt[i]) and np.any(f_opt[j] < f_opt[i]):
                    X_E.discard(i)
                    eliminated.add(i)
                    break

        nondominated_points = [f_opt[i] for i in X_E]
        return nondominated_points


    # Compute Hypervolume indicator
    @staticmethod
    def compute_hypervolume(f_nondominated, reference_point):
        # reference_point: reference point 

        f_nondominated = np.array(f_nondominated)
        reference_point = np.array(reference_point)
        if f_nondominated.shape[0] == 0:
            return 0.0
        
        # compute hypervolume
        hv = HV(ref_point=reference_point)
        return hv(f_nondominated)

    # Compute Multi-objective Spread (Gamma) indicator
    @staticmethod
    def compute_spread(F_method, Q, R):
        F_method = np.array(F_method)
        N, m = F_method.shape
        
        if N < 2:
            return float('inf')
            
        gamma_vals = []
        
        for j in range(m):
            sorted_fj = np.sort(F_method[:, j])
            fj_all = np.concatenate(([Q[j]], sorted_fj, [R[j]]))
            gamma = np.diff(fj_all)  
            
            gamma_0, gamma_N = gamma[0], gamma[-1]
            gamma_inner = gamma[1:-1] 
            gamma_bar = np.mean(gamma_inner)
            
            sum_abs_diff = np.sum(np.abs(gamma_inner - gamma_bar))
            numerator = gamma_0 + gamma_N + sum_abs_diff
            denominator = gamma_0 + gamma_N + (N - 1) * gamma_bar
            
            if denominator == 0:
                gamma_vals.append(float('inf'))
            else:
                gamma_vals.append(numerator / denominator)
                
        return np.max(gamma_vals)

    # Find local nondominated points for each method
    @staticmethod
    def find_local_nondominated(f_opts, methods):
        local_dict = {}
        for method in methods:
            solutions_f = np.array(f_opts[method])
            local_dict[method] = computePoints.find_nondominated_points(solutions_f)
        return local_dict

    # Find global Pareto front from all local nondominated points
    @staticmethod
    def find_global_nondominated(local_dict):
        all_points = [p for method in local_dict for p in local_dict[method]]
        return computePoints.find_nondominated_points(all_points)

    # Purity, Hypervolume, IGD+
    @staticmethod
    def compute_metrics(f_opts, x_opts, methods):
        metrics = {}
        results = {}

        local_dict = computePoints.find_local_nondominated(f_opts, methods)

        F_global = computePoints.find_global_nondominated(local_dict)

        # Reference point to compute HV and limits Q, R for Spread
        all_local_pts = [p for method in methods for p in local_dict[method]]
        all_pts_arr = np.vstack(all_local_pts) if len(all_local_pts) > 0 else np.array([])
        
        if len(all_pts_arr) > 0:
            reference_point = np.max(all_pts_arr, axis=0) * 1.1
            Q = np.min(all_pts_arr, axis=0)  # Ideal point
            R = np.max(all_pts_arr, axis=0)  # Nadir point
        else:
            reference_point, Q, R = None, None, None

        # Step 3: Compute metrics for each method
        for method in methods:
            solutions_f = np.array(f_opts[method])
            solutions_x = np.array(x_opts[method])
            F_method = np.array(local_dict[method])

            indices = np.array(
                [i for i in range(len(solutions_f)) if any(np.all(solutions_f[i] == q) for q in F_method)],
                dtype=int
            )
            X_method = solutions_x[indices] if len(indices) > 0 else np.array([])

            num_nondominated = len(F_method)
            F_method_in_global = [p for p in F_method if any(np.all(p == q) for q in F_global)]

            purity_metric = len(F_method_in_global) / num_nondominated if num_nondominated > 0 else 0

            hv_metric = computePoints.compute_hypervolume(F_method, reference_point) if reference_point is not None else 0
            if Q is not None and R is not None:
                gamma_metric = computePoints.compute_spread(F_method, Q, R)
            else:
                gamma_metric = float('inf')

            metrics[method] = {
                "Purity": purity_metric,
                "N": num_nondominated,
                "Global_nondominated": len(F_method_in_global),
                "Hypervolume": hv_metric,
                "Spread_Gamma": gamma_metric,
            }

            results[method] = {
                "F_nondominated": F_method,
                "X_nondominated": X_method
            }

        # Save global Pareto front 
        results["_global"] = {
            "F_nondominated": np.array(F_global),
            "X_nondominated": np.array([])
        }

        return metrics, results



