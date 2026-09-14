import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import computePoints as sfs_utils
from computePoints import get_global_nondominated_indices, computePoints

from plots import plot_start_end_point

from sfs_algorithms import AMPG1_sfs, AMPG2_sfs, PGM_sfs, accPGM_sfs, MPGE_sfs

def run_file(
    filename: str,
    methods: List[str] = None,
    num_initial_points: int = 250,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[int, Dict, Dict]:
    

    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    n_features = X.shape[1]

    # Split data: Train (80%), Val (20%)
    X_train_raw, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=seed, shuffle=True
    )

    # Standardize data using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    # Calculate Q and rho
    bins_list = sfs_utils.compute_bins(X_train_scaled)
    X_train_discretized = sfs_utils.transform_to_discrete(X_train_scaled, bins_list)

    rho = sfs_utils.calculate_rho(X_train_discretized, y_train).flatten()
    Q = sfs_utils.calculate_Q(X_train_discretized, y_train, rho)
    
    # Create initial points using Dirichlet distribution
    np.random.seed(seed)
    alpha_dirichlet = 1.0 / n_features
    W_starts = np.random.dirichlet([alpha_dirichlet] * n_features, size=num_initial_points)

    f_starts = np.zeros((num_initial_points, 2))
    for i, w in enumerate(W_starts):
        f_starts[i, 0] = (w.T @ Q @ w).item() if hasattr(w.T @ Q @ w, "item") else w.T @ Q @ w
        f_starts[i, 1] = (-rho.T @ w).item() if hasattr(-rho.T @ w, "item") else -rho.T @ w

    all_f_opts_padded = {}
    algorithms_res = {}

    for method in methods:
        t0 = time.time()
        
        w_list, F_list = [], []
        total_iterations = 0
        num_success = 0
        total_stepsize = 0.0

        f_opts_padded = np.full((num_initial_points, 2), np.inf)

        if method in ["AMPG1", "AMPG2", "MPGE", "PGM", "accPGM"]:
            for start_idx, w_start in enumerate(W_starts):
                if method == "AMPG1":
                    x_opt, F_opt, info, iter_val, _, _, _, step_val = AMPG1_sfs(n_features, 2, w_start, Q, rho)
                elif method == "AMPG2":
                    x_opt, F_opt, info, iter_val, _, _, _, step_val = AMPG2_sfs(n_features, 2, w_start, Q, rho)
                elif method == "MPGE":
                    x_opt, F_opt, info, iter_val, _, _, _, step_val = MPGE_sfs(n_features, 2, w_start, Q, rho)
                elif method == "PGM":
                    x_opt, F_opt, info, iter_val, _, _, _, step_val = PGM_sfs(n_features, 2, w_start, Q, rho)
                elif method == "accPGM":
                    x_opt, F_opt, info, iter_val, _, _, _, step_val = accPGM_sfs(n_features, 2, w_start, Q, rho)
                
                if info in (0, 1):
                    w_list.append(x_opt)
                    F_list.append(F_opt)
                    f_opts_padded[start_idx] = F_opt
                    num_success += 1
                    total_stepsize += step_val
                
                total_iterations += iter_val
                
        all_f_opts_padded[method] = f_opts_padded
        
        algorithms_res[method] = {
            "w": np.array(w_list),
            "F": np.array(F_list),
            "time": time.time() - t0,
            "iterations": total_iterations / num_initial_points,
            "success": num_success,
            "stepsize": total_stepsize / num_initial_points,
        }

    final_results = {}
    for name, res in algorithms_res.items():
        w_list = res["w"]
        F_list = res["F"]

        best_value = 0.0
        obj_value = (
            [[w_list[i], F_list[i, 0], F_list[i, 1]] for i in range(len(F_list))]
            if len(F_list) > 0
            else []
        )

        final_results[name] = {
            "obj_value": obj_value,
            "best_value": best_value,
            "time": res["time"],
            "iterations": res["iterations"],
            "success": res["success"],
            "stepsize": res["stepsize"],
        }

    return n_features, final_results, f_starts, all_f_opts_padded


def run_all_files(
    path: str,
    methods: List[str] = None,
    num_initial_points: int = 250,
    seed: int = 42,
    verbose: bool = False,
):

    all_obj_values = {}
    all_best_values = {}
    all_times = {}
    all_iterations = {}
    all_success = {}
    all_stepsize = {}
    all_f_starts_global = {}
    all_f_opts_padded_global = {}

    path_obj = Path(path)

    for file_path in path_obj.iterdir():
        if file_path.suffix in [".data", ".csv"]:
            filename = file_path.name
            print(f"\nProcessing file: {filename}...")

            original_features, final_results, f_starts, all_f_opts_padded = run_file(
                str(file_path),
                methods=methods,
                num_initial_points=num_initial_points,
                seed=seed,
                verbose=verbose,
            )

            all_obj_values[filename] = {}
            all_best_values[filename] = {}
            all_times[filename] = {}
            all_iterations[filename] = {}
            all_success[filename] = {}
            all_stepsize[filename] = {}
            all_f_starts_global[filename] = f_starts
            all_f_opts_padded_global[filename] = all_f_opts_padded

            for alg_name in methods:
                metrics_data = final_results[alg_name]
                all_obj_values[filename][alg_name] = metrics_data["obj_value"]
                all_best_values[filename][alg_name] = metrics_data["best_value"]
                all_times[filename][alg_name] = metrics_data["time"]
                all_iterations[filename][alg_name] = metrics_data["iterations"]
                all_success[filename][alg_name] = metrics_data["success"]
                all_stepsize[filename][alg_name] = metrics_data["stepsize"]

            # Tính toán và in metrics cho dataset hiện tại
            f_opts_current = {}
            x_opts_current = {}
            for method in methods:
                pts = final_results[method]["obj_value"]
                if len(pts) > 0:
                    x_opts_current[method] = [p[0] for p in pts]
                    f_opts_current[method] = [[p[1], p[2]] for p in pts]
                else:
                    x_opts_current[method] = []
                    f_opts_current[method] = []
            
            cmp_metrics_current, _ = computePoints.compute_metrics(f_opts_current, x_opts_current, methods)
            
            data_metrics_current = []
            for method in methods:
                res = final_results[method]
                iters = res["iterations"]
                time_ms = res["time"] * 1000
                pts = num_initial_points
                avg_time_ms = time_ms / pts if pts > 0 else 0.0
                row = {
                    "Method": method,
                    "Iter": iters,
                    "Time(ms)": time_ms,
                    "Time/Iter": avg_time_ms / iters if iters > 0 else 0.0,
                    "Success": res["success"],
                    "Stepsize": res["stepsize"],
                    "Global_nondom": cmp_metrics_current.get(method, {}).get("Global_nondominated", 0),
                    "HV": cmp_metrics_current.get(method, {}).get("Hypervolume", float("nan")),
                    "Purity": cmp_metrics_current.get(method, {}).get("Purity", float("nan")),
                    "Spread_Delta": cmp_metrics_current.get(method, {}).get("Spread_Delta", float("nan")),
                }
                data_metrics_current.append(row)
            
            df_metrics_current = pd.DataFrame(data_metrics_current)
            print(f"\n{'=' * 100}")
            print(f"RESULTS FOR DATASET: {filename}")
            print(f"{'=' * 100}")
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1200):
                print(df_metrics_current.to_string(index=False))
            print(f"{'=' * 100}\n")

    return (
        all_obj_values,
        all_best_values,
        all_times,
        all_iterations,
        all_success,
        all_stepsize,
        all_f_starts_global,
        all_f_opts_padded_global,
    )


if __name__ == "__main__":
    print("Running....")
    # Path to the benchmark datasets and the methods to run
    PATH = "data"

    methods_to_run = [
        "AMPG1",
        "AMPG2",
        "MPGE",
        "PGM",  
        "accPGM"
    ]

    # Number of initial points to generate for the optimization algorithms
    num_initial_points = 100
    verbose = True

    (
        all_obj_values,
        all_best_values,
        all_times,
        all_iterations,
        all_success,
        all_stepsize,
        all_f_starts_global,
        all_f_opts_padded_global,
    ) = run_all_files(
        PATH,
        methods=methods_to_run,
        num_initial_points=num_initial_points,
        verbose=verbose,
    )


    # Create output directory for results
    alpha, beta, c0, c1 = AMPG1_sfs.__defaults__[-4:] 
    out_dir = Path(f"Result_sfs_benchmark_alpha={alpha}_beta={beta}_c0={c0}_c1={c1}")
    out_dir.mkdir(parents=True, exist_ok=True)

    
    metrics_results = {}
    for dataset_name, alg_data in all_obj_values.items():
        metrics_results[dataset_name] = {}
        f_opts = {}
        x_opts = {}
        for method in methods_to_run:
            pts = alg_data.get(method, [])
            if len(pts) > 0:
                x_opts[method] = [p[0] for p in pts]
                f_opts[method] = [[p[1], p[2]] for p in pts]
            else:
                x_opts[method] = []
                f_opts[method] = []
        
        cmp_metrics, _ = computePoints.compute_metrics(f_opts, x_opts, methods_to_run)
        
        for method in methods_to_run:
            metrics_results[dataset_name][f"Iterations ({method})"] = all_iterations.get(dataset_name, {}).get(method, 0)
            metrics_results[dataset_name][f"Time ({method})"] = all_times.get(dataset_name, {}).get(method, 0.0)
            metrics_results[dataset_name][f"Success ({method})"] = all_success.get(dataset_name, {}).get(method, 0)
            metrics_results[dataset_name][f"Stepsize ({method})"] = all_stepsize.get(dataset_name, {}).get(method, 0.0)
            metrics_results[dataset_name][f"Global Nondom ({method})"] = cmp_metrics.get(method, {}).get("Global_nondominated", 0)
            metrics_results[dataset_name][f"HV ({method})"] = cmp_metrics.get(method, {}).get("Hypervolume", float("nan"))
            metrics_results[dataset_name][f"Purity ({method})"] = cmp_metrics.get(method, {}).get("Purity", float("nan"))
            metrics_results[dataset_name][f"Spread_Delta ({method})"] = cmp_metrics.get(method, {}).get("Spread_Delta", float("nan"))

    # 2. Prepare Metrics DataFrame
    data_metrics = []
    for dataset, mets in metrics_results.items():
        for method in methods_to_run:
            if f"HV ({method})" not in mets and f"Time ({method})" not in mets:
                continue
                
            iters = mets.get(f"Iterations ({method})", 0)
            time_ms = mets.get(f"Time ({method})", 0.0) * 1000
            pts = num_initial_points
            avg_time_ms = time_ms / pts if pts > 0 else 0.0
            row = {
                "Dataset": dataset,
                "Method": method,
                "Iter": iters,
                "Time(ms)": time_ms,
                "Time/Iter": avg_time_ms / iters if iters > 0 else 0.0,
                "Success": mets.get(f"Success ({method})", None),
                "Stepsize": mets.get(f"Stepsize ({method})", None),
                "Global_nondominated": mets.get(f"Global Nondom ({method})", 0),
                "Hypervolume": mets.get(f"HV ({method})", float("nan")),
                "Purity": mets.get(f"Purity ({method})", float("nan")),
                "Spread_Delta": mets.get(f"Spread_Delta ({method})", float("nan")),
            }
            data_metrics.append(row)

    df_metrics = pd.DataFrame(data_metrics)
    
    print("\n" + "=" * 100)
    print("SUMMARY OF PERFORMANCE METRICS (Iterations, Time, Time/Iter, Success, Stepsize, HV, Spread_Delta, Global Non-Dominated)")
    print("=" * 100)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1200):
        print(df_metrics.to_string(index=False))
    print("=" * 100)

    # 3. Export to Excel file
    output_file = out_dir / "Results.xlsx"
    with pd.ExcelWriter(str(output_file), engine='openpyxl') as writer:
        if not df_metrics.empty:
            df_metrics.to_excel(writer, sheet_name='Results', index=False)
            
    print(f"\nSuccessfully exported results to Excel file: {output_file}")

    # Plot Trajectory for all methods and datasets using plots.py
    sol_dist_dir = out_dir
    sol_dist_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, method_data in all_obj_values.items():
        dataset_name = Path(filename).stem
        
        methods_F_dict = {}
        for method, pts in method_data.items():
            F_pts = np.array([[p[1], p[2]] for p in pts]) if pts else np.empty((0, 2))
            methods_F_dict[method] = F_pts

        plot_start_end_point(
            problem_name=dataset_name,
            m=2,
            methods={m: None for m in methods_F_dict.keys()},
            f_starts=all_f_starts_global[filename],
            f_opts=all_f_opts_padded_global[filename],
            save_dir=str(sol_dist_dir)
        )
