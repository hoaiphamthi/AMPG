import numpy as np

import pandas as pd
from tabulate import tabulate
from inip import inip
from data import data
from computePoints import computePoints
from evalg import evalg
from evalh import evalh
from plots import plot_start_end_point

from robust_algorithms import MPGE, AMPG1, AMPG2, PGM, accPGM


import os

WARMUP_ENABLED = True
WARMUP_RUNS = 3

alpha, beta, c0, c1 = AMPG1.__defaults__[-4:] 
filename = f"Results_robust_problem_alpha={alpha}_beta={beta}_c0={c0}_c1={c1}"
savedir = f"Results_robust_problem_alpha={alpha}_beta={beta}_c0={c0}_c1={c1}"

os.makedirs(savedir, exist_ok=True)

def run_problem(problem_index, problems, methods, computePoints, num_starts=100):
    np.random.seed(42) 

    selected_problem = problems[problem_index]
    delta_random = (8 * np.random.rand() + 2) / 100


    [n, m, l, u, x_samples] = inip(selected_problem, num_starts)
    real_num_starts = len(x_samples) 

    ## Select x0 in dom(F) to compute  (controlling the uncertainty of the robust problem)
    x0_ref = x_samples[0]

    delta = delta_random * np.linalg.norm(x0_ref)
    [dimA, A, b] = data(n, m, delta)

    # Compute f_start for each starting point
    f_starts = np.full((real_num_starts, m), np.nan)

    for i in range(real_num_starts):
        x0 = x_samples[i]
        for j in range(m):
            g = evalg(n, x0, j, selected_problem)
            h, info = evalh(n, x0, j, A, b)
            if info == 0:
                f_starts[i, j] = g + h
            else:
                break
    
    # warmup runs for each method to avoid cold start effects
    if WARMUP_ENABLED:
        x0_warmup = x_samples[0]
        for method_name, method_func in methods.items():
            for _ in range(WARMUP_RUNS):
                method_func(n, m, l, u, x0_warmup, dimA, A, b, selected_problem)

    print(f"Run problem: {selected_problem}")

    x_opts = {method: np.full((real_num_starts, n), np.nan) for method in methods}
    f_opts = {method: np.full((real_num_starts, m), np.inf) for method in methods}
    iters_dict = {method: np.full(real_num_starts, np.nan) for method in methods}
    times_dict = {method: np.full(real_num_starts, np.nan) for method in methods}
    ngev_dict  = {method: np.full(real_num_starts, np.nan) for method in methods}
    nhev_dict  = {method: np.full(real_num_starts, np.nan) for method in methods}
    success_counts = {method: 0 for method in methods}
    stepsizes_dict = {method: np.full(real_num_starts, np.nan) for method in methods}

    # Compute error counts for each method
    error_codes_to_track = [-1, 1, 2, 3, 4]
    error_counts = {
        method: {code: 0 for code in error_codes_to_track}
        for method in methods
    }

    # Run each method for each starting point
    for method_name, method_func in methods.items():
        print(f"  Running method: {method_name}")

        for start_idx in range(real_num_starts):
            x0 = x_samples[start_idx]
            try:
                x_opt, f_opt, info, iter_count, ngev, nhev, elapsed_time, stepsize = method_func(
                        n, m, l, u, x0, dimA, A, b, selected_problem
                )

                if info == 0 or info == 1:
                    x_opts[method_name][start_idx] = x_opt
                    f_opts[method_name][start_idx] = f_opt
                    iters_dict[method_name][start_idx] = iter_count
                    times_dict[method_name][start_idx] = elapsed_time
                    ngev_dict[method_name][start_idx] = ngev
                    nhev_dict[method_name][start_idx] = nhev
                    success_counts[method_name] += 1
                    stepsizes_dict[method_name][start_idx] = stepsize

                if info != 0:
                    error_counts[method_name][info] += 1

            except Exception as e:
                print(f"    Error while running {method_name} with start {start_idx}: {e}")
                error_counts[method_name][-1] += 1  # Count undefined errors
                continue

    # Compute metrics for each method 
    metrics = {method: {"N": 0, "Purity": np.nan, "Hypervolume": np.nan, "Spread_Gamma": np.nan} for method in methods}
    results_list = None

    try:
        f_opts_clean = {}
        x_opts_clean = {}
        x_starts_clean = {}
        any_valid = False

        for method in methods:
            f_arr = np.array(f_opts[method])
            x_arr = np.array(x_opts[method])
            mask = np.all(np.isfinite(f_arr), axis=1)
            any_valid = any_valid or bool(np.any(mask))
            f_opts_clean[method] = f_arr[mask]
            x_opts_clean[method] = x_arr[mask]
            x_starts_clean[method] = np.array(x_samples)[mask]

        if any_valid:
            metrics, results_list = computePoints.compute_metrics(f_opts_clean, x_opts_clean, methods)

    except Exception as e:
        print(f"Error while computing metrics for problem {selected_problem}: {e}")


    if any_valid:
        if results_list is not None:
            plot_start_end_point(
                problem_name=selected_problem,
                m=m,
                methods=methods,
                f_starts=f_starts,
                f_opts=f_opts,
                save_dir=savedir
            )
            print(f"  Plot saved for {selected_problem}")
    

    # Error statistics table for this problem (to be saved to Excel)
    error_summary = []
    for method in methods:
        row = {"Method": method}
        for code in error_codes_to_track:
            row[f"Info {code}"] = error_counts[method][code]
        row["Total Errors"] = sum(error_counts[method].values())
        error_summary.append(row)
    
    df_errors = pd.DataFrame(error_summary)

    table = []
    for method_name in methods:
        if success_counts[method_name] > 0:
            time_ms       = round(np.nanmean(times_dict[method_name]) * 1000, 8)
            iter_mean     = round(np.nanmean(iters_dict[method_name]), 8)
            stepsize      = round(np.nanmean(stepsizes_dict[method_name]), 8)
            nhev_mean     = round(np.nanmean(nhev_dict[method_name]), 2)
        else:
            time_ms = iter_mean = stepsize = nhev_mean = np.nan

        table.append([
            method_name,
            success_counts[method_name],
            time_ms,
            iter_mean,
            stepsize,
            nhev_mean,
            metrics.get(method_name, {}).get("Global_nondominated", 0),
            metrics.get(method_name, {}).get("Purity", np.nan),
            metrics.get(method_name, {}).get("Hypervolume", np.nan),
            metrics.get(method_name, {}).get("Spread_Gamma", np.nan),
        ])

    # Save to data frame
    columns = ["Method", "Success", "Time(ms)", "Iter", "Stepsize", "H-Ev",  
               "Global_nondominated", "Purity", "Hypervolume", "Spread_Gamma"]

    df = pd.DataFrame(table, columns=columns)
    df.insert(0, "Problem", selected_problem)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print()
    
    # Add Problem column to df_errors
    df_errors.insert(0, "Problem", selected_problem)

    return df, df_errors    


def run_all_problems(nmax):
    all_dfs = []
    all_errors = []


    print("\n======================= START =======================")

    for problem_index in range(len(problems)):
        result = run_problem(
            problem_index,
            problems,
            methods,
            computePoints,
            nmax
        )
        
        if result is not None:
            df, df_errors = result
            all_dfs.append(df)
            all_errors.append(df_errors)

    print("\n======================= DONE =======================")

    # === Save all results ===
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_errors = pd.concat(all_errors, ignore_index=True)
        
        # Save to 2 separate sheets in the same Excel file
        with pd.ExcelWriter(f"{savedir}/Results.xlsx", engine='openpyxl') as writer:
            final_df.to_excel(writer, sheet_name='Results', index=False)
            final_errors.to_excel(writer, sheet_name='Error_Stats', index=False)
        
        print(f"All results saved to {filename}.xlsx (2 sheets: Results, Error_Stats)")
    else:
        print("No data was saved.")


problems = [
    "AP1", 
    "AP4", 
    "DTLZ1", 
    "DTLZ2", "DTLZ3", "DTLZ4", "FA1", "Far1",
    "FDS", "IKK1", "LE1", "LTDZ", "MGH33", "MLF1", "MMR4", "MOP2",
    "MOP3", "MOP5", "SK1", "TKLY1", "Toi9", "ZDT6",  

]

# List of algorithms
methods = {
    "MPGE": MPGE,
    "AMPG1": AMPG1,
    "AMPG2": AMPG2,
    "PGM": PGM,
    "accPGM": accPGM,
}

if __name__ == "__main__":

    num_starts = 100
    run_all_problems(num_starts)



