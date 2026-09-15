import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import re
from computePoints import NonDominatedSorting, get_global_nondominated_indices

# ========== CONFIG (from plot_performance_profiles) ==========
#EXCEL_PATH = r"Result_sfs_benchmark_alpha=1.5_beta=6_c0=0.2_c1=0.19\Results.xlsx"
EXCEL_PATH = r"Results_robust_problem_alpha=1.5_beta=6_c0=0.2_c1=0.19\Results.xlsx"

METRICS = [
    "Time(ms)", "Iter",
    "Stepsize",
    "Hypervolume", "Spread_Gamma", "Purity",
    "Global_nondominated", "H-Ev",
]

HIGHER_IS_BETTER = {
    "Hypervolume": True,
    "Purity": True,
    "Stepsize": True,
    "Global_nondominated": True,
}

NONDOM_ALIASES = ["Global_nondominated", "Nondominated"]

COLORS = ["red", "dodgerblue", "limegreen", "purple", "chocolate",
          "orange", "deepskyblue", "magenta", "deeppink", "teal",
          "navy", "coral", "goldenrod", "indigo", "seagreen"]

MARKERS = ["o", "s", "D", "^", "v", "p", "*", "h", "+", "x", "<", ">"]

METHOD_STYLE = {
    "AMPG1": {"color": "red", "marker": "o"},
    "AMPG2": {"color": "dodgerblue", "marker": "s"},
    "MPGE": {"color": "limegreen", "marker": "D"},
    "PGM": {"color": "purple", "marker": "^"},
    "accPGM": {"color": "orange", "marker": "*"}
}

color_cycle = COLORS # Using the combined colors list

def resolve_column(df, aliases):
    for c in aliases:
        if c in df.columns:
            return c
    return None

def compute_profile(ratios, tau_grid):
    n = len(ratios)
    if n == 0:
        return np.zeros_like(tau_grid)
    return np.array([np.sum((ratios <= tau) & np.isfinite(ratios)) / n for tau in tau_grid])

def plot_performance_profiles(excel_path=None, save_path=None):
    if excel_path is None:
        excel_path = EXCEL_PATH
    
    dir_name = os.path.basename(os.path.dirname(os.path.abspath(excel_path)))
    if dir_name.startswith("Result_"):
        dir_name = dir_name[7:]

    if save_path is None:
        save_dir = f"Performance_profiles_{dir_name}"
    else:
        # Nếu truyền save_path dạng file .png, bỏ đuôi png làm thư mục
        if save_path.lower().endswith(".png"):
            save_dir = save_path[:-4]
        else:
            save_dir = save_path

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(excel_path):
        print(f"[LỖI] Không tìm thấy file: {excel_path}")
        return

    df = pd.read_excel(excel_path, sheet_name="Results")

    if "Dataset" in df.columns and "Problem" not in df.columns:
        df = df.rename(columns={"Dataset": "Problem"})

    df = df[df["Success"] > 0].copy()

    if df.empty:
        print("[LỖI] Không có dữ liệu hợp lệ (Success > 0)")
        return

    problems = df["Problem"].unique()
    methods = list(df["Method"].unique())

    # Move AMPG2 and AMPG1 to the end so they are plotted last (AMPG1 on very top)
    for m in ["AMPG2", "AMPG1"]:
        if m in methods:
            methods.remove(m)
            methods.append(m)

    print(f"\n{'='*50}")
    print(f"Performance Profiles")
    print(f"File: {excel_path}")
    print(f"Problems: {len(problems)}     Methods: {len(methods)}")
    print(f"Output Directory: {save_dir}")
    print(f"{'='*50}")

    win_counts = {m: {metric: 0 for metric in METRICS} for m in methods}

    for idx, metric in enumerate(METRICS):
        fig, ax = plt.subplots(figsize=(8, 6))

        if metric == "Global_nondominated":
            col_name = resolve_column(df, NONDOM_ALIASES)
        else:
            col_name = metric if metric in df.columns else None

        display_title = metric
        if metric == "Spread_Gamma":
            display_title = r"Spread ($\Gamma$)"
        elif metric == "Spread_Gamma":
            display_title = r"Spread ($\Gamma$)"

        if col_name is None:
            plt.close(fig)
            continue

        pivot = df.pivot_table(index="Problem", columns="Method",
                               values=col_name, aggfunc="mean")

        higher = HIGHER_IS_BETTER.get(metric, False)

        if higher:
            best_per_prob = pivot.max(axis=1).values
        else:
            best_per_prob = pivot.where(pivot > 0).min(axis=1).fillna(0).values

        ratios_dict = {}
        for method in methods:
            if method not in pivot.columns:
                continue
            vals = pivot[method].values
            
            with np.errstate(divide="ignore", invalid="ignore"):
                if higher:
                    ratios = np.where(vals > 0, best_per_prob / vals, np.inf)
                else:
                    ratios = np.where(best_per_prob > 0, vals / best_per_prob, np.inf)
            
            ratios = np.maximum(1.0, ratios)
            ratios[~np.isfinite(ratios)] = np.nan

            ratios_dict[method] = ratios

        # Calculate absolute maximum possible tau first
        tau_max_abs = 1.0
        for method, ratios in ratios_dict.items():
            finite = ratios[np.isfinite(ratios)]
            if len(finite) > 0:
                tau_max_abs = max(tau_max_abs, np.max(finite))

        # Ensure tau_max is at least 1.0
        tau_max = max(1.01, tau_max_abs * 1.05) # Add a very small margin (5%)

        n_grid = 100
        tau_grid = np.logspace(0, np.log2(tau_max), n_grid, base=2)

        for mi, method in enumerate(methods):
            if method not in ratios_dict:
                continue
            profile = compute_profile(ratios_dict[method], tau_grid)
            
            style = METHOD_STYLE.get(method, {})
            color = style.get("color", COLORS[mi % len(COLORS)])
            marker = style.get("marker", MARKERS[mi % len(MARKERS)])

            # Draw step line without label
            ax.step(tau_grid, profile, where='post', color=color, linewidth=1.8)
            
            # Staggered marker indices to prevent overlapping
            n_markers = 10
            step_size = max(1, len(tau_grid) // n_markers)
            offset = (mi * 3) % step_size  # Shift offset based on method index
            show_idx = np.arange(offset, len(tau_grid), step_size)
            show_idx = np.unique(np.append([0], show_idx))
            
            # Draw markers on the actual line without label
            ax.plot(tau_grid[show_idx], profile[show_idx], color=color,
                    marker=marker, linestyle='None', markersize=6)
            
            # Dummy plot for legend to show both line and marker
            ax.plot([], [], color=color, marker=marker, linewidth=1.8, 
                    markersize=6, label=method)

            win_counts[method][metric] = int(0)

        for prob in pivot.index:
            row = pivot.loc[prob].dropna()
            if row.empty:
                continue
            best = row.idxmax() if higher else row.idxmin()
            if best in win_counts:
                win_counts[best][metric] += 1

        ax.set_xscale("log", base=2)
        ax.set_xlim(1, tau_max)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(r"$\tau$", fontsize=12)
        ax.set_ylabel(r"$\rho(\tau)$", fontsize=12)
        ax.set_title(display_title, fontweight="bold", fontsize=14)
        ax.grid(True, linestyle="--", color="lightgray", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        order_dict = {"AMPG1": 0, "AMPG2": 1, "MPGE": 2, "PGM": 3, "accPGM": 4}
        sorted_pairs = sorted(zip(handles, labels), key=lambda x: order_dict.get(x[1], 99))
        if sorted_pairs:
            handles, labels = zip(*sorted_pairs)
        
        ax.legend(handles, labels, fontsize=12, loc="lower right", framealpha=0.85,
                  ncol=1 if len(methods) <= 6 else 2)

        safe_metric_name = metric.replace("/", "_").replace("\\", "_")
        out_file = os.path.join(save_dir, f"{safe_metric_name}.png")
        plt.tight_layout()
        fig.savefig(out_file, bbox_inches="tight", dpi=200)
        plt.close(fig)

    print(f"\nSaved plots to folder: {save_dir}")

    print(f"\n{'='*60}")
    print(f"{'Metric':<20} {'Best Method':<15} {'Wins':<6}")
    print(f"{'-'*60}")
    for metric in METRICS:
        if metric == "Global_nondominated":
            col_name = resolve_column(df, NONDOM_ALIASES)
        else:
            col_name = metric if metric in df.columns else None
            
        if col_name is None:
            continue
            
        best_method = max(methods, key=lambda m: win_counts[m][metric])
        best_wins = win_counts[best_method][metric]
        print(f"{metric:<20} {best_method:<15} {best_wins}/{len(problems)}")
    print(f"{'='*60}\n")

    return win_counts



def plot_start_end_point(problem_name, m, methods, f_starts, f_opts, save_dir="."):
    if m not in [2, 3]:
        return

    method_names = list(methods.keys())
    n_methods = len(method_names)

    n_cols = n_methods
    n_rows = 1

    
    if m == 3:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 6, n_rows * 6),
                                 subplot_kw={'projection': '3d'})
    else:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 6, n_rows * 6))

    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    f1_min, f1_max = np.inf, -np.inf
    f2_min, f2_max = np.inf, -np.inf
    f3_min, f3_max = np.inf, -np.inf

    for method_name in method_names:
        f_ends = f_opts[method_name]
        for i in range(len(f_starts)):
            if np.all(np.isfinite(f_ends[i])):
                f1_min = min(f1_min, f_starts[i, 0], f_ends[i, 0])
                f1_max = max(f1_max, f_starts[i, 0], f_ends[i, 0])
                f2_min = min(f2_min, f_starts[i, 1], f_ends[i, 1])
                f2_max = max(f2_max, f_starts[i, 1], f_ends[i, 1])
                if m == 3:
                    f3_min = min(f3_min, f_starts[i, 2], f_ends[i, 2])
                    f3_max = max(f3_max, f_starts[i, 2], f_ends[i, 2])

    if np.isfinite(f1_min):
        margin_f1 = (f1_max - f1_min) * 0.15 if f1_max > f1_min else 0.1
        margin_f2 = (f2_max - f2_min) * 0.15 if f2_max > f2_min else 0.1
        f1_min -= margin_f1
        f1_max += margin_f1
        f2_min -= margin_f2
        f2_max += margin_f2
        if m == 3:
            margin_f3 = (f3_max - f3_min) * 0.15 if f3_max > f3_min else 0.1
            f3_min -= margin_f3
            f3_max += margin_f3
    else:
        f1_min, f1_max = 0, 1
        f2_min, f2_max = 0, 1
        f3_min, f3_max = 0, 1

    for idx, method_name in enumerate(method_names):
        ax = axes[idx]
        color = color_cycle[idx % len(color_cycle)]
        f_ends = np.array(f_opts[method_name])

        valid_indices = [i for i in range(len(f_starts)) if np.all(np.isfinite(f_ends[i]))]
        valid_f_ends = f_ends[valid_indices]
        
        is_local_nondominated = np.zeros(len(f_starts), dtype=bool)
        if len(valid_f_ends) > 0:
            local_front_indices = NonDominatedSorting().do(valid_f_ends, only_non_dominated_front=True)
            for j in local_front_indices:
                is_local_nondominated[valid_indices[j]] = True

        for i in range(len(f_starts)):
            if np.all(np.isfinite(f_ends[i])):
                if is_local_nondominated[i]:
                    end_marker = '*'
                    end_color = 'red'
                    end_size = 100
                else:
                    end_marker = 'D'
                    end_color = 'blue'
                    end_size = 20
                    
                if m == 3:
                    ax.plot([f_starts[i, 0], f_ends[i, 0]],
                            [f_starts[i, 1], f_ends[i, 1]],
                            [f_starts[i, 2], f_ends[i, 2]],
                            color="gray", alpha=0.3, linewidth=0.8)
                    ax.scatter(f_starts[i, 0], f_starts[i, 1], f_starts[i, 2],
                               color="yellow", s=10, alpha=0.8, marker='o', edgecolors="gray", linewidth=0.5)
                    ax.scatter(f_ends[i, 0], f_ends[i, 1], f_ends[i, 2],
                               color=end_color, s=end_size, alpha=0.8, marker=end_marker, edgecolors="gray", linewidth=0.5)
                else:
                    ax.plot([f_starts[i, 0], f_ends[i, 0]],
                            [f_starts[i, 1], f_ends[i, 1]],
                            color="gray", alpha=0.3, linewidth=0.8)
                    ax.scatter(f_starts[i, 0], f_starts[i, 1],
                               color="yellow", s=10, alpha=0.8, marker='o', edgecolors="gray", linewidth=0.5)
                    ax.scatter(f_ends[i, 0], f_ends[i, 1],
                               color=end_color, s=end_size, alpha=0.8, marker=end_marker, edgecolors="gray", linewidth=0.5)

        ax.set_xlim(f1_min, f1_max)
        ax.set_ylim(f2_min, f2_max)
        ax.set_xlabel(r"$f_1$")
        ax.set_ylabel(r"$f_2$")
        if m == 3:
            ax.set_zlim(f3_min, f3_max)
            ax.set_zlabel(r"$f_3$")
        if m == 2:
            ax.grid(True, linestyle="--", color="lightgray", alpha=0.7)
        ax.set_title(method_name, fontweight="bold", fontsize=14)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Trajectory from Start to End - {problem_name}',
                 fontweight="bold", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    import os
    plt.savefig(os.path.join(save_dir, f"Trajectory_{problem_name}.png"), bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else EXCEL_PATH
    plot_performance_profiles(path)
