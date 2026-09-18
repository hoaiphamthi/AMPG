# New adaptive proximal gradient algorithms for solving multiobjective composite optimization problems
This repository contains Python implementations of paper "New adaptive proximal gradient algorithms for solving multiobjective composite optimization problems".

# Authors: PHAM THI HOAI, PHAM THI KHANH, JEN-CHIH YAO

## Overview
This project implements algorithms for solving multi-objective composite optimization problems, where one function is differentiable with a global Lipschitz constant, and the other is convex, proper, and closed. Two main experiments are conducted: robust multi-objective optimization and supervised feature selection.

### Implemented Algorithms

*   **AMPG1 / AMPG2**: The main proposed algorithms in the paper
*   **MPGE**: The algorithm proposed by Bello et al. (2025) in the paper " A proximal gradient method with an explicit line search for multiobjective optimization" (DOI: 10.1007/s10589-025-00711-x)
*   **PGM / accPGM**: The algorithms proposed by Zhao et al. (2025) in the paper "Proximal gradient method for convex multiobjective optimization problems without Lipschitz continuous gradients" (DOI: 10.1007/s10589-025-00663-2)


## Project Structure

*   **`main_robust_problem.py`**: Main script to run  multiobjective robust optimization problems.
*   **`main_sfs_benchmark.py`**: Main script to run biobjective supervised feature selection benchmarks via multiobjective optimization.
*   **`robust_algorithms.py`**: Implementations of the optimization algorithms for multiobjective robust optimization problems.
*   **`sfs_algorithms.py`**: Implementations of the algorithms for biobjective supervised feature selection problems.
*   **`computePoints.py`**: Utilities for computing Pareto fronts (non-dominated sorting), hypervolume, spread (Delta), and other metrics.
*   **`evalg.py`, `evalh.py`, `evalgradg.py`**: Functions evaluating the objective components ($g$, $h$) and gradients for robust problems.
*   **`data.py`, `inip.py`**: Utilities for generating constraint data and initializing problem parameters of robust problems.
*   **`plots.py`**: Plotting utilities for Pareto fronts and convergence.

## Requirements

Dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Multiobjective Robust Optimization Problems

To run the multiobjective robust optimization problem across benchmark datasets presented in our paper:

```bash
python main_robust_problem.py
```
This script will:
*   Evaluate multiple algorithms (AMPG1, AMPG2, PGM, etc.) from different starting points.
*   Compute metrics (Purity, Hypervolume, Spread, Global non-dominated points).
*   Generate an Excel file (e.g., `Results.xlsx`) containing detailed results and error statistics in the output directory.

### 2. Biobjecitive Supervised Feature Selection (B-SFS)

To run the biobjective supervised feature selection problem across benchmark datasets presented in our paper:

```bash
python main_sfs_benchmark.py
```
This script evaluates the algorithms on datasets located in the `data` folder extracted from the corresponding ZIP file. Results and metrics will be saved similarly.


### 3. Plotting Performance Profiles

To generate performance profiles:

* Open the plots.py file.
* Replace the path of the result Excel file from either of the two experiments into EXCEL_PATH inside plots.py.
* Run the plots.py script to generate the plots:

```bash
python plots.py
```

## Outputs

Execution creates a timestamped or parameter-named output directory (e.g., `Results_robust_problem_alpha=...`) containing:
*   **`Results.xlsx`**: Contains two sheets:
    *   `Results`: Algorithm performance metrics (Time, Iterations, Hypervolume, etc.).
    *   `Error_Stats`: Breakdown of error codes encountered during execution.
*   Plots for starting/ending points (if enabled).
*   For Performance profiles: run the `plots.py` file, replacing the path with the corresponding result file path.

## Appendix: Benchmark datasets for SFS problems
The datasets used in the B-SFS experiments are sourced from standard public repositories:

| Dataset | Source |
| :--- | :--- |
| **CMC** | [UCI Repository](https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice) |
| **Heart** | [UCI Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| **Parkinsons** | [UCI Repository](https://archive.ics.uci.edu/dataset/174/parkinsons) |
| **Ionosphere** | [UCI Repository](https://archive.ics.uci.edu/dataset/52/ionosphere) |
| **Gallstone** | [UCI Repository](https://archive.ics.uci.edu/dataset/1150/gallstone-1) |
| **Musk-v1** | [UCI Repository](https://archive.ics.uci.edu/dataset/74/musk+version+1) |
| **Arrhythmia** | [UCI Repository](https://archive.ics.uci.edu/dataset/5/arrhythmia) |
| **Darwin** | [UCI Repository](https://archive.ics.uci.edu/dataset/732/darwin) |

