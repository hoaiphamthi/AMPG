import numpy as np
import time
from quadprog import solve_qp

from evalg import evalg
from evalh import evalh
from evalgradg import evalgradg


# -----------------------------------------------------------------------------
# Subproblems
# -----------------------------------------------------------------------------

def subproblem_robust(n, m, l, u, x, alpha, JGx, Hx, dimA, A_list, b_list):
    info = -1
    
    try:
        sdimA = np.sum(dimA)
        total_vars = 1 + n + sdimA

        # quadprog (Goldfarb-Idnani) requires matrix P to be strictly positive definite.
        # Variables tau and w lack quadratic terms, making P positive semi-definite and causing errors.
        # Add reg = 1e-8 to the diagonal to enforce a PD matrix and prevent inversion errors.
        reg = 1e-8
        P = np.zeros((total_vars, total_vars))
        P[1:n+1, 1:n+1] = np.eye(n) / alpha
        P[0, 0] = reg
        P[1+n:, 1+n:] = np.eye(sdimA) * reg

        q = np.zeros(total_vars)
        q[0] = 1.0
        q[1:n+1] = -x / alpha

        a = -q

        index_starts = 1 + n + np.hstack([0, np.cumsum(dimA[:-1])]).astype(int)

        A_in = np.zeros((m, total_vars))
        b_in = np.zeros(m)

        A_eq = np.zeros((n * m, total_vars))

        for i in range(m):
            start_idx_y = index_starts[i]
            end_idx_y = start_idx_y + dimA[i]

            A_in[i, 0] = -1.0
            A_in[i, 1:n+1] = JGx[i, :]
            A_in[i, start_idx_y : end_idx_y] = b_list[i].flatten()
            b_in[i] = Hx[i] + np.dot(JGx[i, :], x)

            row_start_eq = i * n
            row_end_eq = (i + 1) * n

            A_eq[row_start_eq : row_end_eq, 1:n+1] = -np.eye(n)
            A_eq[row_start_eq : row_end_eq, start_idx_y : end_idx_y] = A_list[i].T

        G_p_lower = np.zeros((n, total_vars))
        G_p_lower[:, 1:n+1] = -np.eye(n)
        h_p_lower = - l 
    
        G_p_upper = np.zeros((n, total_vars))
        G_p_upper[:, 1:n+1] = np.eye(n)
        h_p_upper = u  

        G_w_lower = np.zeros((sdimA, total_vars))
        G_w_lower[:, 1+n:] = -np.eye(sdimA)
        h_w_lower = np.zeros(sdimA)

        G = np.vstack([A_in, G_p_lower, G_p_upper, G_w_lower])
        h = np.hstack([b_in, h_p_lower, h_p_upper, h_w_lower])

        b_eq = np.zeros(n * m)

        C = np.hstack([A_eq.T, -G.T])
        b_combined = np.hstack([b_eq, -h])
        meq = n * m

        sol = solve_qp(P, a, C, b_combined, meq)

        info = 0
        xopt = sol[0]

        tau = xopt[0]
        p = xopt[1:n+1]
        theta = tau + 0.5 * np.linalg.norm(p - x)**2 / alpha

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Critical error in 'robust_subproblem': {e}")
        info = 3
        theta, tau, p = np.nan, np.nan, np.nan

    return theta, tau, p, info


def subproblem_robust_accPGM(n, m, l, u, x, y, alpha, JGy, Gy, Fx, dimA, A_list, b_list):
    info = -1
    
    try:
        sdimA = np.sum(dimA)
        total_vars = 1 + n + sdimA

        # quadprog (Goldfarb-Idnani) requires matrix P to be strictly positive definite.
        # Variables tau and w lack quadratic terms, making P positive semi-definite and causing errors.
        # Add reg = 1e-8 to the diagonal to enforce a PD matrix and prevent inversion errors.
        reg = 1e-8
        P = np.zeros((total_vars, total_vars))
        P[1:n+1, 1:n+1] = np.eye(n) / alpha
        P[0, 0] = reg
        P[1+n:, 1+n:] = np.eye(sdimA) * reg

        q = np.zeros(total_vars)
        q[0] = 1.0
        q[1:n+1] = -y / alpha

        a = -q

        index_starts = 1 + n + np.hstack([0, np.cumsum(dimA[:-1])]).astype(int)

        A_in = np.zeros((m, total_vars))
        b_in = np.zeros(m)

        A_eq = np.zeros((n * m, total_vars))

        for i in range(m):
            start_idx_y = index_starts[i] if i < len(index_starts) else index_starts[0]
            end_idx_y = start_idx_y + dimA[i]

        
            A_in[i, 0] = -1.0
            A_in[i, 1:n+1] = JGy[i, :]
            A_in[i, start_idx_y : end_idx_y] = b_list[i].flatten()
            b_in[i] = np.dot(JGy[i, :], y) - Gy[i] + Fx[i]

            row_start_eq = i * n
            row_end_eq = (i + 1) * n
            A_eq[row_start_eq : row_end_eq, 1:n+1] = -np.eye(n)
            A_eq[row_start_eq : row_end_eq, start_idx_y : end_idx_y] = A_list[i].T


        G_p_lower = np.zeros((n, total_vars))
        G_p_lower[:, 1:n+1] = -np.eye(n)
        h_p_lower = -l

        G_p_upper = np.zeros((n, total_vars))
        G_p_upper[:, 1:n+1] = np.eye(n)
        h_p_upper = u

        G_w_lower = np.zeros((sdimA, total_vars))
        G_w_lower[:, 1+n:] = -np.eye(sdimA)
        h_w_lower = np.zeros(sdimA)

        G = np.vstack([A_in, G_p_lower, G_p_upper, G_w_lower])
        h = np.hstack([b_in, h_p_lower, h_p_upper, h_w_lower])

        b_eq = np.zeros(n * m)

        C = np.hstack([A_eq.T, -G.T])
        b_combined = np.hstack([b_eq, -h])
        meq = n * m

        sol = solve_qp(P, a, C, b_combined, meq)

        info = 0
        xopt = sol[0]

        tau = xopt[0]
        p = xopt[1:n+1]
        theta = tau + 0.5 * np.linalg.norm(p - y)**2 / alpha

    except Exception as e:
        print(f"Critical error in 'robust_subproblem_accPGM': {e}")
        info = 3
        theta, tau, p = np.nan, np.nan, np.nan

    return theta, tau, p, info


# -----------------------------------------------------------------------------
# AMPG Methods
# -----------------------------------------------------------------------------

def _AMPG_core(n, m, l, u, x, dimA, A, b, problem_name, opt, 
               alpha, beta, c0=0.2, c1=0.19):
    
    start_time = time.perf_counter()

    # Initial parametes 
    tol = 1e-6
    max_iter = 1000
    info = 0

    # Counters
    ngev = 0
    nhev = 0
    tk_sum = 0.0

    G = np.zeros(m)
    H = np.zeros(m)
    JG_next = np.zeros((m, n))

    xk = x.copy()
    tk = 1.0

    JG = np.array([evalgradg(n, xk, i, problem_name) for i in range(m)])


    for iter in range(1, max_iter + 1):
        tk_sum += tk
        # Evaluate F
        for i in range(m):
            G[i] = evalg(n, xk, i, problem_name)
            H[i], info = evalh(n, xk, i, A, b)
            
        ngev += m
        nhev += m

        if info != 0:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return xk, G + H, info, iter, ngev, nhev, elapsed_time, stepsize
        
        F = G + H

        # Step 1: Compute x_{k+1} by solving the subproblem
        _, _, xk_next, info = subproblem_robust(n, m, l, u, xk, tk, JG, H, dimA, A, b)

        # Check for an error while solving the subproblem
        if info != 0:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return xk, F, info, iter, ngev, nhev, elapsed_time, stepsize

    
        #  ----------------------------------------------------------
        # Stopping criteria
        #  ----------------------------------------------------------
        d = xk_next - xk
        # Compute norm (d)
        step_norm = np.linalg.norm(d)

        # Compute norm(x-xprev,inf)/max(1,norm(xprev,inf)
        supnorm_x_xk = np.linalg.norm(d, np.inf) / max(1.0, np.linalg.norm(xk, np.inf))

        # Test optimality
        if step_norm <= tol or supnorm_x_xk <= tol:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return xk, F, info, iter, ngev, nhev, elapsed_time, stepsize

        # Compute the Jacobian of G
        for i in range(m):
            JG_next[i, :] = evalgradg(n, xk_next, i, problem_name)

        grad_d = JG_next - JG

        # AMPG1
        if opt == 1:
            val_compare = step_norm ** 2
            B_val = np.max(np.abs(grad_d @ d))

        # AMPG2
        else:
            val_compare = step_norm
            B_val = np.max(np.linalg.norm(grad_d, axis=1))

        if B_val > c0 * val_compare / tk:
            tk = c1 * val_compare / B_val
        else:
            tk = (1 + alpha * (np.log(iter)) ** beta / (iter ** 1.1)) * tk


        xk = xk_next.copy()
        JG = JG_next.copy()


    info = 1
    elapsed_time = time.perf_counter() - start_time
    stepsize = tk_sum / max_iter
    return xk, F, info, max_iter, ngev, nhev, elapsed_time, stepsize


def AMPG1(n, m, l, u, x, dimA, A, b, problem_name, alpha=1.5, beta=6, c0=0.2, c1=0.19):
    return _AMPG_core(n, m, l, u, x, dimA, A, b, problem_name, opt=1, 
                      alpha=alpha, beta=beta, c0=c0, c1=c1)


def AMPG2(n, m, l, u, x, dimA, A, b, problem_name, alpha=1.5, beta=6, c0=0.2, c1=0.19):
    return _AMPG_core(n, m, l, u, x, dimA, A, b, problem_name, opt=2,
                      alpha=alpha, beta=beta, c0=c0, c1=c1)


# -----------------------------------------------------------------------------
# MPGE Method and Line Search Procedures
# -----------------------------------------------------------------------------

def explicit_LS_one(n, x, d, t_trial, jk, G_jk, JGd_jk, problem_name):
    # Parameters
    gamma = 1.9999
    t_min = 1e-15
    sigma1 = 0.5e-1
    sigma2 = 9.5e-1

    # Counters:
    ngev = 0

    # Define ftest
    gtest = JGd_jk + gamma * (np.linalg.norm(d)**2) / 2.0

    # Evaluate f
    g = evalg(n, x + t_trial * d, jk, problem_name)
    ngev += 1

    while True:
        # Test the descent condition for f_jk
        if g <= G_jk + t_trial * gtest: 
            Gend = g
            info = 0
            return t_trial, Gend, ngev, info
        
        if t_trial <= t_min:
            Gend = g
            info = 4
            return t_trial, Gend, ngev, info
        
        if JGd_jk < 0:
            t_q = - (JGd_jk * t_trial ** 2) / 2.0 / (g - G_jk - JGd_jk * t_trial)

            if sigma1 * t_trial <= t_q <= sigma2 * t_trial:
                t_trial = t_q
            else: 
                t_trial = t_trial / 2.0
        else:
            t_trial = t_trial / 2.0

        g = evalg(n, x + t_trial * d, jk, problem_name)
        ngev += 1

     
def explicit_LS(n, m, x, d, t_trial, jk_first, G, JGd, problem_name):
    # Parameters
    gamma = 1.9999

    # Counters
    ngev = 0

    # Define gtest
    G_test = JGd + gamma * (np.linalg.norm(d)**2) / 2.0

    Gindex = list(range(m))
    Gindex[0] = jk_first
    Gindex[jk_first] = 0

    G_trial = np.zeros(m)

    # Test the sufficient descent condition (sdc) at t_trial = 1
    sdc = True
    iA = -1 

    for j in range(m):
        i = Gindex[j]

        G_trial[i] = evalg(n, x + t_trial * d, i, problem_name)
        ngev += 1

        if G_trial[i] > G[i] + t_trial * G_test[i]:
            sdc = False
            iA = i 
            g_trial_iA = G_trial[i] 
            break

    if sdc:
        info = 0 
        return t_trial, G_trial, ngev, info
    
    t_trial, G_trial, ngevbt, info = backtrackingMO(t_trial, n, m, x, d, G, JGd, iA, g_trial_iA, G_test, problem_name)
    ngev += ngevbt

    return t_trial, G_trial, ngev, info


def backtrackingMO(t_trial, n, m, x, d, G, JGd, iA, g_trial_iA, G_test, problem_name):
    # Parameters
    t_min = 1e-15
    sigma1 = 0.5e-1
    sigma2 = 9.5e-1

    # Counters
    outiter = 0
    ngev = 0

    Gend = np.zeros(m)
    info = 0 

    # -------------------------------------------------------------------
    #    Main loop
    # -------------------------------------------------------------------    
    while True:
        # Test the vector Armijo condition
        if outiter == 0:
            sdc = False
            ind = iA # First violating index
            g_trial = g_trial_iA # Value of G at iA

        elif info == 0: 
            sdc = True

            for i in range(m):
                if i == ind:
                    continue
                    
                g_trial = evalg(n, x + t_trial * d, i, problem_name)
                ngev += 1

                Gend[i] = g_trial
                
                if g_trial > G[i] + t_trial * G_test[i]:
                    sdc = False
                    ind = i
                    break
                    
        # Finish backtracking with the current point
        if sdc:
            info = 0
            return t_trial, Gend, ngev, info
            
        # Test if stp is too small
        if t_trial <= t_min:
            t_trial = t_min
            info = 4
            return t_trial, Gend, ngev, info
            
        outiter += 1
        
        # Compute new trial stepsize based on g_jk
        while True:

            # Test the descent condition for g_jk
            if g_trial <= G[ind] + t_trial * G_test[ind]:
                info = 0
                break
                
            if t_trial <= t_min:
                info = 4
                break
                
            if JGd[ind] < 0:
            
                t_q = - (JGd[ind] * t_trial ** 2) / 2.0 / (g_trial - G[ind] - JGd[ind] * t_trial)
                
                if sigma1 * t_trial <= t_q <= sigma2 * t_trial:
                    t_trial = t_q
                else: 
                     t_trial = t_trial / 2.0
            else:
                t_trial = t_trial / 2.0
                
            g_trial = evalg(n, x + t_trial * d, ind, problem_name)
            ngev += 1
            
        Gend[ind] = g_trial


def MPGE(n, m, l, u, x, dimA, A, b, problem_name):
    start_time = time.perf_counter()

    # Initial parametes 
    tol = 1e-6
    itermax = 1000
    alpha = 1.0

    # Counters
    ngev = 0 
    nhev = 0 
    iter = 0

    tk_sum = 0.0

    G = np.zeros(m)
    H = np.zeros(m)
    F = np.zeros(m)
    JG = np.zeros((m, n))

    # Evaluate F
    for i in range(m):
        G[i] = evalg(n, x, i, problem_name)  
        H[i], info = evalh(n, x, i, A, b)

    ngev += m
    nhev += m

    #  Check for an error while evaluating H
    if info != 0:
        elapsed_time = time.perf_counter() - start_time
        stepsize = tk_sum / iter if iter > 0 else 0.0
        return x, F, info, iter, ngev, nhev, elapsed_time, stepsize
        
    F = G + H

    supnorm_x_p = 1.0

    while True:
        iter += 1

        # Test the maximum number of iterations
        if iter > itermax:
            info = 1
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / itermax
            return x, F, info, itermax, ngev, nhev, elapsed_time, stepsize
        
        # Compute the Jacobian of G
        for i in range(m):
            JG[i, :] = evalgradg(n, x, i, problem_name)
        
        # Solve the subproblem
        _, _, p, info = subproblem_robust(n, m, l, u, x, alpha, JG, H, dimA, A, b)

        # Check for an error while solving the subproblem
        if info != 0:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return x, F, info, iter, ngev, nhev, elapsed_time, stepsize
        

        # Compute linesearch
        d = p - x

        #  ----------------------------------------------------------
        # Stopping criteria
        #  ----------------------------------------------------------

        # Compute norm(x-xprev,inf)/max(1,norm(xprev,inf)
        if iter > 1:
            supnorm_x_p = np.linalg.norm(x - xprev, np.inf) / max(1.0, np.linalg.norm(xprev, np.inf))
        
        # Compute norm (d)
        step_norm = np.linalg.norm(d)

        # Test optimality
        if step_norm <= tol or supnorm_x_p <= tol:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter 
            return x, F, info, iter, ngev, nhev, elapsed_time, stepsize
                    
        #  ----------------------------------------------------------
        # Iterate
        #  ----------------------------------------------------------
        xprev = np.copy(x)

        t_trial = 1.0
        JGd = JG @ d

        # Step 3: jk == jk*
        jk = np.argmax(JGd)

        # Step 3.1
        t_trial, Gtrial_jk, ngevLS, info = explicit_LS_one(n, x, d, t_trial, jk, G[jk], JGd[jk], problem_name)
        ngev += ngevLS

        # Check for an error in the line search procedure
        if info != 0:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return x, F, info, iter, ngev, nhev, elapsed_time, stepsize
        
        # Update x_trial, G_trial, H_trial, F_trial
        x_trial = x + t_trial * d
        F_trial = np.zeros(m)
        G_trial = np.zeros(m)
        H_trial = np.zeros(m)

        G_trial[jk] = Gtrial_jk
        Fdec = True     # Flag for checking F descent condition
        jk_first = -1

        for i in range(m):
            if i == jk:
                continue

            G_trial[i] = evalg(n, x_trial, i, problem_name)
            ngev += 1

            H_trial[i], info = evalh(n, x_trial, i, A, b)
            nhev += 1

            if info != 0:
                elapsed_time = time.perf_counter() - start_time
                stepsize = tk_sum / iter
                return x, F, info, iter, ngev, nhev, elapsed_time, stepsize
        
            F_trial[i] = G_trial[i] + H_trial[i]

            if F_trial[i] > F[i]:
                Fdec = False
                jk_first = i 
                break
                
        # Step 4:
        if Fdec:
           
            H_trial[jk], info = evalh(n, x_trial, jk, A, b)
            nhev += 1

            if info != 0:
                elapsed_time = time.perf_counter() - start_time
                stepsize = tk_sum / iter
                return x, F, info, iter, ngev, nhev, elapsed_time, stepsize

            F_trial[jk] = G_trial[jk] + H_trial[jk]

            # Update x
            x = x_trial
            G = G_trial
            H = H_trial
            F = F_trial
            tk_sum += t_trial
           

        # Step 3.3
        else:
            t_trial, G, ngevLS, info = explicit_LS(n, m, x, d, t_trial, jk_first, G, JGd, problem_name)
            ngev += ngevLS

            #  Check for an error in the line seach procedure
            if info != 0:
                elapsed_time = time.perf_counter() - start_time
                stepsize = tk_sum / iter
                return x, F, info, iter, ngev, nhev, elapsed_time, stepsize
            
            # Update x
            x = x + t_trial * d

            for i in range(m):
                H[i], info = evalh(n, x, i, A, b)
                
            nhev += m
            if info != 0:
                elapsed_time = time.perf_counter() - start_time
                stepsize = tk_sum / iter
                return x, F, info, iter, ngev, nhev, elapsed_time, stepsize

            # Evaluate F
            F = G + H
            tk_sum += t_trial
            

# -----------------------------------------------------------------------------
# PGM and accPGM Methods
# -----------------------------------------------------------------------------

def PGM(n, m, l, u, xk, dimA, A, b, problem_name):
    start_time = time.perf_counter()

    # Parameters:
    delta = 0.3
    eta = 0.5
    sigma = 1

    tol = 1e-6
    max_iter = 1000

    # Counters:
    ngev = 0
    nhev = 0
    tk_sum = 0.0

    G = np.zeros(m)
    H = np.zeros(m)
    F = np.zeros(m)
    JG = np.zeros((m, n))
    JG_next = np.zeros((m, n))
    diff = 0
    
    # alpha0 = sigma
    alpha = sigma

    xk_next = np.zeros(n)
    
    for i in range(m):
        JG[i, :] = evalgradg(n, xk, i, problem_name)


    for iter in range(1, max_iter + 1):
        # Compute F
        for i in range(m):
            G[i] = evalg(n, xk, i, problem_name)
            H[i], info = evalh(n, xk, i, A, b)
            
        ngev += m
        nhev += m

        if info != 0:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return xk, F, info, iter, ngev, nhev, elapsed_time, stepsize
        
        F = G + H

        # Solve the subproblem: line search procedure in Step 3
        while True:
            _, _, xk_next, info = subproblem_robust(n, m, l, u, xk, alpha, JG, H, dimA, A, b)
            
            # Check for an error while solving subproblem.
            if info != 0:
                elapsed_time = time.perf_counter() - start_time
                stepsize = tk_sum / iter
                return xk, F, info, iter, ngev, nhev, elapsed_time, stepsize

            for i in range(m):
                JG_next[i, :] = evalgradg(n, xk_next, i, problem_name)

            grad_diff = JG_next - JG
            diff = np.linalg.norm(xk_next - xk)

            if np.any(alpha * np.linalg.norm(grad_diff, axis=1) > delta * diff):
                alpha *= eta 
                continue
            else:
                break

        tk_sum += alpha
        
        # ----------------------------------------------------------
        # Stopping criteria
        #  ----------------------------------------------------------
        d = xk_next - xk
        # Compute norm (d)
        step_norm = np.linalg.norm(d)

        # Compute norm(x-xprev,inf)/max(1,norm(xprev,inf)
        supnorm_x_xk = np.linalg.norm(d, np.inf) / max(1.0, np.linalg.norm(xk, np.inf))

        # Test optimality
        if supnorm_x_xk <= tol or step_norm <= tol:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return xk, F, info, iter, ngev, nhev, elapsed_time, stepsize
            
        xk = xk_next.copy()
        JG = JG_next.copy()


    info = 1
    elapsed_time = time.perf_counter() - start_time
    stepsize = tk_sum / max_iter
    return xk, F, info, max_iter, ngev, nhev, elapsed_time, stepsize


def accPGM(n, m, l, u, xk, dimA, A, b, problem_name):
    start_time = time.perf_counter()

    # Parameters:
    eta = 0.5
    delta = 0.3
    sigma = 1

    tol = 1e-5
    max_iter = 500

    yk = xk.copy()
    xk_next = np.zeros(n)
    bk = 1

    # alpha0 = sigma
    alpha = sigma
    tk_sum = 0.0
    
    # counters:
    ngev = 0
    nhev = 0
    
    Gx = np.zeros(m)
    Gy = np.zeros(m)
    Hx = np.zeros(m)
    Hy = np.zeros(m)
    Fx = np.zeros(m)
    Fy = np.zeros(m)
    JGy = np.zeros((m, n))
    JGx_next = np.zeros((m, n))

    for iter in range(1, max_iter + 1):
        # ---- Compute F
        for i in range(m):
            Gy[i] = evalg(n, yk, i, problem_name)
            Hy[i], info = evalh(n, yk, i, A, b)
            JGy[i, :] = evalgradg(n, yk, i, problem_name)
           
        ngev += m
        nhev += m

        if info != 0:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return xk, Fy, info, iter, ngev, nhev, elapsed_time, stepsize
        
        Fy = Gy + Hy

        # Compute Fx 
        if iter == 1:
            Gx, Hx = Gy.copy(), Hy.copy()
        else:
            for i in range(m):
                Gx[i] = evalg(n, xk, i, problem_name)
                Hx[i], info = evalh(n, xk, i, A, b)
            
            ngev += m
            nhev += m

            if info != 0:
                stepsize = tk_sum / iter
                elapsed_time = time.perf_counter() - start_time
                return yk, Fy, info, iter, ngev, nhev, elapsed_time, stepsize
        
        Fx = Gx + Hx
        
        # Solves the subproblem: Line search procedure A in Step 3
        while True:
            _, _, xk_next, info = subproblem_robust_accPGM(n, m, l, u, xk, yk, alpha, JGy, Gy, Fx, dimA, A, b)

            if info != 0:
                stepsize = tk_sum / iter
                elapsed_time = time.perf_counter() - start_time
                return yk, Fy, info, iter, ngev, nhev, elapsed_time, stepsize
            

            for i in range(m):
                JGx_next[i, :] = evalgradg(n, xk_next, i, problem_name)

            grad_diff = JGy - JGx_next
            diff = np.linalg.norm(yk - xk_next)

            if np.any(alpha * np.linalg.norm(grad_diff, axis=1) > delta * diff):
                alpha *= eta
                continue
            else:
                break
        
        tk_sum += alpha
        
        # ----------------------------------------------------------
        # Stopping criteria
        #  ----------------------------------------------------------
        d = yk - xk_next
        # Compute norm (d)
        step_norm = np.linalg.norm(d)

        # Compute norm(x-xprev,inf)/max(1,norm(xprev,inf)
        supnorm_yk_xk_next = np.linalg.norm(d, np.inf) / max(1.0, np.linalg.norm(yk, np.inf))

        # Test optimality
        if supnorm_yk_xk_next <= tol or step_norm <= tol:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return yk, Fy, info, iter, ngev, nhev, elapsed_time, stepsize
        

        bk_next = (1 + np.sqrt(1 + 4 * bk ** 2)) / 2
        yk = xk_next + (bk - 1) / bk_next * (xk_next - xk)
        xk = xk_next.copy()

            
    info = 1
    stepsize = tk_sum / max_iter
    elapsed_time = time.perf_counter() - start_time
    return yk, Fy, info, max_iter, ngev, nhev, elapsed_time, stepsize
