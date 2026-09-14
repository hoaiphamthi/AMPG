import numpy as np
import time
import quadprog

# -----------------------------------------------------------------------------
# Subproblems
# -----------------------------------------------------------------------------

def sfs_subproblem(n, m, x, JG, alpha):
    total_vars = 1 + n
    epsilon = 1e-8
    
    P_diag = np.zeros(total_vars)
    P_diag[0] = epsilon
    P_diag[1:] = 1.0 / alpha + epsilon
    G_qp = np.diag(P_diag).astype(np.float64)
    
    q = np.zeros(total_vars)
    q[0] = 1.0
    q[1:] = -x.flatten() / alpha
    a_qp = -q.astype(np.float64)
    
    ones_m = -np.ones((m, 1))
    G_1 = np.hstack([ones_m, JG])
    h_1 = JG @ x.flatten()
    
    zeros_n = np.zeros((n, 1))
    G_2 = np.hstack([zeros_n, -np.eye(n)])
    h_2 = np.zeros(n)
    
    G_final = np.vstack([G_1, G_2])
    h_final = np.concatenate([h_1, h_2])
    
    A_eq = np.zeros((1, total_vars))
    A_eq[0, 1:] = 1.0
    b_eq = np.array([1.0])
    
    C_T = np.vstack([A_eq, -G_final])
    b_qp = np.concatenate([b_eq, -h_final]).astype(np.float64)
    C_qp = C_T.T.astype(np.float64)
    
    try:
        sol = quadprog.solve_qp(G_qp, a_qp, C_qp, b_qp, meq=1)
        z_opt = sol[0]
        u_opt = z_opt[1:]
        return u_opt, 0
    except Exception as e:
        print(f"Error in 'sfs_subproblem': {e}")
        
    return None, -1


def sfs_subproblem_acc(n, m, x, y, alpha, JGy, Gy, Gx):
    total_vars = 1 + n
    epsilon = 1e-8
    
    P_diag = np.zeros(total_vars)
    P_diag[0] = epsilon
    P_diag[1:] = 1.0 / alpha + epsilon
    G_qp = np.diag(P_diag).astype(np.float64)
    
    q = np.zeros(total_vars)
    q[0] = 1.0
    q[1:] = -y.flatten() / alpha
    a_qp = -q.astype(np.float64)
    
    ones_m = -np.ones((m, 1))
    G_1 = np.hstack([ones_m, JGy])
    
    h_1 = JGy @ y.flatten() - Gy + Gx
    
    zeros_n = np.zeros((n, 1))
    G_2 = np.hstack([zeros_n, -np.eye(n)])
    h_2 = np.zeros(n)
    
    G_final = np.vstack([G_1, G_2])
    h_final = np.concatenate([h_1, h_2])
    
    A_eq = np.zeros((1, total_vars))
    A_eq[0, 1:] = 1.0
    b_eq = np.array([1.0])
    
    C_T = np.vstack([A_eq, -G_final])
    b_qp = np.concatenate([b_eq, -h_final]).astype(np.float64)
    C_qp = C_T.T.astype(np.float64)
    
    try:
        sol = quadprog.solve_qp(G_qp, a_qp, C_qp, b_qp, meq=1)
        z_opt = sol[0]
        u_opt = z_opt[1:]
        return u_opt, 0
    except Exception as e:
        print(f"Error in 'sfs_subproblem_acc': {e}")
        
    return None, -1

# -----------------------------------------------------------------------------
# Common Evaluation Functions for SFS
# -----------------------------------------------------------------------------
def _evalg(x, j, Q, rho):
    if j == 0:
        return (x.T @ Q @ x).item()
    if j == 1:
        return (-rho.T @ x).item()

def _evalgradg(x, j, Q, rho):
    if j == 0:
        return (2 * Q @ x).flatten()
    if j == 1:
        return (-rho).flatten()

# -----------------------------------------------------------------------------
# AMPG Methods
# -----------------------------------------------------------------------------
def _AMPG_core(n, m, x, Q, rho, opt, alpha, beta, c0=0.2, c1=0.19):
    start_time = time.perf_counter()

    # Initial parameters 
    tol = 1e-6
    max_iter = 1000
    info = 0

    # Counters
    ngev = 0
    nhev = 0
    tk_sum = 0.0

    G = np.zeros(m)
    JG_next = np.zeros((m, n))

    xk = x.copy()
    tk = 1.0

    JG = np.array([_evalgradg(xk, i, Q, rho) for i in range(m)])

    for iter in range(1, max_iter + 1):
        tk_sum += tk
        # Evaluate F
        for i in range(m):
            G[i] = _evalg(xk, i, Q, rho)
            
        ngev += m

        # Step 1: Compute x_{k+1} by solving the subproblem
        xk_next, info = sfs_subproblem(n, m, xk, JG, tk)

        # Check for an error while solving the subproblem
        if info != 0 or xk_next is None:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return xk, G, info, iter, ngev, nhev, elapsed_time, stepsize
            
        xk_next = xk_next.reshape(xk.shape)

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
            return xk_next, G, info, iter, ngev, nhev, elapsed_time, stepsize

        # Compute the Jacobian of G
        for i in range(m):
            JG_next[i, :] = _evalgradg(xk_next, i, Q, rho)

        grad_diff = JG_next - JG

        # AMPG1
        if opt == 1:
            val_compare = step_norm ** 2
            B_val = np.max(np.abs(grad_diff @ d.flatten()))

        # AMPG2
        else:
            val_compare = step_norm
            B_val = np.max(np.linalg.norm(grad_diff, axis=1))

        if B_val > c0 * val_compare / tk:
            tk = c1 * val_compare / B_val
        else:
            tk = (1 + alpha * (np.log(iter)) ** beta / (iter ** 1.1)) * tk

        xk = xk_next.copy()
        JG = JG_next.copy()


    info = 1
    elapsed_time = time.perf_counter() - start_time
    stepsize = tk_sum / max_iter
    return xk, G, info, max_iter, ngev, nhev, elapsed_time, stepsize


def AMPG1_sfs(n, m, x, Q, rho, alpha=1.5, beta=6, c0=0.2, c1=0.19):
    return _AMPG_core(n, m, x, Q, rho, opt=1, 
                      alpha=alpha, beta=beta, c0=c0, c1=c1)

def AMPG2_sfs(n, m, x, Q, rho, alpha=1.5, beta=6, c0=0.2, c1=0.19):
    return _AMPG_core(n, m, x, Q, rho, opt=2,
                      alpha=alpha, beta=beta, c0=c0, c1=c1)

def AMPG1_sfs_sk(n, m, x, Q, rho, alpha=1.5, beta=6, c0=0.15, c1=0.08):
    return _AMPG_core(n, m, x, Q, rho, opt=1, 
                      alpha=alpha, beta=beta, c0=c0, c1=c1)

def AMPG2_sfs_sk(n, m, x, Q, rho, alpha=1.5, beta=6, c0=0.2, c1=0.11):
    return _AMPG_core(n, m, x, Q, rho, opt=2,
                      alpha=alpha, beta=beta, c0=c0, c1=c1)


# -----------------------------------------------------------------------------
# MPGE Method and Line Search Procedures
# -----------------------------------------------------------------------------

def explicit_LS_one(n, x, d, t_trial, jk, G_jk, JGd_jk, Q, rho):
    # Parameters
    gamma = 1.9999
    t_min = 1e-15
    sigma1 = 0.5e-1
    sigma2 = 9.5e-1

    # Counters:
    ngev = 0

    # Define gtest
    gtest = JGd_jk + gamma * (np.linalg.norm(d)**2) / 2.0

    # Evaluate g
    g = _evalg(x + t_trial * d, jk, Q, rho)
    ngev += 1

    while True:
        # Test the descent condition for f_jk
        if g <= G_jk + t_trial * gtest: 
            return t_trial, g, ngev, 0
        
        if t_trial <= t_min:
            return t_trial, g, ngev, 4
        
        if JGd_jk < 0:
            t_q = - (JGd_jk * t_trial ** 2) / 2.0 / (g - G_jk - JGd_jk * t_trial)

            if sigma1 * t_trial <= t_q <= sigma2 * t_trial:
                t_trial = t_q
            else: 
                t_trial = t_trial / 2.0
        else:
            t_trial = t_trial / 2.0

        g = _evalg(x + t_trial * d, jk, Q, rho)
        ngev += 1

     
def explicit_LS(n, m, x, d, t_trial, jk_first, G, JGd, Q, rho):
    # Parameters
    gamma = 1.9999

    # Counters
    ngev = 0

    # Define G_test
    G_test = JGd + gamma * (np.linalg.norm(d)**2) / 2.0

    Gindex = list(range(m))
    Gindex[0] = jk_first
    Gindex[jk_first] = 0

    G_trial = np.zeros(m)

    # Test the sufficient descent condition (sdc) at t_trial = 1
    sdc = True
    iA = -1 
    g_trial_iA = 0.0

    for j in range(m):
        i = Gindex[j]

        G_trial[i] = _evalg(x + t_trial * d, i, Q, rho)
        ngev += 1

        if G_trial[i] > G[i] + t_trial * G_test[i]:
            sdc = False
            iA = i 
            g_trial_iA = G_trial[i] 
            break

    if sdc:
        return t_trial, G_trial, ngev, 0
    
    t_trial, G_trial, ngevbt, info = backtrackingMO(t_trial, n, m, x, d, G, JGd, iA, g_trial_iA, G_test, Q, rho)
    ngev += ngevbt

    return t_trial, G_trial, ngev, info


def backtrackingMO(t_trial, n, m, x, d, G, JGd, iA, g_trial_iA, G_test, Q, rho):
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
            ind = iA 
            g_trial = g_trial_iA 

        elif info == 0: 
            sdc = True

            for i in range(m):
                if i == ind:
                    continue
                    
                g_trial = _evalg(x + t_trial * d, i, Q, rho)
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
                
            g_trial = _evalg(x + t_trial * d, ind, Q, rho)
            ngev += 1
            
        Gend[ind] = g_trial


def MPGE_sfs(n, m, x, Q, rho, alpha=1.0):
    start_time = time.perf_counter()

    # Initial parameters 
    tol = 1e-6
    itermax = 1000

    # Counters
    ngev = 0 
    nhev = 0 # Included for compatibility in return variables
    iter = 0

    tk_sum = 0.0

    G = np.zeros(m)
    JG = np.zeros((m, n))

    # Evaluate G (For SFS, H = 0, so F = G)
    for i in range(m):
        G[i] = _evalg(x, i, Q, rho)  

    ngev += m
    supnorm_x_p = 1.0
    xprev = np.copy(x)

    while True:
        iter += 1

        # Test the maximum number of iterations
        if iter > itermax:
            info = 1
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / itermax
            return x, G, info, itermax, ngev, nhev, elapsed_time, stepsize
        
        # Compute the Jacobian of G
        for i in range(m):
            JG[i, :] = _evalgradg(x, i, Q, rho)
        
        # Solve the subproblem
        p, info = sfs_subproblem(n, m, x, JG, alpha)

        # Check for an error while solving the subproblem
        if info != 0 or p is None:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return x, G, info, iter, ngev, nhev, elapsed_time, stepsize
        
        p = p.reshape(x.shape)

        # Compute linesearch
        d = p - x

        #  ----------------------------------------------------------
        # Stopping criteria
        #  ----------------------------------------------------------

        # Compute norm(x-xprev,inf)/max(1,norm(xprev,inf))
        if iter > 1:
            supnorm_x_p = np.linalg.norm(x - xprev, np.inf) / max(1.0, np.linalg.norm(xprev, np.inf))
        
        # Compute norm (d)
        step_norm = np.linalg.norm(d)

        # Test optimality
        if step_norm <= tol or supnorm_x_p <= tol:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter 
            return x, G, info, iter, ngev, nhev, elapsed_time, stepsize
                    
        #  ----------------------------------------------------------
        # Iterate
        #  ----------------------------------------------------------
        xprev = np.copy(x)

        t_trial = 1.0
        JGd = JG @ d.flatten()

        # Step 3: jk == jk*
        jk = np.argmax(JGd)

        # Step 3.1
        t_trial, Gtrial_jk, ngevLS, info = explicit_LS_one(n, x, d, t_trial, jk, G[jk], JGd[jk], Q, rho)
        ngev += ngevLS

        # Check for an error in the line search procedure
        if info != 0:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return x, G, info, iter, ngev, nhev, elapsed_time, stepsize
        
        # Update x_trial, G_trial
        x_trial = x + t_trial * d
        G_trial = np.zeros(m)
        G_trial[jk] = Gtrial_jk
        
        Fdec = True     # Flag for checking F descent condition (using G)
        jk_first = -1

        for i in range(m):
            if i == jk:
                continue

            G_trial[i] = _evalg(x_trial, i, Q, rho)
            ngev += 1

            if G_trial[i] > G[i]:
                Fdec = False
                jk_first = i 
                break
                
        # Step 4:
        if Fdec:
            # Update x
            x = x_trial.copy()
            G = G_trial.copy()
            tk_sum += t_trial
           
        # Step 3.3
        else:
            t_trial, G, ngevLS, info = explicit_LS(n, m, x, d, t_trial, jk_first, G, JGd, Q, rho)
            ngev += ngevLS

            #  Check for an error in the line seach procedure
            if info != 0:
                elapsed_time = time.perf_counter() - start_time
                stepsize = tk_sum / iter
                return x, G, info, iter, ngev, nhev, elapsed_time, stepsize
            
            # Update x
            x = x + t_trial * d
            tk_sum += t_trial
            

# -----------------------------------------------------------------------------
# PGM and accPGM Methods
# -----------------------------------------------------------------------------

def PGM_sfs(n, m, xk, Q, rho):
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
    JG = np.zeros((m, n))
    JG_next = np.zeros((m, n))
    diff = 0
    
    alpha = sigma

    xk_next = np.zeros(n)
    
    for i in range(m):
        JG[i, :] = _evalgradg(xk, i, Q, rho)

    for iter in range(1, max_iter + 1):
        # Compute G
        for i in range(m):
            G[i] = _evalg(xk, i, Q, rho)
            
        ngev += m

        # Solve the subproblem: line search procedure in Step 3
        while True:
            xk_next, info = sfs_subproblem(n, m, xk, JG, alpha)
            
            # Check for an error while solving subproblem.
            if info != 0 or xk_next is None:
                elapsed_time = time.perf_counter() - start_time
                stepsize = tk_sum / iter
                return xk, G, info, iter, ngev, nhev, elapsed_time, stepsize
            
            xk_next = xk_next.reshape(xk.shape)

            for i in range(m):
                JG_next[i, :] = _evalgradg(xk_next, i, Q, rho)

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

        # Compute norm(x-xprev,inf)/max(1,norm(xprev,inf))
        supnorm_x_xk = np.linalg.norm(d, np.inf) / max(1.0, np.linalg.norm(xk, np.inf))

        # Test optimality
        if supnorm_x_xk <= tol or step_norm <= tol:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return xk_next, G, info, iter, ngev, nhev, elapsed_time, stepsize
            
        xk = xk_next.copy()
        JG = JG_next.copy()

    info = 1
    elapsed_time = time.perf_counter() - start_time
    stepsize = tk_sum / max_iter
    return xk, G, info, max_iter, ngev, nhev, elapsed_time, stepsize



def accPGM_sfs(n, m, xk, Q, rho):
    start_time = time.perf_counter()

    # Parameters:
    eta = 0.5
    delta = 0.3
    sigma = 1

    tol = 1e-6
    max_iter = 500

    yk = xk.copy()
    xk_next = np.zeros(n)
    bk = 1

    alpha = sigma
    tk_sum = 0.0
    
    # counters:
    ngev = 0
    nhev = 0
    
    Gx = np.zeros(m)
    Gy = np.zeros(m)
    JGy = np.zeros((m, n))
    JGx_next = np.zeros((m, n))

    for iter in range(1, max_iter + 1):
        
        # ---- Compute Gy
        for i in range(m):
            Gy[i] = _evalg(yk, i, Q, rho)
            JGy[i, :] = _evalgradg(yk, i, Q, rho)
           
        ngev += m

        # Compute Gx 
        if iter == 1:
            Gx = Gy.copy()
        else:
            for i in range(m):
                Gx[i] = _evalg(xk, i, Q, rho)
            ngev += m
        
        # Solves the subproblem: Line search procedure A in Step 3
        while True:
            xk_next, info = sfs_subproblem_acc(n, m, xk, yk, alpha, JGy, Gy, Gx)

            if info != 0 or xk_next is None:
                stepsize = tk_sum / iter
                elapsed_time = time.perf_counter() - start_time
                return yk, Gy, info, iter, ngev, nhev, elapsed_time, stepsize
            
            xk_next = xk_next.reshape(xk.shape)

            for i in range(m):
                JGx_next[i, :] = _evalgradg(xk_next, i, Q, rho)

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

        # Compute norm(x-xprev,inf)/max(1,norm(xprev,inf))
        supnorm_yk_xk_next = np.linalg.norm(d, np.inf) / max(1.0, np.linalg.norm(yk, np.inf))

        # Test optimality
        if supnorm_yk_xk_next <= tol or step_norm <= tol:
            elapsed_time = time.perf_counter() - start_time
            stepsize = tk_sum / iter
            return yk, Gy, info, iter, ngev, nhev, elapsed_time, stepsize
        
        bk_next = (1 + np.sqrt(1 + 4 * bk ** 2)) / 2
        yk = xk_next + (bk - 1) / bk_next * (xk_next - xk)
        xk = xk_next.copy()
        bk = bk_next

            
    info = 1
    stepsize = tk_sum / max_iter
    elapsed_time = time.perf_counter() - start_time
    return yk, Gy, info, max_iter, ngev, nhev, elapsed_time, stepsize
