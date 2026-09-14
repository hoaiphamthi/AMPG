import numpy as np

def evalgradg(n, x, ind, problem_name):
    """
    Function to evaluate gradient of G
    """
    # Initialize gradient vector (0-based)
    g = np.zeros(n)
    
    # ----------------------------------------------------------------------
    # AP1
    if problem_name == 'AP1':
        if ind == 0:
            g[0] = (x[0] - 1.0) ** 3
            g[1] = 2.0 * (x[1] - 2.0) ** 3
            return g
        if ind == 1:
            g[0] = 0.5 * np.exp((x[0] + x[1]) / 2.0) + 2.0 * x[0]
            g[1] = 0.5 * np.exp((x[0] + x[1]) / 2.0) + 2.0 * x[1]
            return g
        if ind == 2:
            g[0] = -(1.0 / 6.0) * np.exp(-x[0])
            g[1] = -(1.0 / 3.0) * np.exp(-x[1])
            return g

    # ----------------------------------------------------------------------
    # AP2
    if problem_name == 'AP2':
        if ind == 0:
            g[0] = 2.0 * x[0]
            return g
        if ind == 1:
            g[0] = 2.0 * (x[0] - 1.0)
            return g

    # ----------------------------------------------------------------------
    # AP3
    if problem_name == 'AP3':
        if ind == 0:
            g[0] = (x[0] - 1.0) ** 3
            g[1] = 2.0 * (x[1] - 2.0) ** 3
            return g
        if ind == 1:
            g[0] = -4.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
            g[1] = 2.0 * (x[1] - x[0] ** 2)
            return g

    # ----------------------------------------------------------------------
    # AP4
    if problem_name == 'AP4':
        if ind == 0:
            g[0] = (4.0 / 9.0) * (x[0] - 1.0) ** 3
            g[1] = (8.0 / 9.0) * (x[1] - 2.0) ** 3
            g[2] = (12.0 / 9.0) * (x[2] - 3.0) ** 3
            return g
        if ind == 1:
            common_term = (1.0 / 3.0) * np.exp((x[0] + x[1] + x[2]) / 3.0)
            g[0] = common_term + 2.0 * x[0]
            g[1] = common_term + 2.0 * x[1]
            g[2] = common_term + 2.0 * x[2]
            return g
        if ind == 2:
            g[0] = -0.25 * np.exp(-x[0])
            g[1] = -(1.0 / 3.0) * np.exp(-x[1])
            g[2] = -0.25 * np.exp(-x[2])
            return g

    # ----------------------------------------------------------------------
    # BK1
    if problem_name == 'BK1':
        if ind == 0:
            g = 2.0 * x
            return g
        if ind == 1:
            g = 2.0 * (x - 5.0)
            return g

    # ----------------------------------------------------------------------
    # DD1
    if problem_name == 'DD1':
        if ind == 0:
            g = 2.0 * x
            return g
        if ind == 1:
            g[0] = 3.0
            g[1] = 2.0
            g[2] = -1.0 / 3.0
            g[3] = 3.0e-2 * (x[3] - x[4]) ** 2
            g[4] = -3.0e-2 * (x[3] - x[4]) ** 2
            return g

    # ----------------------------------------------------------------------
    # DGO1
    if problem_name == 'DGO1':
        if ind == 0:
            g[0] = np.cos(x[0])
            return g
        if ind == 1:
            g[0] = np.cos(x[0] + 0.7)
            return g

    # ----------------------------------------------------------------------
    # DGO2
    if problem_name == 'DGO2':
        if ind == 0:
            g[0] = 2.0 * x[0]
            return g
        if ind == 1:
            g[0] = x[0] / np.sqrt(81.0 - x[0] ** 2)
            return g

    # ----------------------------------------------------------------------
    # DTLZ1
    if problem_name == 'DTLZ1':
        k = 5
        m = 3
        m_py = m - 1
        
        x_subset = x[m_py:]
        term_cos = np.cos(20.0 * np.pi * (x_subset - 0.5))
        term_sin = np.sin(20.0 * np.pi * (x_subset - 0.5))
        
        faux = 100.0 * (k + np.sum((x_subset - 0.5) ** 2 - term_cos))
        
        f_base = 0.5 * (1.0 + faux)
        prod_x = np.prod(x[0 : m - ind - 1])
        
        if ind > 0:
            term_1_minus_x = (1.0 - x[m - ind - 1])
            f_base *= term_1_minus_x
        else:
            term_1_minus_x = 1.0
        
        f_val = f_base * prod_x
        
        # Derivative w.r.t x[i] (i < m-ind-1)
        for i in range(m - ind - 1):
            if x[i] == 0:
                g[i] = 0.5 * (1.0 + faux) * term_1_minus_x * np.prod(np.delete(x[0:m-ind-1], i))
            else:
                g[i] = f_val / x[i]
        
        # Derivative w.r.t x[m-ind-1]
        if ind > 0:
            g[m - ind - 1] = -0.5 * (1.0 + faux) * prod_x
            
        # Derivative w.r.t x[i] (i >= m)
        dfaux_dx = 100.0 * (2.0 * (x_subset - 0.5) + 20.0 * np.pi * term_sin)
        
        faux_g = 0.5 * prod_x
        if ind > 0:
            faux_g *= term_1_minus_x
            
        g[m_py:] = faux_g * dfaux_dx
        
        return g

    # ----------------------------------------------------------------------
    # DTLZ2
    if problem_name == 'DTLZ2':
        k = 5
        m = 3
        m_py = m - 1
        
        x_subset = x[m_py:]
        faux = 1.0 + np.sum((x_subset - 0.5) ** 2)
        
        f_val = 1.0 * faux
        
        # Calculate f(x)
        cos_vals = np.cos(x[0 : m - ind - 1] * np.pi / 2.0)
        prod_cos = np.prod(cos_vals)
        f_val *= prod_cos
        
        if ind > 0:
            sin_val = np.sin(x[m - ind - 1] * np.pi / 2.0)
            f_val *= sin_val
        else:
            sin_val = 1.0
            
        # Derivative w.r.t x[i] (i < m-ind)
        for i in range(m - ind - 1):
            if cos_vals[i] == 0:
                g[i] = 0 # Already at minimum/maximum
            else:
                g[i] = f_val / cos_vals[i] * (-np.pi / 2.0) * np.sin(x[i] * np.pi / 2.0)
                
        # Derivative w.r.t x[m-ind+1] (Python: x[m-ind])
        if ind > 0:
            g[m - ind - 1] = (f_val / sin_val) * (np.pi / 2.0) * np.cos(x[m - ind - 1] * np.pi / 2.0)
            
        # Derivative w.r.t x[i] (i >= m)
        faux_g = prod_cos
        if ind > 0:
            faux_g *= sin_val
            
        dfaux_dx = 2.0 * (x_subset - 0.5)
        g[m_py:] = faux_g * dfaux_dx
        
        return g

    # ----------------------------------------------------------------------
    # DTLZ3
    if problem_name == 'DTLZ3':
        k = 5
        m = 3
        m_py = m - 1
        
        x_subset = x[m_py:]
        term_cos = np.cos(20.0 * np.pi * (x_subset - 0.5))
        term_sin = np.sin(20.0 * np.pi * (x_subset - 0.5))
        
        faux = 1.0 + 100.0 * (k + np.sum((x_subset - 0.5) ** 2 - term_cos))
        
        f_val = 1.0 * faux
        
        cos_vals = np.cos(x[0 : m - ind - 1] * np.pi / 2.0)
        prod_cos = np.prod(cos_vals)
        f_val *= prod_cos
        
        if ind > 0:
            sin_val = np.sin(x[m - ind - 1] * np.pi / 2.0)
            f_val *= sin_val
        else:
            sin_val = 1.0
            
        # Derivative w.r.t x[i] (i < m-ind)
        for i in range(m - ind - 1):
            if cos_vals[i] == 0:
                g[i] = 0
            else:
                g[i] = f_val / cos_vals[i] * (-np.pi / 2.0) * np.sin(x[i] * np.pi / 2.0)
                
        # Derivative w.r.t x[m-ind+1] (Python: x[m-ind])
        if ind > 0:
            g[m - ind - 1] = (f_val / sin_val) * (np.pi / 2.0) * np.cos(x[m - ind - 1] * np.pi / 2.0)
            
        # Derivative w.r.t x[i] (i >= m)
        faux_g = prod_cos
        if ind > 0:
            faux_g *= sin_val
            
        dfaux_dx = 100.0 * (2.0 * (x_subset - 0.5) + 20.0 * np.pi * term_sin)
        g[m_py:] = faux_g * dfaux_dx
        
        return g

    # ----------------------------------------------------------------------
    # DTLZ4
    if problem_name == 'DTLZ4':
        alpha = 2.0
        k = 5
        m = 3
        m_py = m - 1
        
        x_subset = x[m_py:]
        faux = 1.0 + np.sum((x_subset - 0.5) ** 2)
        
        f_val = 1.0 * faux
        
        x_alpha = x[0 : m - ind - 1] ** alpha
        cos_vals = np.cos(x_alpha * np.pi / 2.0)
        prod_cos = np.prod(cos_vals)
        f_val *= prod_cos
        
        if ind > 0:
            sin_val = np.sin(x[m - ind - 1] ** alpha * np.pi / 2.0)
            f_val *= sin_val
        else:
            sin_val = 1.0
            
        # Derivative w.r.t x[i] (i < m-ind)
        for i in range(m - ind - 1):
            if cos_vals[i] == 0:
                g[i] = 0
            else:
                g[i] = (f_val / cos_vals[i] * (-np.pi / 2.0) * alpha * x[i]**(alpha - 1.0) * np.sin(x_alpha[i] * np.pi / 2.0))
                
        # Derivative w.r.t x[m-ind+1] (Python: x[m-ind])
        if ind > 0:
            i = m - ind - 1
            g[i] = (f_val / sin_val) * (np.pi / 2.0) * alpha * x[i]**(alpha - 1.0) * np.cos(x[i]**alpha * np.pi / 2.0)
            
        # Derivative w.r.t x[i] (i >= m)
        faux_g = prod_cos
        if ind > 0:
            faux_g *= sin_val
            
        dfaux_dx = 2.0 * (x_subset - 0.5)
        g[m_py:] = faux_g * dfaux_dx
        
        return g

    # ----------------------------------------------------------------------
    # FA1
    if problem_name == 'FA1':
        if ind == 0:
            g[0] = 4.0 * np.exp(-4.0 * x[0]) / (1.0 - np.exp(-4.0))
            return g
        if ind == 1:
            a = np.exp(-4.0 * x[0])
            b = 1.0 - np.exp(-4.0)
            t = (1.0 - a) / (b * (x[1] + 1.0))
            g[0] = -2.0 * a / b * (t ** (-0.5))
            g[1] = 1.0 - 0.5 * (t ** 0.5)
            return g
        if ind == 2:
            a = np.exp(-4.0 * x[0])
            b = 1.0 - np.exp(-4.0)
            t = (1.0 - a) / (b * (x[2] + 1.0))
            g[0] = -0.4 * (t ** (-0.9)) * a / b
            g[2] = 1.0 - 0.9 * (t ** 0.1)
            return g

    # ----------------------------------------------------------------------
    # Far1
    if problem_name == 'Far1':
        if ind == 0:
            g[0] = (60.0 * (x[0] - 0.1) * np.exp(15.0 * (-(x[0] - 0.1) ** 2 - x[1] ** 2))
                   + 40.0 * (x[0] - 0.6) * np.exp(20.0 * (-(x[0] - 0.6) ** 2 - (x[1] - 0.6) ** 2))
                   - 40.0 * (x[0] + 0.6) * np.exp(20.0 * (-(x[0] + 0.6) ** 2 - (x[1] - 0.6) ** 2))
                   - 40.0 * (x[0] - 0.6) * np.exp(20.0 * (-(x[0] - 0.6) ** 2 - (x[1] + 0.6) ** 2))
                   - 40.0 * (x[0] + 0.6) * np.exp(20.0 * (-(x[0] + 0.6) ** 2 - (x[1] + 0.6) ** 2)))
            g[1] = (60.0 * x[1] * np.exp(15.0 * (-(x[0] - 0.1) ** 2 - x[1] ** 2))
                   + 40.0 * (x[1] - 0.6) * np.exp(20.0 * (-(x[0] - 0.6) ** 2 - (x[1] - 0.6) ** 2))
                   - 40.0 * (x[1] - 0.6) * np.exp(20.0 * (-(x[0] + 0.6) ** 2 - (x[1] - 0.6) ** 2))
                   - 40.0 * (x[1] + 0.6) * np.exp(20.0 * (-(x[0] - 0.6) ** 2 - (x[1] + 0.6) ** 2))
                   - 40.0 * (x[1] + 0.6) * np.exp(20.0 * (-(x[0] + 0.6) ** 2 - (x[1] + 0.6) ** 2)))
            return g
        if ind == 1:
            g[0] = (- 80.0 * x[0] * np.exp(20.0 * (-x[0] ** 2 - x[1] ** 2))
                   - 40.0 * (x[0] - 0.4) * np.exp(20.0 * (-(x[0] - 0.4) ** 2 - (x[1] - 0.6) ** 2))
                   + 40.0 * (x[0] + 0.5) * np.exp(20.0 * (-(x[0] + 0.5) ** 2 - (x[1] - 0.7) ** 2))
                   + 40.0 * (x[0] - 0.5) * np.exp(20.0 * (-(x[0] - 0.5) ** 2 - (x[1] + 0.7) ** 2))
                   - 40.0 * (x[0] + 0.4) * np.exp(20.0 * (-(x[0] + 0.4) ** 2 - (x[1] + 0.8) ** 2)))
            g[1] = (- 80.0 * x[1] * np.exp(20.0 * (-x[0] ** 2 - x[1] ** 2))
                   - 40.0 * (x[1] - 0.6) * np.exp(20.0 * (-(x[0] - 0.4) ** 2 - (x[1] - 0.6) ** 2))
                   + 40.0 * (x[1] - 0.7) * np.exp(20.0 * (-(x[0] + 0.5) ** 2 - (x[1] - 0.7) ** 2))
                   + 40.0 * (x[1] + 0.7) * np.exp(20.0 * (-(x[0] - 0.5) ** 2 - (x[1] + 0.7) ** 2))
                   - 40.0 * (x[1] + 0.8) * np.exp(20.0 * (-(x[0] + 0.4) ** 2 - (x[1] + 0.8) ** 2)))
            return g

    # ----------------------------------------------------------------------
    # FDS
    if problem_name == 'FDS':
        i_matlab = np.arange(1, n + 1)
        if ind == 0:
            g = 4.0 * i_matlab * (x - i_matlab) ** 3 / (n ** 2)
            return g
        if ind == 1:
            g = np.exp(np.sum(x) / n) / n + 2.0 * x
            return g
        if ind == 2:
            g = -i_matlab * (n - i_matlab + 1.0) * np.exp(-x) / (n * (n + 1.0))
            return g

    # ----------------------------------------------------------------------
    # FF1
    if problem_name == 'FF1':
        if ind == 0:
            common_exp = np.exp(-(x[0] - 1.0) ** 2 - (x[1] + 1.0) ** 2)
            g[0] = 2.0 * (x[0] - 1.0) * common_exp
            g[1] = 2.0 * (x[1] + 1.0) * common_exp
            return g
        if ind == 1:
            common_exp = np.exp(-(x[0] + 1.0) ** 2 - (x[1] - 1.0) ** 2)
            g[0] = 2.0 * (x[0] + 1.0) * common_exp
            g[1] = 2.0 * (x[1] - 1.0) * common_exp
            return g

    # ----------------------------------------------------------------------
    # Hil1
    if problem_name == 'Hil1':
        a = (2.0 * np.pi / 360.0) * (45.0 + 40.0 * np.sin(2.0 * np.pi * x[0])
                                    + 25.0 * np.sin(2.0 * np.pi * x[1]))
        b = 1.0 + 0.5 * np.cos(2.0 * np.pi * x[0])
        pi2 = 2.0 * np.pi
        
        dadx0 = (2.0 * np.pi / 360.0) * 40.0 * np.cos(pi2 * x[0]) * pi2
        dadx1 = (2.0 * np.pi / 360.0) * 25.0 * np.cos(pi2 * x[1]) * pi2
        dbdx0 = -0.5 * np.sin(pi2 * x[0]) * pi2
        
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        
        if ind == 0: # f = cos(a) * b
            g[0] = -sin_a * dadx0 * b + cos_a * dbdx0
            g[1] = -sin_a * dadx1 * b
            return g
        if ind == 1: # f = sin(a) * b
            g[0] = cos_a * dadx0 * b + sin_a * dbdx0
            g[1] = cos_a * dadx1 * b
            return g

    # ----------------------------------------------------------------------
    # IKK1
    if problem_name == 'IKK1':
        if ind == 0:
            g[0] = 2.0 * x[0]
            return g
        if ind == 1:
            g[0] = 2.0 * (x[0] - 20.0)
            return g
        if ind == 2:
            g[1] = 2.0 * x[1]
            return g

    # ----------------------------------------------------------------------
    # IM1
    if problem_name == 'IM1':
        if ind == 0:
            g[0] = 1.0 / np.sqrt(x[0])
            return g
        if ind == 1:
            g[0] = 1.0 - x[1]
            g[1] = -x[0]
            return g

    # ----------------------------------------------------------------------
    # JOS1
    if problem_name == 'JOS1':
        if ind == 0:
            g = 2.0 * x / n
            return g
        if ind == 1:
            g = 2.0 * (x - 2.0) / n
            return g

    # ----------------------------------------------------------------------
    # JOS4
    if problem_name == 'JOS4':
        if ind == 0:
            g[0] = 1.0
            return g
        if ind == 1:
            faux = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
            t = x[0] / faux
            g[0] = -0.25 * (t ** (-0.75)) - 4.0 * (t ** 3)
            g[1:] = (9.0 / (n - 1)) * (1.0 - 0.75 * (t ** 0.25) + 3.0 * (t ** 4.0))
            return g

    # ----------------------------------------------------------------------
    # KW2
    if problem_name == 'KW2':
        exp1 = np.exp(-x[0]**2 - (x[1] + 1.0)**2)
        exp2 = np.exp(-x[0]**2 - x[1]**2)
        exp3 = np.exp(-(x[0] + 2.0)**2 - x[1]**2)
        exp4 = np.exp(-x[1]**2 - (1.0 - x[0])**2)
        exp5 = np.exp(-(2.0 - x[1])**2 - x[0]**2)
        
        if ind == 0:
            g[0] = (6.0 * (1.0 - x[0]) * exp1 + 
                    6.0 * (1.0 - x[0])**2 * exp1 * x[0] + 
                    10.0 * (1.0 / 5.0 - 3.0 * x[0]**2) * exp2 - 
                    20.0 * (x[0] / 5.0 - x[0]**3 - x[1]**5) * exp2 * x[0] - 
                    6.0 * exp3 * (x[0] + 2.0) - 1.0)
            g[1] = (6.0 * (1.0 - x[0])**2 * exp1 * (x[1] + 1.0) - 
                    50.0 * x[1]**4 * exp2 - 
                    10.0 * (x[0] / 5.0 - x[0]**3 - x[1]**5) * exp2 * 2.0 * x[1] - 
                    6.0 * exp3 * x[1] - 0.5)
            return g
        if ind == 1:
            g[0] = (-6.0 * (1.0 + x[1])**2 * exp4 * (1.0 - x[0]) + 
                    50.0 * x[0]**4 * exp2 - 
                    20.0 * (-x[1] / 5.0 + x[1]**3 + x[0]**5) * exp2 * x[0] - 
                    6.0 * exp5 * x[0])
            g[1] = (-6.0 * (1.0 + x[1]) * exp4 + 
                    6.0 * (1.0 + x[1])**2 * exp4 * x[1] + 
                    10.0 * (-1.0 / 5.0 + 3.0 * x[1]**2) * exp2 - 
                    20.0 * (-x[1] / 5.0 + x[1]**3 + x[0]**5) * exp2 * x[1] + 
                    6.0 * exp5 * (2.0 - x[1]))
            return g

    # ----------------------------------------------------------------------
    # LE1
    if problem_name == 'LE1':
        if ind == 0:
            t = 0.25 * (x[0] ** 2 + x[1] ** 2) ** (-0.875)
            g[0] = x[0] * t
            g[1] = x[1] * t
            return g
        if ind == 1:
            t = 0.5 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) ** (-0.75)
            g[0] = (x[0] - 0.5) * t
            g[1] = (x[1] - 0.5) * t
            return g

    # ----------------------------------------------------------------------
    # Lov1
    if problem_name == 'Lov1':
        if ind == 0:
            g[0] = 2.1 * x[0]
            g[1] = 2.0 * 0.98 * x[1]
            return g
        if ind == 1:
            g[0] = 2.0 * 0.99 * (x[0] - 3.0)
            g[1] = 2.0 * 1.03 * (x[1] - 2.5)
            return g

    # ----------------------------------------------------------------------
    # Lov2
    if problem_name == 'Lov2':
        if ind == 0:
            g[1] = 1.0
            return g
        if ind == 1:
            g[0] = (-3.0 * x[0] ** 2 * (x[0] + 1.0) - (x[1] - x[0] ** 3))
            g[0] = -g[0] / (x[0] + 1.0) ** 2
            g[1] = -1.0 / (x[0] + 1.0)
            return g

    # ----------------------------------------------------------------------
    # Lov3
    if problem_name == 'Lov3':
        if ind == 0:
            g[0] = 2.0 * x[0]
            g[1] = 2.0 * x[1]
            return g
        if ind == 1:
            g[0] = 2.0 * (x[0] - 6.0)
            g[1] = -2.0 * (x[1] + 0.3)
            return g

    # ----------------------------------------------------------------------
    # Lov4
    if problem_name == 'Lov4':
        if ind == 0:
            exp1 = np.exp(-(x[0] + 2.0) ** 2 - x[1] ** 2)
            exp2 = np.exp(-(x[0] - 2.0) ** 2 - x[1] ** 2)
            g[0] = 2.0 * x[0] - 8.0 * ((x[0] + 2.0) * exp1 + (x[0] - 2.0) * exp2)
            g[1] = 2.0 * x[1] - 8.0 * (x[1] * exp1 + x[1] * exp2)
            return g
        if ind == 1:
            g[0] = 2.0 * (x[0] - 6.0)
            g[1] = 2.0 * (x[1] + 0.5)
            return g

    # ----------------------------------------------------------------------
    # Lov5
    if problem_name == 'Lov5':
        MM = np.array([[-1.0, -0.03, 0.011],
                       [-0.03, -1.0, 0.07],
                       [0.011, 0.07, -1.01]])
        p1 = np.array([x[0], x[1] - 0.15, x[2]])
        a1 = 0.35
        A1 = np.sqrt(2.0 * np.pi / a1) * np.exp(np.dot(p1, MM @ p1) / (a1 ** 2))
        
        p2 = np.array([x[0], x[1] + 1.1, 0.5 * x[2]])
        a2 = 3.0
        A2 = np.sqrt(2.0 * np.pi / a2) * np.exp(np.dot(p2, MM @ p2) / (a2 ** 2))
        
        sqrt2_2 = np.sqrt(2.0) / 2.0
        
        grad_A1_p1 = A1 * (2.0 / (a1**2)) * (MM @ p1)
        grad_A2_p2 = A2 * (2.0 / (a2**2)) * (MM @ p2)
        
        # d_p1 / d_x = [1, 0, 0; 0, 1, 0; 0, 0, 1]
        # d_p2 / d_x = [1, 0, 0; 0, 1, 0; 0, 0, 0.5]
        
        grad_A1 = grad_A1_p1 # * [1,1,1]
        grad_A2 = grad_A2_p2 * np.array([1, 1, 0.5])
        
        grad_faux = grad_A1 + grad_A2
        
        if ind == 0: # f = -(sqrt2_2 * (x[0] + faux))
            g[0] = -sqrt2_2 * (1.0 + grad_faux[0])
            g[1] = -sqrt2_2 * grad_faux[1]
            g[2] = -sqrt2_2 * grad_faux[2]
            return g
        if ind == 1: # f = -(sqrt2_2 * (-x[0] + faux))
            g[0] = -sqrt2_2 * (-1.0 + grad_faux[0])
            g[1] = -sqrt2_2 * grad_faux[1]
            g[2] = -sqrt2_2 * grad_faux[2]
            return g

    # ----------------------------------------------------------------------
    # Lov6
    if problem_name == 'Lov6':
        if ind == 0:
            g[0] = 1.0
            return g
        if ind == 1:
            g[0] = (-0.5 / np.sqrt(x[0]) - np.sin(10.0 * np.pi * x[0]) - 
                   10.0 * np.pi * x[0] * np.cos(10.0 * np.pi * x[0]))
            g[1:] = 2.0 * x[1:]
            return g

    # ----------------------------------------------------------------------
    # LTDZ
    if problem_name == 'LTDZ':
        c1 = np.cos(x[0] * np.pi / 2.0)
        c2 = np.cos(x[1] * np.pi / 2.0)
        s1 = np.sin(x[0] * np.pi / 2.0)
        s2 = np.sin(x[1] * np.pi / 2.0)
        piby2 = np.pi / 2.0
        term_x3 = 1.0 + x[2]
        
        if ind == 0: # f = -(3.0 - term_x3 * c1 * c2)
            g[0] = -( -term_x3 * (-s1 * piby2) * c2 )
            g[1] = -( -term_x3 * c1 * (-s2 * piby2) )
            g[2] = - ( -c1 * c2 )
            return g
        if ind == 1: # f = -(3.0 - term_x3 * c1 * s2)
            g[0] = -( -term_x3 * (-s1 * piby2) * s2 )
            g[1] = -( -term_x3 * c1 * (c2 * piby2) )
            g[2] = - ( -c1 * s2 )
            return g
        if ind == 2: # f = -(3.0 - term_x3 * s1) # Original MATLAB has error (used s1 * s1 instead of c1 * s1)
            # Based on evalg, f3 = 3 - (1+x3)*s1
            g[0] = -( -term_x3 * (c1 * piby2) )
            g[1] = 0.0
            g[2] = - ( -s1 )
            return g

    # ----------------------------------------------------------------------
    # MGH9
    if problem_name == 'MGH9':
        t = (7.0 - ind) / 2.0
        exp_term = np.exp(-x[1] * (t - x[2]) ** 2 / 2.0)
        g[0] = exp_term
        g[1] = -x[0] * exp_term * (t - x[2]) ** 2 / 2.0
        g[2] = x[0] * exp_term * x[1] * (t - x[2])
        return g

    # ----------------------------------------------------------------------
    # MGH16
    if problem_name == 'MGH16':
        t = (ind + 1) / 5.0
        term1 = 2.0 * (x[0] + t * x[1] - np.exp(t))
        term2 = 2.0 * (x[2] + x[3] * np.sin(t) - np.cos(t))
        g[0] = term1
        g[1] = t * term1
        g[2] = term2
        g[3] = np.sin(t) * term2
        return g

    # ----------------------------------------------------------------------
    # MGH26
    if problem_name == 'MGH26':
        t = np.sum(np.cos(x))
        gaux1 = 2.0 * (n - t + (ind + 1) * (1.0 - np.cos(x[ind])) - np.sin(x[ind]))
        g = gaux1 * np.sin(x)
        g[ind] += gaux1 * ((ind + 1) * np.sin(x[ind]) - np.cos(x[ind]))
        return g

    # ----------------------------------------------------------------------
    # MGH33
    if problem_name == 'MGH33':
        i_matlab = np.arange(1, n + 1)
        faux = np.dot(i_matlab, x)
        faux = 2.0 * ((ind + 1) * faux - 1.0)
        g = faux * i_matlab * (ind + 1)
        return g

    # ----------------------------------------------------------------------
    # MHHM2
    if problem_name == 'MHHM2':
        if ind == 0:
            g[0] = 2.0 * (x[0] - 0.8)
            g[1] = 2.0 * (x[1] - 0.6)
            return g
        if ind == 1:
            g[0] = 2.0 * (x[0] - 0.85)
            g[1] = 2.0 * (x[1] - 0.7)
            return g
        if ind == 2:
            g[0] = 2.0 * (x[0] - 0.9)
            g[1] = 2.0 * (x[1] - 0.6)
            return g

    # ----------------------------------------------------------------------
    # MLF1
    if problem_name == 'MLF1':
        if ind == 0:
            g[0] = np.sin(x[0]) / 20.0 + (1.0 + x[0] / 20.0) * np.cos(x[0])
            return g
        if ind == 1:
            g[0] = np.cos(x[0]) / 20.0 - (1.0 + x[0] / 20.0) * np.sin(x[0])
            return g

    # ----------------------------------------------------------------------
    # MLF2
    if problem_name == 'MLF2':
        if ind == 0:
            term1 = x[0] ** 2 + x[1] - 11.0
            term2 = x[0] + x[1] ** 2 - 7.0
            g[0] = (2.0 * x[0] * term1 + term2) / 100.0
            g[1] = (term1 + 2.0 * x[1] * term2) / 100.0
            return g
        if ind == 1:
            term1 = 4.0 * x[0] ** 2 + 2.0 * x[1] - 11.0
            term2 = 2.0 * x[0] + 4.0 * x[1] ** 2 - 7.0
            g[0] = (8.0 * x[0] * term1 + 2.0 * term2) / 100.0
            g[1] = (2.0 * term1 + 8.0 * x[1] * term2) / 100.0
            return g

    # ----------------------------------------------------------------------
    # MMR1
    if problem_name == 'MMR1':
        if ind == 0:
            g[0] = 1.0
            return g
        if ind == 1:
            g[0] = -(2.0 - 0.8 * np.exp(-((x[1] - 0.6) / 0.4) ** 2) - 
                   np.exp(-((x[1] - 0.2) / 0.04) ** 2)) / (x[0] ** 2)
            g[1] = (10.0 * np.exp(-((x[1] - 0.6) / 0.4) ** 2) * (x[1] - 0.6) + 
                   1250.0 * np.exp(-((x[1] - 0.2) / 0.04) ** 2) * (x[1] - 0.2))
            g[1] /= x[0]
            return g

    # ----------------------------------------------------------------------
    # MMR2
    if problem_name == 'MMR2':
        if ind == 0:
            g[0] = 1.0
            return g
        if ind == 1:
            a = 1.0 + 10.0 * x[1]
            faux = x[0] / a
            sin_term = np.sin(8.0 * np.pi * x[0])
            g[0] = (-2.0 * faux / a - sin_term / a - 
                   8.0 * np.pi * faux * np.cos(8.0 * np.pi * x[0])) * a
            g[1] = 10.0 * (1.0 - faux ** 2 - faux * sin_term)
            g[1] += a * (20.0 * faux * x[0] / (a**2) + 10.0 / (a**2) * x[0] * sin_term)
            return g

    # ----------------------------------------------------------------------
    # MMR3
    if problem_name == 'MMR3':
        if ind == 0:
            g[0] = 3.0 * x[0] ** 2
            return g
        if ind == 1:
            g[0] = -3.0 * (x[1] - x[0]) ** 2
            g[1] = 3.0 * (x[1] - x[0]) ** 2
            return g

    # ----------------------------------------------------------------------
    # MMR4
    if problem_name == 'MMR4':
        if ind == 0:
            t = (2.0 * x[0] + x[1] + 2.0 * x[2] + 1.0) ** 2
            g[0] = 1.0 + 72.0 / t
            g[1] = -2.0 + 36.0 / t
            g[2] = -1.0 + 72.0 / t
            return g
        if ind == 1:
            g[0] = -3.0
            g[1] = 1.0
            g[2] = -1.0
            return g

    # ----------------------------------------------------------------------
    # MOP2
    if problem_name == 'MOP2':
        if ind == 0:
            faux = np.sum((x - 1.0 / np.sqrt(n)) ** 2)
            g = 2.0 * (x - 1.0 / np.sqrt(n)) * np.exp(-faux)
            return g
        if ind == 1:
            faux = np.sum((x + 1.0 / np.sqrt(n)) ** 2)
            g = 2.0 * (x + 1.0 / np.sqrt(n)) * np.exp(-faux)
            return g

    # ----------------------------------------------------------------------
    # MOP3
    if problem_name == 'MOP3':
        if ind == 0:
            A1 = 0.5 * np.sin(1.0) - 2.0 * np.cos(1.0) + np.sin(2.0) - 1.5 * np.cos(2.0)
            A2 = 1.5 * np.sin(1.0) - np.cos(1.0) + 2.0 * np.sin(2.0) - 0.5 * np.cos(2.0)
            B1 = 0.5 * np.sin(x[0]) - 2.0 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
            B2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2.0 * np.sin(x[1]) - 0.5 * np.cos(x[1])
            g[0] = (2.0 * (A1 - B1) * (-0.5 * np.cos(x[0]) - 2.0 * np.sin(x[0])) +
                   2.0 * (A2 - B2) * (-1.5 * np.cos(x[0]) - np.sin(x[0])))
            g[1] = (2.0 * (A1 - B1) * (-np.cos(x[1]) - 1.5 * np.sin(x[1])) +
                   2.0 * (A2 - B2) * (-2.0 * np.cos(x[1]) - 0.5 * np.sin(x[1])))
            return g
        if ind == 1:
            g[0] = 2.0 * (x[0] + 3.0)
            g[1] = 2.0 * (x[1] + 1.0)
            return g

    # ----------------------------------------------------------------------
    # MOP5
    if problem_name == 'MOP5':
        if ind == 0:
            g[0] = x[0] + 2.0 * x[0] * np.cos(x[0] ** 2 + x[1] ** 2)
            g[1] = x[1] + 2.0 * x[1] * np.cos(x[0] ** 2 + x[1] ** 2)
            return g
        if ind == 1:
            g[0] = 3.0 * (3.0 * x[0] - 2.0 * x[1] + 4.0) / 4.0 + 2.0 * (x[0] - x[1] + 1.0) / 27.0
            g[1] = -2.0 * (3.0 * x[0] - 2.0 * x[1] + 4.0) / 4.0 - 2.0 * (x[0] - x[1] + 1.0) / 27.0
            return g
        if ind == 2:
            term1 = (x[0] ** 2 + x[1] ** 2 + 1.0) ** 2
            term2 = np.exp(-x[0] ** 2 - x[1] ** 2)
            g[0] = -2.0 * x[0] / term1 + 2.2 * x[0] * term2
            g[1] = -2.0 * x[1] / term1 + 2.2 * x[1] * term2
            return g

    # ----------------------------------------------------------------------
    # MOP6
    if problem_name == 'MOP6':
        if ind == 0:
            g[0] = 1.0
            return g
        if ind == 1:
            a = 1.0 + 10.0 * x[1]
            b = np.sin(8.0 * np.pi * x[0])
            t = x[0] / a
            g[0] = -2.0 * t - b - 8.0 * np.pi * x[0] * np.cos(8.0 * np.pi * x[0])
            g[1] = 10.0 * (1.0 - t ** 2 - t * b) + 10.0 * x[0] / a * (2.0 * t + b)
            return g

    # ----------------------------------------------------------------------
    # MOP7
    if problem_name == 'MOP7':
        if ind == 0:
            g[0] = x[0] - 2.0
            g[1] = 2.0 * (x[1] + 1.0) / 13.0
            return g
        if ind == 1:
            g[0] = (x[0] + x[1] - 3.0) / 18.0 - (-x[0] + x[1] + 2.0) / 4.0
            g[1] = (x[0] + x[1] - 3.0) / 18.0 + (-x[0] + x[1] + 2.0) / 4.0
            return g
        if ind == 2:
            g[0] = 2.0 * (x[0] + 2.0 * x[1] - 1.0) / 175.0 - 2.0 * (-x[0] + 2.0 * x[1]) / 17.0
            g[1] = 4.0 * (x[0] + 2.0 * x[1] - 1.0) / 175.0 + 4.0 * (-x[0] + 2.0 * x[1]) / 17.0
            return g

    # ----------------------------------------------------------------------
    # PNR
    if problem_name == 'PNR':
        if ind == 0:
            g[0] = 4.0 * x[0] ** 3 - 2.0 * x[0] - 10.0 * x[1]
            g[1] = 4.0 * x[1] ** 3 + 2.0 * x[1] - 10.0 * x[0]
            return g
        if ind == 1:
            g[0] = 2.0 * (x[0] - 1.0)
            g[1] = 2.0 * x[1]
            return g

    # ----------------------------------------------------------------------
    # QV1
    if problem_name == 'QV1':
        if ind == 0:
            faux = np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0)
            faux = 0.25 * (faux / n) ** (-0.75)
            g = faux * (2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)) / n
            return g
        if ind == 1:
            x_shifted = x - 1.5
            faux = np.sum(x_shifted ** 2 - 10.0 * np.cos(2.0 * np.pi * x_shifted) + 10.0)
            faux = 0.25 * (faux / n) ** (-0.75)
            g = faux * (2.0 * x_shifted + 20.0 * np.pi * np.sin(2.0 * np.pi * x_shifted)) / n
            return g

    # ----------------------------------------------------------------------
    # SD
    if problem_name == 'SD':
        if ind == 0:
            g[0] = 2.0
            g[1] = np.sqrt(2.0)
            g[2] = np.sqrt(2.0)
            g[3] = 1.0
            return g
        if ind == 1:
            g[0] = -2.0 / (x[0] ** 2)
            g[1] = -2.0 * np.sqrt(2.0) / (x[1] ** 2)
            g[2] = -2.0 * np.sqrt(2.0) / (x[2] ** 2)
            g[3] = -2.0 / (x[3] ** 2)
            return g

    # ----------------------------------------------------------------------
    # SLCDT1
    if problem_name == 'SLCDT1':
        term1 = (x[0] + x[1]) / np.sqrt(1.0 + (x[0] + x[1]) ** 2)
        term2 = (x[0] - x[1]) / np.sqrt(1.0 + (x[0] - x[1]) ** 2)
        term_exp = -2.0 * 0.85 * (x[0] + x[1]) * np.exp(-(x[0] + x[1]) ** 2)
        if ind == 0:
            g[0] = 0.5 * (term1 + term2 + 1.0) + term_exp
            g[1] = 0.5 * (term1 - term2 - 1.0) + term_exp
            return g
        if ind == 1:
            g[0] = 0.5 * (term1 + term2 - 1.0) + term_exp
            g[1] = 0.5 * (term1 - term2 + 1.0) + term_exp
            return g

    # ----------------------------------------------------------------------
    # SLCDT2
    if problem_name == 'SLCDT2':
        if ind == 0:
            g[0] = 4.0 * (x[0] - 1.0) ** 3
            g[1:] = 2.0 * (x[1:] - 1.0)
            return g
        if ind == 1:
            g = 2.0 * (x + 1.0)
            g[1] = 4.0 * (x[1] + 1.0) ** 3
            return g
        if ind == 2:
            i_matlab = np.arange(1, n + 1)
            signs = (-1.0) ** (i_matlab + 1)
            g = 2.0 * (x - signs)
            g[2] = 4.0 * (x[2] - 1.0) ** 3 # x[2] - signs[2] = x[2] - 1.0
            return g

    # ----------------------------------------------------------------------
    # SP1
    if problem_name == 'SP1':
        if ind == 0:
            g[0] = 2.0 * (x[0] - 1.0) + 2.0 * (x[0] - x[1])
            g[1] = -2.0 * (x[0] - x[1])
            return g
        if ind == 1:
            g[0] = 2.0 * (x[0] - x[1])
            g[1] = 2.0 * (x[1] - 3.0) - 2.0 * (x[0] - x[1])
            return g

    # ----------------------------------------------------------------------
    # SSFYY2
    if problem_name == 'SSFYY2':
        if ind == 0:
            g[0] = 2.0 * x[0] + 5.0 * np.pi * np.sin(x[0] * np.pi / 2.0)
            return g
        if ind == 1:
            g[0] = 2.0 * (x[0] - 4.0)
            return g

    # ----------------------------------------------------------------------
    # SK1
    if problem_name == 'SK1':
        if ind == 0:
            g[0] = 4.0 * x[0] ** 3 + 9.0 * x[0] ** 2 - 20.0 * x[0] - 10.0
            return g
        if ind == 1:
            g[0] = 2.0 * x[0] ** 3 - 6.0 * x[0] ** 2 - 20.0 * x[0] + 10.0
            return g

    # ----------------------------------------------------------------------
    # SK2
    if problem_name == 'SK2':
        if ind == 0:
            g[0] = 2.0 * (x[0] - 2.0)
            g[1] = 2.0 * (x[1] + 3.0)
            g[2] = 2.0 * (x[2] - 5.0)
            g[3] = 2.0 * (x[3] - 4.0)
            return g
        if ind == 1:
            faux = 1.0 + np.dot(x, x) / 100.0
            sin_sum = np.sum(np.sin(x))
            g = (-np.cos(x) * faux + sin_sum * x / 50.0) / (faux ** 2)
            return g

    # ----------------------------------------------------------------------
    # TKLY1
    if problem_name == 'TKLY1':
        if ind == 0:
            g[0] = 1.0
            return g
        if ind == 1:
            def calc_A(val):
                return (2.0 - np.exp(-((val - 0.1) / 4.0e-3) ** 2) -
                        0.8 * np.exp(-((val - 0.9) / 0.4) ** 2))
            def calc_grad_A(val):
                return (500.0 * np.exp(-((val - 0.1) / 4.0e-3) ** 2) * ((val - 0.1) / 4.0e-3) +
                        4.0 * np.exp(-((val - 0.9) / 0.4) ** 2) * ((val - 0.9) / 0.4))

            A1 = calc_A(x[1])
            A2 = calc_A(x[2])
            A3 = calc_A(x[3])
            
            g[0] = -A1 * A2 * A3 / (x[0] ** 2)
            g[1] = A2 * A3 / x[0] * calc_grad_A(x[1])
            g[2] = A1 * A3 / x[0] * calc_grad_A(x[2])
            g[3] = A1 * A2 / x[0] * calc_grad_A(x[3])
            return g

    # ----------------------------------------------------------------------
    # Toi4
    if problem_name == 'Toi4':
        if ind == 0:
            g[0] = 2.0 * x[0]
            g[1] = 2.0 * x[1]
            return g
        if ind == 1:
            g[0] = x[0] - x[1]
            g[1] = -(x[0] - x[1])
            g[2] = x[2] - x[3]
            g[3] = -(x[2] - x[3])
            return g

    # ----------------------------------------------------------------------
    # Toi8
    if problem_name == 'Toi8':
        if ind == 0:
            g[0] = 4.0 * (2.0 * x[0] - 1.0)
            return g
        else: # ind != 0
            g[ind - 1] = 4.0 * (ind + 1) * (2.0 * x[ind - 1] - x[ind])
            g[ind] = -2.0 * (ind + 1) * (2.0 * x[ind - 1] - x[ind])
            return g

    # ----------------------------------------------------------------------
    # Toi9
    if problem_name == 'Toi9':
        if ind == 0:
            g[0] = 4.0 * (2.0 * x[0] - 1.0)
            g[1] = 2.0 * x[1]
            return g
        if 0 < ind < n - 1:
            g[ind - 1] = (4.0 * (ind + 1) * (2.0 * x[ind - 1] - x[ind]) - 
                         2.0 * ind * x[ind - 1])
            g[ind] = -2.0 * (ind + 1) * (2.0 * x[ind - 1] - x[ind]) + 2.0 * (ind + 1) * x[ind]
            return g
        if ind == n - 1:
            g[n - 2] = 4.0 * n * (2.0 * x[n - 2] - x[n - 1]) - 2.0 * (n - 1.0) * x[n - 2]
            g[n - 1] = -2.0 * n * (2.0 * x[n - 2] - x[n - 1])
            return g

    # ----------------------------------------------------------------------
    # Toi10 (Rosenbrock)
    if problem_name == 'Toi10':
        ind_py = ind
        g[ind_py] = -400.0 * (x[ind_py + 1] - x[ind_py] ** 2) * x[ind_py]
        g[ind_py + 1] = 200.0 * (x[ind_py + 1] - x[ind_py] ** 2) + 2.0 * (x[ind_py + 1] - 1.0)
        return g

    # ----------------------------------------------------------------------
    # VU1
    if problem_name == 'VU1':
        if ind == 0:
            term = (x[0] ** 2 + x[1] ** 2 + 1.0) ** 2
            g[0] = -2.0 * x[0] / term
            g[1] = -2.0 * x[1] / term
            return g
        if ind == 1:
            g[0] = 2.0 * x[0]
            g[1] = 6.0 * x[1]
            return g

    # ----------------------------------------------------------------------
    # VU2
    if problem_name == 'VU2':
        if ind == 0:
            g[0] = 1.0
            g[1] = 1.0
            return g
        if ind == 1:
            g[0] = 2.0 * x[0]
            g[1] = 2.0
            return g

    # ----------------------------------------------------------------------
    # ZDT1
    if problem_name == 'ZDT1':
        if ind == 0:
            g[0] = 1.0
            return g
        if ind == 1:
            faux = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
            t = x[0] / faux
            g[0] = -0.5 * (t ** (-0.5))
            g[1:] = 9.0 / (n - 1) * (1.0 - 0.5 * np.sqrt(t))
            return g

    # ----------------------------------------------------------------------
    # ZDT2
    if problem_name == 'ZDT2':
        if ind == 0:
            g[0] = 1.0
            return g
        if ind == 1:
            faux = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
            t = x[0] / faux
            g[0] = -2.0 * t
            g[1:] = 9.0 / (n - 1) * (1.0 + t ** 2)
            return g

    # ----------------------------------------------------------------------
    # ZDT3
    if problem_name == 'ZDT3':
        if ind == 0:
            g[0] = 1.0
            return g
        if ind == 1:
            faux = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
            t = x[0] / faux
            a = np.sin(10.0 * np.pi * x[0])
            g[0] = -0.5 * (t ** (-0.5)) - a - 10.0 * np.pi * x[0] * np.cos(10.0 * np.pi * x[0])
            g[1:] = 9.0 / (n - 1) * (1.0 - 0.5 * np.sqrt(t))
            return g

    # ----------------------------------------------------------------------
    # ZDT4
    if problem_name == 'ZDT4':
        if ind == 0:
            g[0] = 1.0
            return g
        if ind == 1:
            x_subset = x[1:]
            faux = 1.0 + 10.0 * (n - 1) + np.sum(x_subset ** 2 - 10.0 * np.cos(4.0 * np.pi * x_subset))
            t = x[0] / faux
            g[0] = -0.5 * (t ** (-0.5))
            a = np.sin(4.0 * np.pi * x_subset)
            g[1:] = (2.0 * x_subset + 40.0 * np.pi * a) * (1.0 - 0.5 * np.sqrt(t))
            return g

    # ----------------------------------------------------------------------
    # ZDT6
    if problem_name == 'ZDT6':
        if ind == 0:
            a = np.exp(-4.0 * x[0])
            b = np.sin(6.0 * np.pi * x[0])
            g[0] = 4.0 * a * (b ** 6) - 36.0 * np.pi * a * (b ** 5) * np.cos(6.0 * np.pi * x[0])
            return g
        if ind == 1:
            a = np.exp(-4.0 * x[0])
            b = np.sin(6.0 * np.pi * x[0])
            gaux1 = 1.0 - a * (b ** 6)
            gaux2 = 1.0 + 9.0 * (np.sum(x[1:]) / (n - 1)) ** 0.25
            t = gaux1 / gaux2
            A1 = 9.0 * 0.25 * (np.sum(x[1:]) / (n - 1)) ** (-0.75) / (n - 1)
            g[0] = -2.0 * t * (4.0 * a * (b ** 6) - 36.0 * np.pi * a * (b ** 5) * np.cos(6.0 * np.pi * x[0]))
            g[1:] = A1 * (1.0 + t ** 2)
            return g

    # ----------------------------------------------------------------------
    # ZLT1
    if problem_name == 'ZLT1':
        g = 2.0 * x
        g[ind] = 2.0 * (x[ind] - 1.0)
        return g

    # ----------------------------------------------------------------------
    # If not found
    raise ValueError(f"Problem '{problem_name}' or index {ind} is not defined in evalgradg.")


