import numpy as np

def evalg(n, x, ind, problem_name):
    """
    Function to evaluate G
    """
    # ----------------------------------------------------------------------
    # AP1
    if problem_name == 'AP1':
        if ind == 0:
            f = 0.25 * ((x[0] - 1.0) ** 4 + 2.0 * (x[1] - 2.0) ** 4)
            return f
        if ind == 1:
            f = np.exp((x[0] + x[1]) / 2.0) + x[0] ** 2 + x[1] ** 2
            return f
        if ind == 2:
            f = (1.0 / 6.0) * (np.exp(-x[0]) + 2.0 * np.exp(-x[1]))
            return f

    # ----------------------------------------------------------------------
    # AP2
    if problem_name == 'AP2':
        if ind == 0:
            f = x[0] ** 2 - 4.0
            return f
        if ind == 1:
            f = (x[0] - 1.0) ** 2
            return f

    # ----------------------------------------------------------------------
    # AP3
    if problem_name == 'AP3':
        if ind == 0:
            f = 0.25 * ((x[0] - 1.0) ** 4 + 2.0 * (x[1] - 2.0) ** 4)
            return f
        if ind == 1:
            f = (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
            return f

    # ----------------------------------------------------------------------
    # AP4
    if problem_name == 'AP4':
        if ind == 0:
            f = (1.0 / 9.0) * ((x[0] - 1.0) ** 4 + 2.0 * (x[1] - 2.0) ** 4 
                              + 3.0 * (x[2] - 3.0) ** 4)
            return f
        if ind == 1:
            f = np.exp((x[0] + x[1] + x[2]) / 3.0) + x[0] ** 2 + x[1] ** 2 + x[2] ** 2
            return f
        if ind == 2:
            f = (1.0 / 12.0) * (3.0 * np.exp(-x[0]) + 4.0 * np.exp(-x[1]) + 3.0 * np.exp(-x[2]))
            return f

    # ----------------------------------------------------------------------
    # BK1
    if problem_name == 'BK1':
        if ind == 0:
            f = x[0] ** 2 + x[1] ** 2
            return f
        if ind == 1:
            f = (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2
            return f

    # ----------------------------------------------------------------------
    # DD1
    if problem_name == 'DD1':
        if ind == 0:
            f = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2
            # Vectorized: f = np.dot(x, x)
            return f
        if ind == 1:
            f = 3.0 * x[0] + 2.0 * x[1] - x[2] / 3.0 + 1.0e-2 * (x[3] - x[4]) ** 3
            return f

    # ----------------------------------------------------------------------
    # DGO1
    if problem_name == 'DGO1':
        if ind == 0:
            f = np.sin(x[0])
            return f
        if ind == 1:
            f = np.sin(x[0] + 0.7)
            return f

    # ----------------------------------------------------------------------
    # DGO2
    if problem_name == 'DGO2':
        if ind == 0:
            f = x[0] ** 2
            return f
        if ind == 1:
            f = 9.0 - np.sqrt(81.0 - x[0] ** 2)
            return f

    # ----------------------------------------------------------------------
    # DTLZ1
    if problem_name == 'DTLZ1':
        k = 5
        m = 3
        
        # 1-based loop (m to n) -> 0-based (m-1 to n-1)
        x_subset = x[m-1:] 
        faux = np.sum((x_subset - 0.5) ** 2 - np.cos(20.0 * np.pi * (x_subset - 0.5)))
        faux = 100.0 * (k + faux)
        
        f = 0.5 * (1.0 + faux)
        
        # 1-based loop (1 to m-ind) -> 0-based (0 to m-ind-1)
        f *= np.prod(x[0 : m - ind - 1])
        
        if ind > 0:
            f *= (1.0 - x[m - ind - 1]) # 1-based (m-ind+1) -> 0-based (m-ind)
        
        return f

    # ----------------------------------------------------------------------
    # DTLZ2
    if problem_name == 'DTLZ2':
        k = 5
        m = 3
        
        x_subset = x[m-1:]
        faux = np.sum((x_subset - 0.5) ** 2)
        
        f = 1.0 + faux
        
        for i in range(m - ind - 1): # 0-based
            f *= np.cos(x[i] * np.pi / 2.0)
            
        if ind > 0:
            f *= np.sin(x[m - ind - 1] * np.pi / 2.0) # 0-based
            
        return f

    # ----------------------------------------------------------------------
    # DTLZ3
    if problem_name == 'DTLZ3':
        k = 5
        m = 3
        
        x_subset = x[m-1:]
        faux = np.sum((x_subset - 0.5) ** 2 - np.cos(20.0 * np.pi * (x_subset - 0.5)))
        faux = 100.0 * (k + faux)
        
        f = 1.0 + faux
        
        for i in range(m - ind - 1): # 0-based
            f *= np.cos(x[i] * np.pi / 2.0)
            
        if ind > 0:
            f *= np.sin(x[m - ind - 1] * np.pi / 2.0) # 0-based
            
        return f

    # ----------------------------------------------------------------------
    # DTLZ4
    if problem_name == 'DTLZ4':
        alpha = 2.0
        k = 5
        m = 3
        
        x_subset = x[m-1:]
        faux = np.sum((x_subset - 0.5) ** 2)
        
        f = 1.0 + faux
        
        for i in range(m - ind - 1): # 0-based
            f *= np.cos((x[i] ** alpha) * np.pi / 2.0)
            
        if ind > 0:
            f *= np.sin((x[m - ind - 1] ** alpha) * np.pi / 2.0) # 0-based
            
        return f

    # ----------------------------------------------------------------------
    # FA1
    if problem_name == 'FA1':
        if ind == 0:
            f = (1.0 - np.exp(-4.0 * x[0])) / (1.0 - np.exp(-4.0))
            return f
        if ind == 1:
            faux = (1.0 - np.exp(-4.0 * x[0])) / (1.0 - np.exp(-4.0))
            f = (x[1] + 1.0) * (1.0 - (faux / (x[1] + 1.0)) ** 0.5)
            return f
        if ind == 2:
            faux = (1.0 - np.exp(-4.0 * x[0])) / (1.0 - np.exp(-4.0))
            f = (x[2] + 1.0) * (1.0 - (faux / (x[2] + 1.0)) ** 0.1)
            return f

    # ----------------------------------------------------------------------
    # Far1
    if problem_name == 'Far1':
        if ind == 0:
            f = (- 2.0 * np.exp(15.0 * (- (x[0] - 0.1) ** 2 - x[1] ** 2))
                 - np.exp(20.0 * (- (x[0] - 0.6) ** 2 - (x[1] - 0.6) ** 2))
                 + np.exp(20.0 * (- (x[0] + 0.6) ** 2 - (x[1] - 0.6) ** 2))
                 + np.exp(20.0 * (- (x[0] - 0.6) ** 2 - (x[1] + 0.6) ** 2))
                 + np.exp(20.0 * (- (x[0] + 0.6) ** 2 - (x[1] + 0.6) ** 2)))
            return f
        if ind == 1:
            f = (+ 2.0 * np.exp(20.0 * (- x[0] ** 2 - x[1] ** 2))
                 + np.exp(20.0 * (- (x[0] - 0.4) ** 2 - (x[1] - 0.6) ** 2))
                 - np.exp(20.0 * (- (x[0] + 0.5) ** 2 - (x[1] - 0.7) ** 2))
                 - np.exp(20.0 * (- (x[0] - 0.5) ** 2 - (x[1] + 0.7) ** 2))
                 + np.exp(20.0 * (- (x[0] + 0.4) ** 2 - (x[1] + 0.8) ** 2)))
            return f

    # ----------------------------------------------------------------------
    # FDS
    if problem_name == 'FDS':
        if ind == 0:
            i_matlab = np.arange(1, n + 1)
            f = np.sum(i_matlab * (x - i_matlab) ** 4) / (n ** 2)
            return f
        if ind == 1:
            f = np.exp(np.sum(x) / n) + np.dot(x, x) # norm(x)^2 = dot(x,x)
            return f
        if ind == 2:
            i_matlab = np.arange(1, n + 1)
            f = np.sum(i_matlab * (n - i_matlab + 1.0) * np.exp(-x))
            f = f / (n * (n + 1.0))
            return f

    # ----------------------------------------------------------------------
    # FF1
    if problem_name == 'FF1':
        if ind == 0:
            f = 1.0 - np.exp(-(x[0] - 1.0) ** 2 - (x[1] + 1.0) ** 2)
            return f
        if ind == 1:
            f = 1.0 - np.exp(-(x[0] + 1.0) ** 2 - (x[1] - 1.0) ** 2)
            return f

    # ----------------------------------------------------------------------
    # Hil1
    if problem_name == 'Hil1':
        a = (2.0 * np.pi / 360.0) * (45.0 + 40.0 * np.sin(2.0 * np.pi * x[0])
                                    + 25.0 * np.sin(2.0 * np.pi * x[1]))
        b = 1.0 + 0.5 * np.cos(2.0 * np.pi * x[0])
        if ind == 0:
            f = np.cos(a) * b
            return f
        if ind == 1:
            f = np.sin(a) * b
            return f

    # ----------------------------------------------------------------------
    # IKK1
    if problem_name == 'IKK1':
        if ind == 0:
            f = x[0] ** 2
            return f
        if ind == 1:
            f = (x[0] - 20.0) ** 2
            return f
        if ind == 2:
            f = x[1] ** 2
            return f

    # ----------------------------------------------------------------------
    # IM1
    if problem_name == 'IM1':
        if ind == 0:
            f = 2.0 * np.sqrt(x[0])
            return f
        if ind == 1:
            f = x[0] * (1.0 - x[1]) + 5.0
            return f

    # ----------------------------------------------------------------------
    # JOS1
    if problem_name == 'JOS1':
        if ind == 0:
            f = np.dot(x, x) / n
            return f
        if ind == 1:
            f = np.sum((x - 2.0) ** 2) / n
            return f

    # ----------------------------------------------------------------------
    # JOS4
    if problem_name == 'JOS4':
        if ind == 0:
            f = x[0]
            return f
        if ind == 1:
            faux = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
            f = faux * (1.0 - (x[0] / faux) ** 0.25 - (x[0] / faux) ** 4.0)
            return f

    # ----------------------------------------------------------------------
    # KW2
    if problem_name == 'KW2':
        if ind == 0:
            f = (- 3.0 * (1.0 - x[0])**2 * np.exp(-x[0]**2 - (x[1] + 1.0)**2)
                 + 10.0 * (x[0] / 5.0 - x[0]**3 - x[1]**5) * np.exp(-x[0]**2 - x[1]**2)
                 + 3.0 * np.exp(-(x[0] + 2.0)**2 - x[1]**2) - 0.5 * (2.0 * x[0] + x[1]))
            return f
        if ind == 1:
            f = (- 3.0 * (1.0 + x[1])**2 * np.exp(-x[1]**2 - (1.0 - x[0])**2)
                 + 10.0 * (-x[1] / 5.0 + x[1]**3 + x[0]**5) * np.exp(-x[0]**2 - x[1]**2)
                 + 3.0 * np.exp(-(2.0 - x[1])**2 - x[0]**2))
            return f

    # ----------------------------------------------------------------------
    # LE1
    if problem_name == 'LE1':
        if ind == 0:
            f = (x[0] ** 2 + x[1] ** 2) ** 0.125
            return f
        if ind == 1:
            f = ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) ** 0.25
            return f

    # ----------------------------------------------------------------------
    # Lov1
    if problem_name == 'Lov1':
        if ind == 0:
            f = -(-1.05 * x[0] ** 2 - 0.98 * x[1] ** 2)
            return f
        if ind == 1:
            f = -(-0.99 * (x[0] - 3.0) ** 2 - 1.03 * (x[1] - 2.5) ** 2)
            return f

    # ----------------------------------------------------------------------
    # Lov2
    if problem_name == 'Lov2':
        if ind == 0:
            f = x[1]
            return f
        if ind == 1:
            f = -( (x[1] - x[0] ** 3) / (x[0] + 1.0) )
            return f

    # ----------------------------------------------------------------------
    # Lov3
    if problem_name == 'Lov3':
        if ind == 0:
            f = -(-x[0] ** 2 - x[1] ** 2)
            return f
        if ind == 1:
            f = -(-(x[0] - 6.0) ** 2 + (x[1] + 0.3) ** 2)
            return f

    # ----------------------------------------------------------------------
    # Lov4
    if problem_name == 'Lov4':
        if ind == 0:
            f = -(-x[0] ** 2 - x[1] ** 2 - 4.0 * (np.exp(-(x[0] + 2.0) ** 2 - x[1] ** 2) +
                                                 np.exp(-(x[0] - 2.0) ** 2 - x[1] ** 2)))
            return f
        if ind == 1:
            f = -(-(x[0] - 6.0) ** 2 - (x[1] + 0.5) ** 2)
            return f

    # ----------------------------------------------------------------------
    # Lov5
    if problem_name == 'Lov5':
        MM = np.array([[-1.0, -0.03, 0.011],
                       [-0.03, -1.0, 0.07],
                       [0.011, 0.07, -1.01]])
        p = np.array([x[0], x[1] - 0.15, x[2]])
        a = 0.35
        A1 = np.sqrt(2.0 * np.pi / a) * np.exp((p @ MM @ p) / (a ** 2))
        
        p = np.array([x[0], x[1] + 1.1, 0.5 * x[2]])
        a = 3.0
        A2 = np.sqrt(2.0 * np.pi / a) * np.exp((p @ MM @ p) / (a ** 2))
        
        faux = A1 + A2
        if ind == 0:
            f = - (np.sqrt(2.0) / 2.0 * (x[0] + faux))
            return f
        if ind == 1:
            f = - (np.sqrt(2.0) / 2.0 * (-x[0] + faux))
            return f

    # ----------------------------------------------------------------------
    # Lov6
    if problem_name == 'Lov6':
        if ind == 0:
            f = x[0]
            return f
        if ind == 1:
            f = (1.0 - np.sqrt(x[0]) - x[0] * np.sin(10.0 * np.pi * x[0])
                 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2)
            return f

    # ----------------------------------------------------------------------
    # LTDZ
    if problem_name == 'LTDZ':
        if ind == 0:
            f = -(3.0 - (1.0 + x[2]) * np.cos(x[0] * np.pi / 2.0) * np.cos(x[1] * np.pi / 2.0))
            return f
        if ind == 1:
            f = -(3.0 - (1.0 + x[2]) * np.cos(x[0] * np.pi / 2.0) * np.sin(x[1] * np.pi / 2.0))
            return f
        if ind == 2:
            f = -(3.0 - (1.0 + x[2]) * np.sin(x[0] * np.pi / 2.0))
            return f

    # ----------------------------------------------------------------------
    # MGH9
    if problem_name == 'MGH9':
        if ind == 0 or ind == 14: y = 9.0e-4
        elif ind == 1 or ind == 13: y = 4.4e-3
        elif ind == 2 or ind == 12: y = 1.75e-2
        elif ind == 3 or ind == 11: y = 5.4e-2
        elif ind == 4 or ind == 10: y = 1.295e-1
        elif ind == 5 or ind == 9: y = 2.42e-1
        elif ind == 6 or ind == 8: y = 3.521e-1
        elif ind == 7: y = 3.989e-1
        
        t = (7.0 - ind) / 2.0
        f = x[0] * np.exp(-x[1] * (t - x[2]) ** 2 / 2.0) - y
        return f

    # ----------------------------------------------------------------------
    # MGH16
    if problem_name == 'MGH16':
        t = (ind + 1) / 5.0
        f = (x[0] + t * x[1] - np.exp(t)) ** 2 + (x[2] + x[3] * np.sin(t) - np.cos(t)) ** 2
        return f

    # ----------------------------------------------------------------------
    # MGH26
    if problem_name == 'MGH26':
        t = np.sum(np.cos(x))
        f = (n - t + (ind + 1) * (1.0 - np.cos(x[ind])) - np.sin(x[ind])) ** 2
        return f

    # ----------------------------------------------------------------------
    # MGH33
    if problem_name == 'MGH33':
        i_matlab = np.arange(1, n + 1)
        faux = np.dot(i_matlab, x)
        f = ((ind + 1) * faux - 1.0) ** 2
        return f

    # ----------------------------------------------------------------------
    # MHHM2
    if problem_name == 'MHHM2':
        if ind == 0:
            f = (x[0] - 0.8) ** 2 + (x[1] - 0.6) ** 2
            return f
        if ind == 1:
            f = (x[0] - 0.85) ** 2 + (x[1] - 0.7) ** 2
            return f
        if ind == 2:
            f = (x[0] - 0.9) ** 2 + (x[1] - 0.6) ** 2
            return f

    # ----------------------------------------------------------------------
    # MLF1
    if problem_name == 'MLF1':
        if ind == 0:
            f = (1.0 + x[0] / 20.0) * np.sin(x[0])
            return f
        if ind == 1:
            f = (1.0 + x[0] / 20.0) * np.cos(x[0])
            return f

    # ----------------------------------------------------------------------
    # MLF2
    if problem_name == 'MLF2':
        if ind == 0:
            f = -(5.0 - ((x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2) / 200.0)
            return f
        if ind == 1:
            f = -(5.0 - ((4.0 * x[0] ** 2 + 2.0 * x[1] - 11.0) ** 2 + 
                        (2.0 * x[0] + 4.0 * x[1] ** 2 - 7.0) ** 2) / 200.0)
            return f

    # ----------------------------------------------------------------------
    # MMR1
    if problem_name == 'MMR1':
        if ind == 0:
            f = x[0]
            return f
        if ind == 1:
            f = (2.0 - 0.8 * np.exp(-((x[1] - 0.6) / 0.4) ** 2) - 
                 np.exp(-((x[1] - 0.2) / 0.04) ** 2))
            f = f / x[0]
            return f

    # ----------------------------------------------------------------------
    # MMR2
    if problem_name == 'MMR2':
        if ind == 0:
            f = x[0]
            return f
        if ind == 1:
            faux = x[0] / (1.0 + 10.0 * x[1])
            f = 1.0 - faux ** 2 - faux * np.sin(8.0 * np.pi * x[0])
            f = f * (1.0 + 10.0 * x[1])
            return f

    # ----------------------------------------------------------------------
    # MMR3
    if problem_name == 'MMR3':
        if ind == 0:
            f = x[0] ** 3
            return f
        if ind == 1:
            f = (x[1] - x[0]) ** 3
            return f

    # ----------------------------------------------------------------------
    # MMR4
    if problem_name == 'MMR4':
        if ind == 0:
            f = x[0] - 2.0 * x[1] - x[2] - 36.0 / (2.0 * x[0] + x[1] + 2.0 * x[2] + 1.0)
            return f
        if ind == 1:
            f = -3.0 * x[0] + x[1] - x[2]
            return f

    # ----------------------------------------------------------------------
    # MOP2
    if problem_name == 'MOP2':
        if ind == 0:
            f = np.sum((x - 1.0 / np.sqrt(n)) ** 2)
            f = 1.0 - np.exp(-f)
            return f
        if ind == 1:
            f = np.sum((x + 1.0 / np.sqrt(n)) ** 2)
            f = 1.0 - np.exp(-f)
            return f

    # ----------------------------------------------------------------------
    # MOP3
    if problem_name == 'MOP3':
        if ind == 0:
            A1 = 0.5 * np.sin(1.0) - 2.0 * np.cos(1.0) + np.sin(2.0) - 1.5 * np.cos(2.0)
            A2 = 1.5 * np.sin(1.0) - np.cos(1.0) + 2.0 * np.sin(2.0) - 0.5 * np.cos(2.0)
            B1 = 0.5 * np.sin(x[0]) - 2.0 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
            B2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2.0 * np.sin(x[1]) - 0.5 * np.cos(x[1])
            f = - (1.0 + (A1 - B1) ** 2 + (A2 - B2) ** 2)
            f = -f
            return f
        if ind == 1:
            f = - ((x[0] + 3.0) ** 2 + (x[1] + 1.0) ** 2)
            f = -f
            return f

    # ----------------------------------------------------------------------
    # MOP5
    if problem_name == 'MOP5':
        if ind == 0:
            f = 0.5 * (x[0] ** 2 + x[1] ** 2) + np.sin(x[0] ** 2 + x[1] ** 2)
            return f
        if ind == 1:
            f = ((3.0 * x[0] - 2.0 * x[1] + 4.0) ** 2 / 8.0 +
                 (x[0] - x[1] + 1.0) ** 2 / 27.0 + 15.0)
            return f
        if ind == 2:
            f = 1.0 / (x[0] ** 2 + x[1] ** 2 + 1.0) - 1.1 * np.exp(-x[0] ** 2 - x[1] ** 2)
            return f

    # ----------------------------------------------------------------------
    # MOP6
    if problem_name == 'MOP6':
        if ind == 0:
            f = x[0]
            return f
        if ind == 1:
            a = 1.0 + 10.0 * x[1]
            t = x[0] / a
            f = a * (1.0 - t ** 2 - t * np.sin(8.0 * np.pi * x[0]))
            return f

    # ----------------------------------------------------------------------
    # MOP7
    if problem_name == 'MOP7':
        if ind == 0:
            f = (x[0] - 2.0) ** 2 / 2.0 + (x[1] + 1.0) ** 2 / 13.0 + 3.0
            return f
        if ind == 1:
            f = (x[0] + x[1] - 3.0) ** 2 / 36.0 + (-x[0] + x[1] + 2.0) ** 2 / 8.0 - 17.0
            return f
        if ind == 2:
            f = (x[0] + 2.0 * x[1] - 1.0) ** 2 / 175.0 + (-x[0] + 2.0 * x[1]) ** 2 / 17.0 - 13.0
            return f

    # ----------------------------------------------------------------------
    # PNR
    if problem_name == 'PNR':
        if ind == 0:
            f = x[0] ** 4 + x[1] ** 4 - x[0] ** 2 + x[1] ** 2 - 10.0 * x[0] * x[1] + 20.0
            return f
        if ind == 1:
            f = (x[0] - 1) ** 2 + x[1] ** 2
            return f

    # ----------------------------------------------------------------------
    # QV1
    if problem_name == 'QV1':
        if ind == 0:
            f = np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0)
            f = (f / n) ** 0.25
            return f
        if ind == 1:
            f = np.sum((x - 1.5) ** 2 - 10.0 * np.cos(2.0 * np.pi * (x - 1.5)) + 10.0)
            f = (f / n) ** 0.25
            return f

    # ----------------------------------------------------------------------
    # SD
    if problem_name == 'SD':
        if ind == 0:
            f = 2.0 * x[0] + np.sqrt(2.0) * (x[1] + x[2]) + x[3]
            return f
        if ind == 1:
            f = 2.0 / x[0] + 2.0 * np.sqrt(2.0) / x[1] + 2.0 * np.sqrt(2.0) / x[2] + 2.0 / x[3]
            return f

    # ----------------------------------------------------------------------
    # SLCDT1
    if problem_name == 'SLCDT1':
        if ind == 0:
            f = (0.5 * (np.sqrt(1.0 + (x[0] + x[1]) ** 2) +
                        np.sqrt(1.0 + (x[0] - x[1]) ** 2) + x[0] - x[1]) +
                 0.85 * np.exp(-(x[0] + x[1]) ** 2))
            return f
        if ind == 1:
            f = (0.5 * (np.sqrt(1.0 + (x[0] + x[1]) ** 2) +
                        np.sqrt(1.0 + (x[0] - x[1]) ** 2) - x[0] + x[1]) +
                 0.85 * np.exp(-(x[0] + x[1]) ** 2))
            return f

    # ----------------------------------------------------------------------
    # SLCDT2
    if problem_name == 'SLCDT2':
        if ind == 0:
            f = (x[0] - 1.0) ** 4 + np.sum((x[1:] - 1.0) ** 2)
            return f
        if ind == 1:
            f = (x[1] + 1.0) ** 4
            all_terms_sq = (x + 1.0) ** 2
            f += np.sum(all_terms_sq) - all_terms_sq[1] # Sum all except index 1
            return f
        if ind == 2:
            f = (x[2] - 1.0) ** 4
            i_matlab = np.arange(1, n + 1)
            signs = (-1.0) ** (i_matlab + 1)
            all_terms_sq = (x - signs) ** 2
            f += np.sum(all_terms_sq) - all_terms_sq[2] # Sum all except index 2
            return f

    # ----------------------------------------------------------------------
    # SP1
    if problem_name == 'SP1':
        if ind == 0:
            f = (x[0] - 1.0) ** 2 + (x[0] - x[1]) ** 2
            return f
        if ind == 1:
            f = (x[1] - 3.0) ** 2 + (x[0] - x[1]) ** 2
            return f

    # ----------------------------------------------------------------------
    # SSFYY2
    if problem_name == 'SSFYY2':
        if ind == 0:
            f = 10.0 + x[0] ** 2 - 10.0 * np.cos(x[0] * np.pi / 2.0)
            return f
        if ind == 1:
            f = (x[0] - 4.0) ** 2
            return f

    # ----------------------------------------------------------------------
    # SK1
    if problem_name == 'SK1':
        if ind == 0:
            f = -(-x[0] ** 4 - 3.0 * x[0] ** 3 + 10.0 * x[0] ** 2 + 10.0 * x[0] + 10.0)
            return f
        if ind == 1:
            f = -(-0.5 * x[0] ** 4 + 2.0 * x[0] ** 3 + 10.0 * x[0] ** 2 - 10.0 * x[0] + 5.0)
            return f

    # ----------------------------------------------------------------------
    # SK2
    if problem_name == 'SK2':
        if ind == 0:
            f = -(-(x[0] - 2.0) ** 2 - (x[1] + 3.0) ** 2 - 
                  (x[2] - 5.0) ** 2 - (x[3] - 4.0) ** 2 + 5.0)
            return f
        if ind == 1:
            f = -((np.sin(x[0]) + np.sin(x[1]) + np.sin(x[2]) + np.sin(x[3])) /
                  (1.0 + (x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2) / 100.0))
            return f

    # ----------------------------------------------------------------------
    # TKLY1
    if problem_name == 'TKLY1':
        if ind == 0:
            f = x[0]
            return f
        if ind == 1:
            A1 = (2.0 - np.exp(-((x[1] - 0.1) / 4.0e-3) ** 2) -
                  0.8 * np.exp(-((x[1] - 0.9) / 0.4) ** 2))
            A2 = (2.0 - np.exp(-((x[2] - 0.1) / 4.0e-3) ** 2) -
                  0.8 * np.exp(-((x[2] - 0.9) / 0.4) ** 2))
            A3 = (2.0 - np.exp(-((x[3] - 0.1) / 4.0e-3) ** 2) -
                  0.8 * np.exp(-((x[3] - 0.9) / 0.4) ** 2))
            f = A1 * A2 * A3 / x[0]
            return f

    # ----------------------------------------------------------------------
    # Toi4
    if problem_name == 'Toi4':
        if ind == 0:
            f = x[0] ** 2 + x[1] ** 2 + 1.0
            return f
        if ind == 1:
            f = 0.5 * ((x[0] - x[1]) ** 2 + (x[2] - x[3]) ** 2) + 1.0
            return f

    # ----------------------------------------------------------------------
    # Toi8
    if problem_name == 'Toi8':
        if ind == 0:
            f = (2.0 * x[0] - 1.0) ** 2
            return f
        else: # ind != 0
            f = (ind + 1) * (2.0 * x[ind - 1] - x[ind]) ** 2
            return f

    # ----------------------------------------------------------------------
    # Toi9
    if problem_name == 'Toi9':
        if ind == 0:
            f = (2.0 * x[0] - 1.0) ** 2 + x[1] ** 2
            return f
        if 0 < ind < n - 1:
            f = ((ind + 1) * (2.0 * x[ind - 1] - x[ind]) ** 2
                 - ind * x[ind - 1] ** 2 + (ind + 1) * x[ind] ** 2)
            return f
        if ind == n - 1:
            f = n * (2.0 * x[n - 2] - x[n - 1]) ** 2 - (n - 1.0) * x[n - 2] ** 2
            return f

    # ----------------------------------------------------------------------
    # Toi10 (Rosenbrock)
    if problem_name == 'Toi10':
        f = 100.0 * (x[ind + 1] - x[ind] ** 2) ** 2 + (x[ind + 1] - 1.0) ** 2
        return f

    # ----------------------------------------------------------------------
    # VU1
    if problem_name == 'VU1':
        if ind == 0:
            f = 1.0 / (x[0] ** 2 + x[1] ** 2 + 1.0)
            return f
        if ind == 1:
            f = x[0] ** 2 + 3.0 * x[1] ** 2 + 1.0
            return f

    # ----------------------------------------------------------------------
    # VU2
    if problem_name == 'VU2':
        if ind == 0:
            f = x[0] + x[1] + 1.0
            return f
        if ind == 1:
            f = x[0] ** 2 + 2.0 * x[1] - 1.0
            return f

    # ----------------------------------------------------------------------
    # ZDT1
    if problem_name == 'ZDT1':
        if ind == 0:
            f = x[0]
            return f
        if ind == 1:
            faux = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
            f = faux * (1.0 - np.sqrt(x[0] / faux))
            return f

    # ----------------------------------------------------------------------
    # ZDT2
    if problem_name == 'ZDT2':
        if ind == 0:
            f = x[0]
            return f
        if ind == 1:
            faux = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
            f = faux * (1.0 - (x[0] / faux) ** 2)
            return f

    # ----------------------------------------------------------------------
    # ZDT3
    if problem_name == 'ZDT3':
        if ind == 0:
            f = x[0]
            return f
        if ind == 1:
            faux = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
            t = x[0] / faux
            f = faux * (1.0 - np.sqrt(t) - t * np.sin(10.0 * np.pi * x[0]))
            return f

    # ----------------------------------------------------------------------
    # ZDT4
    if problem_name == 'ZDT4':
        if ind == 0:
            f = x[0]
            return f
        if ind == 1:
            x_subset = x[1:]
            faux = np.sum(x_subset ** 2 - 10.0 * np.cos(4.0 * np.pi * x_subset))
            faux += 1.0 + 10.0 * (n - 1)
            t = x[0] / faux
            f = faux * (1.0 - np.sqrt(t))
            return f

    # ----------------------------------------------------------------------
    # ZDT6
    if problem_name == 'ZDT6':
        if ind == 0:
            f = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0])) ** 6
            return f
        if ind == 1:
            f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0])) ** 6
            faux = 1.0 + 9.0 * (np.sum(x[1:]) / (n - 1)) ** 0.25
            f = faux * (1.0 - (f1 / faux) ** 2)
            return f

    # ----------------------------------------------------------------------
    # ZLT1
    if problem_name == 'ZLT1':
        f = (x[ind] - 1.0) ** 2
        all_sq = np.dot(x, x)
        f += all_sq - x[ind] ** 2
        return f

    # ----------------------------------------------------------------------
    # If not found
    raise ValueError(f"Problem '{problem_name}' or index {ind + 1} is not defined in evalg.")