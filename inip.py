import numpy as np

def inip(problem_name, num_starts):

    # Validate nmax early: downstream code expects at least one start point.
    try:
        num_starts = int(num_starts)
    except (TypeError, ValueError):
        raise ValueError(f"nmax must be an integer >= 1, got {num_starts!r}.")
    if num_starts < 1:
        raise ValueError(f"nmax must be an integer >= 1, got {num_starts}.")
        
    # ----------------------------------------------------------------------
    if problem_name == 'AP1':
        n = 2
        m = 3
        l = np.full(n, -1.0e1)
        u = np.full(n, 1.0e1)
        # Optimize for loop with vector operations
        
    
    # ----------------------------------------------------------------------
    elif problem_name == 'AP4':
        n = 3
        m = 3
        l = np.full(n, -1.0e1)
        u = np.full(n, 1.0e1)


    # ----------------------------------------------------------------------
    elif problem_name == 'DTLZ1':
        m = 3
        dimk = 5
        n = dimk + m - 1
        l = np.zeros(n)
        u = np.ones(n)


    # ----------------------------------------------------------------------
    elif problem_name == 'DTLZ2':
        m = 3
        dimk = 5
        n = dimk + m - 1
        l = np.zeros(n)
        u = np.ones(n)
    

    # ----------------------------------------------------------------------
    elif problem_name == 'DTLZ3':
        m = 3
        dimk = 5
        n = dimk + m - 1
        l = np.zeros(n)
        u = np.ones(n)


    # ----------------------------------------------------------------------
    elif problem_name == 'DTLZ4':
        m = 3
        dimk = 5
        n = dimk + m - 1
        # alpha = 2.0          # This variable is defined but not used in 'inip'
        l = np.zeros(n)
        u = np.ones(n)


    # ----------------------------------------------------------------------
    elif problem_name == 'FA1':
        n = 3
        m = 3
        l = np.full(n, 0.01)
        u = np.full(n, 1.0)
 

    # ----------------------------------------------------------------------
    elif problem_name == 'Far1':
        n = 2
        m = 2
        l = np.full(n, -1.0)
        u = np.full(n, 1.0)


    # ----------------------------------------------------------------------
    elif problem_name == 'FDS':
        n = 10
        m = 3
        l = np.full(n, -2.0)
        u = np.full(n, 2.0)
        

    # ----------------------------------------------------------------------
    elif problem_name == 'IKK1':
        n = 2
        m = 3
        l = np.full(n, -5.0e1)
        u = np.full(n, 5.0e1)

    # ----------------------------------------------------------------------
    elif problem_name == 'LE1':
        n = 2
        m = 2
        l = np.full(n, 1.0)
        u = np.full(n, 1.0e1)
      

    # ----------------------------------------------------------------------
    elif problem_name == 'LTDZ':
        n = 3
        m = 3
        l = np.zeros(n)
        u = np.ones(n)


    # ----------------------------------------------------------------------
    elif problem_name == 'MGH33':
        n = 3
        m = n
        l = np.full(n, -1.0)
        u = np.full(n, 1.0)
    

    # ----------------------------------------------------------------------
    elif problem_name == 'MLF1':
        n = 1
        m = 2
        l = np.zeros(n)
        u = np.full(n, 2.0e1)


    # ----------------------------------------------------------------------
    elif problem_name == 'MMR4':
        n = 3
        m = 2
        l = np.zeros(n)
        u = np.full(n, 4.0)
       

    # ----------------------------------------------------------------------
    elif problem_name == 'MOP2':
        n = 2
        m = 2
        l = np.full(n, -1.0)
        u = np.full(n, 1.0)


    # ----------------------------------------------------------------------
    elif problem_name == 'MOP3':
        n = 2
        m = 2
        l = np.full(n, -np.pi)
        u = np.full(n, np.pi)


    # ----------------------------------------------------------------------
    elif problem_name == 'MOP5':
        n = 2
        m = 3
        l = np.full(n, -1.0)
        u = np.full(n, 1.0)
  

    # ----------------------------------------------------------------------
    elif problem_name == 'SK1':
        n = 1
        m = 2
        l = np.full(n, -1.0e2)
        u = np.full(n, 1.0e2)

    # ----------------------------------------------------------------------
    elif problem_name == 'TKLY1':
        n = 4
        m = 2
        l = np.zeros(n)
        u = np.zeros(n)
        l[0] = 0.1
        u[0] = 1.0
        l[1:] = 0.0  
        u[1:] = 1.0


    # ----------------------------------------------------------------------
    elif problem_name == 'Toi9':
        n = 4
        m = n
        l = np.full(n, -1.0)
        u = np.full(n, 1.0)

    # ----------------------------------------------------------------------
    elif problem_name == 'ZDT6':
        n = 10
        m = 2
        l = np.full(n, 0.01)
        u = np.full(n, 1.0)

    # ----------------------------------------------------------------------
    else:
        # Throw error if problem name not found
        raise ValueError(f"Problem '{problem_name}' is not defined in inip.py.")
    
    np.random.seed(42)

    # Initialize uniformly
    x_samples = l + (u - l) * np.random.rand(num_starts, n)

    return n, m, l, u, x_samples
