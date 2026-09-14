import numpy as np
from scipy.optimize import linprog

def evalh(n, x, ind, A_list, b_list):
    '''
    Evaluate

        H_j(x) = max_z { x.T @ z : A_j @ z <= b_j }

    where z is the uncertainty parameter.
    '''
    
    # Bypass evalh for SFS problem since we don't use robust optimization
    if A_list is None and b_list is None:
        return 0.0, 0

    
    try:
        # Lấy A_i và b_i từ list (chỉ số 0-based)
        A_i = A_list[ind]
        b_i = b_list[ind].flatten()

        # Gọi linprog
        '''
        Linprog giải bài toán dạng chuẩn min_x c.T * x 
        sao cho:    A_ub <= b_ub
                    A_eq = b_eq
                    li <= xi <= ui
        '''
        result = linprog(c=-x,
                         A_ub=A_i,
                         b_ub=b_i,
                         A_eq=None,
                         b_eq=None,
                         bounds=(-1e10, 1e10),
                         method='highs-ds',
                         options={'disp': False,
                                  'presolve': True,
                              }
        )

        if result.status == 0:
            info = 0
            h = -result.fun
        else:
            print(f"Lỗi linprog trong evalh cho mục tiêu {ind + 1}: {result.message} (Status: {result.status})")
            info = 2
            h = np.nan
            
    except Exception as e:
        # Bắt các lỗi nghiêm trọng khác
        print(f"Error in evalh (objective {ind}): {e}")
        info = -1
        h = np.nan
        
    return h, info
