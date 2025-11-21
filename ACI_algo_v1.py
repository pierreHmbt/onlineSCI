import numpy as np
import pandas as pd

def infoACI_precompute_q(dt_Xt, dt_Yt, dt_Ypredt, dt_Scorest, info_fun, Error_func, gamma, alpha=.1, q0=1, reac0=0):
    
    T = len(dt_Yt['value'])
    dates = dt_Yt.index
    
    errt_list = [np.nan]
    Qt_list = [q0]
    info_list = [np.nan]
    ll = 0

    # Precompute scores (not allowed in 'real' online)
    Scores_test = dt_Scorest.values
    
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = dt_Xt.values[r]
        Ytest_temp = dt_Yt['value'][r]
        Ypredtest_temp = dt_Ypredt['value'][r]
        scores_test = Scores_test[r]

        err_t = Error_func(Ypredtest_temp, Ytest_temp, Xtest_temp, scores_test, Qt_list[r-1])
        reac_t = err_t*(Qt_list[r-1] >= reac0)*1 + (Qt_list[r-1] < reac0)*1
        
        if info_fun(Ypredtest_temp, Ytest_temp, Xtest_temp, scores_test, Qt_list[r-1]):
            ll += 1
            info_list.append(1)
            qt_new = Qt_list[r-1] + gamma[ll-1]*( reac_t  - alpha)
        else:
            info_list.append(0)
            qt_new = Qt_list[r-1]

        Qt_list.append(qt_new)
        errt_list.append(err_t)

    # Create a DataFrame from the generated data
    dt_Qt = pd.DataFrame({'date': dates, 'value': Qt_list})
    dt_Qt.set_index('date', inplace=True)

    dt_errt = pd.DataFrame({'date': dates, 'value': errt_list})
    dt_errt.set_index('date', inplace=True)
    
    dt_infot = pd.DataFrame({'date': dates, 'value': info_list})
    dt_infot.set_index('date', inplace=True)
    
    return dt_Qt, dt_errt, dt_infot
