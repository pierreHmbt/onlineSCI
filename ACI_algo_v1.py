import numpy as np
import pandas as pd
from scipy.optimize import fsolve, brentq

def ACI_PI_q(dt_Xt, dt_Yt, Score_func, f, info_fun, alpha=.1):
    
    # Def de Pi(u)-alpha
    def PPi_u(u, scores_t, Xt_temp, alpha):
        a = (scores_t > u)
        b = info_fun(Xt_temp, u)

        ll = np.sum(a*b) + 1
        kk = np.sum(b) + 1

        return ll/kk-alpha

    T = len(dt_Yt['value'])
    dates = dt_Xt.index

    qt_list = [np.nan]
    info_list = [np.nan]
    errt_list = [np.nan]
    for r in range(1, T):
        # Get previous time-series and scores
        Xt_temp = dt_Xt.values[:r]
        Yt_temp = dt_Yt['value'][:r]
        scores_t = np.array(Score_func(f(Xt_temp), Yt_temp, Xt_temp))

        # Get current point and score (at time r)
        Xtest_temp = dt_Xt['value'][r]
        Ytest_temp = dt_Yt['value'][r] # not observable yet
        scores_test = Score_func(f(Xtest_temp), Ytest_temp, Xtest_temp) # not observable yet

        # Find the root 
        u_max = fsolve(PPi_u, [0.1], args=(scores_t, Xt_temp, alpha))
        qt_list.append(u_max[0])

        # Coverage
        err_t = 1*(scores_test > u_max)
        errt_list.append(err_t)

        # Set informative or not
        if info_fun(Xtest_temp, u_max):
            info_list.append(1)
        else:
            info_list.append(0)

    # Create a DataFrame from the generated data
    dt_qt = pd.DataFrame({'date': dates, 'value': qt_list})
    dt_qt.set_index('date', inplace=True)


    dt_infot = pd.DataFrame({'date': dates, 'value': info_list})
    dt_infot.set_index('date', inplace=True)


    dt_errt = pd.DataFrame({'date': dates, 'value': errt_list})
    dt_errt.set_index('date', inplace=True)
    
    return dt_qt, dt_errt, dt_infot

def ACI_q(dt_Xt, dt_Yt, Score_func, f, gamma, alpha=.1, q0=1):
    
    T = len(dt_Yt['value'])
    dates = dt_Xt.index
    
    errt_list = [np.nan]
    Qt_list = [q0]
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = dt_Xt['value'][r]
        Ytest_temp = dt_Yt['value'][r]
        scores_test = Score_func(f(Xtest_temp), Ytest_temp, Xtest_temp)
        
        err_t = 1*(scores_test > Qt_list[r-1])
        qt_new = Qt_list[r-1] + gamma[r-1]*( err_t  - alpha)

        Qt_list.append(qt_new)
        errt_list.append(err_t)

    # Create a DataFrame from the generated data
    dt_Qt = pd.DataFrame({'date': dates, 'value': Qt_list})
    dt_Qt.set_index('date', inplace=True)

    dt_errt = pd.DataFrame({'date': dates, 'value': errt_list})
    dt_errt.set_index('date', inplace=True)
    
    return dt_Qt, dt_errt


def infoACI_q(dt_Xt, dt_Yt, Score_func, f, info_fun, Error_func, gamma, alpha=.1, q0=1):
    
    T = len(dt_Yt['value'])
    dates = dt_Xt.index
    
    errt_list = [np.nan]
    Qt_list = [q0]
    info_list = [np.nan]
    ll = 0
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = dt_Xt['value'][r]
        Ytest_temp = dt_Yt['value'][r]
        scores_test = Score_func(f(Xtest_temp), Ytest_temp, Xtest_temp)
        
        err_t = Error_func(Ypredtest_temp, Ytest_temp, Xtest_temp, Qt_list[r-1])
        err_t = err_t*(Qt_list[r-1] >= 0)*1 + (Qt_list[r-1] < 0)*1
        
        if info_fun(Xtest_temp, Qt_list[r-1]):
            ll += 1
            info_list.append(1)
            qt_new = Qt_list[r-1] + gamma[ll-1]*( err_t  - alpha)
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


def infoACI_Meanq(dt_Xt, dt_Yt, Score_func, f, info_fun, Error_func, gamma, alpha=.1, q0=1):
    
    T = len(dt_Yt['value'])
    dates = dt_Xt.index
    
    errt_list = [np.nan]
    Qt_Inter_list = [q0]
    Qt_list = [q0]
    info_list = [np.nan]
    ll = 0
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = dt_Xt['value'][r]
        Ytest_temp = dt_Yt['value'][r]
        scores_test = Score_func(f(Xtest_temp), Ytest_temp, Xtest_temp)
        
        err_t = Error_func(Ypredtest_temp, Ytest_temp, Xtest_temp, Qt_list[r-1])
        err_t = err_t*(Qt_list[r-1] >= 0)*1 + (Qt_list[r-1] < 0)*1
        
        if info_fun(Xtest_temp, Qt_list[r-1]):
            ll += 1
            info_list.append(1)
            qt_Inter_new = Qt_Inter_list[r-1] + gamma[ll-1]*( err_t  - alpha)
            qt_new = ll*Qt_list[r-1]/(ll+1)+qt_Inter_new/(ll+1)
        else:
            info_list.append(0)
            qt_Inter_new=Qt_Inter_list[r-1]
            qt_new = Qt_list[r-1]
        
        Qt_Inter_list.append(qt_Inter_new)
        Qt_list.append(qt_new)
        errt_list.append(err_t)

    # Create a DataFrame from the generated data
    dt_Qt = pd.DataFrame({'date': dates, 'value': Qt_list})
    dt_Qt.set_index('date', inplace=True)

    dt_Qt_Inter = pd.DataFrame({'date': dates, 'value': Qt_Inter_list})
    dt_Qt_Inter.set_index('date', inplace=True)
    
    dt_errt = pd.DataFrame({'date': dates, 'value': errt_list})
    dt_errt.set_index('date', inplace=True)
    
    dt_infot = pd.DataFrame({'date': dates, 'value': info_list})
    dt_infot.set_index('date', inplace=True)
    
    return dt_Qt, dt_Qt_Inter, dt_errt, dt_infot

# pinball loss
def pinball_loss(b, teta, alpha):
    return np.max((b-teta), 0) - alpha*(b-teta)

def DtACI_q(dt_Xt, dt_Yt, gamma_grid, q0_grid, eta, sg, Score_func, f, alpha=.1):
    
    T = len(dt_Yt['value'])
    dates = dt_Xt.index

    k = len(gamma_grid)
    wt = np.repeat(1, k)
    errt_list = [np.nan]
    Qt_list = [np.nan]
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = dt_Xt['value'][r]
        Ytest_temp = dt_Yt['value'][r]
        scores_test = Score_func(f(Xtest_temp), Ytest_temp, Xtest_temp)
        
        pt = wt/np.sum(wt)
        qt = np.random.choice(q0_grid, p=pt)
        
        w_bar = wt*np.exp(-eta * pinball_loss(scores_test, q0_grid, alpha))
        W = np.sum(w_bar)
        wt = (1-sg)*w_bar + W * sg/k
        
        erri = (scores_test > q0_grid)*1
        err_t = (scores_test > qt)*1
        q0_grid = q0_grid + gamma_grid*(erri - alpha)
        
        Qt_list.append(qt)
        errt_list.append(err_t)

    # Create a DataFrame from the generated data
    dt_Qt = pd.DataFrame({'date': dates, 'value': Qt_list})
    dt_Qt.set_index('date', inplace=True)

    dt_errt = pd.DataFrame({'date': dates, 'value': errt_list})
    dt_errt.set_index('date', inplace=True)
    
    return dt_Qt, dt_errt

def infoDtACI_q(dt_Xt, dt_Yt, gamma_grid, q0_grid, eta, sg, Score_func, f, info_fun, alpha=.1):
    
    T = len(dt_Yt['value'])
    dates = dt_Xt.index

    k = len(gamma_grid)
    wt = np.repeat(1, k)
    pt = wt/np.sum(wt)
    qt = np.random.choice(q0_grid, p=pt)
    
    errt_list = [np.nan]
    Qt_list = [qt]
    info_list = [np.nan]
    ll = 0
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = dt_Xt['value'][r]
        Ytest_temp = dt_Yt['value'][r]
        scores_test = Score_func(f(Xtest_temp), Ytest_temp, Xtest_temp)

        if info_fun(Xtest_temp, Qt_list[r-1]):
            pt = wt/np.sum(wt)
            qt = np.random.choice(q0_grid, p=pt)
            
            w_bar = wt*np.exp(-eta * pinball_loss(scores_test, q0_grid, alpha))
            W = np.sum(w_bar)
            wt = (1-sg)*w_bar + W * sg/k
            
            erri = (scores_test > q0_grid)*1
            err_t = (scores_test > qt)*1
            q0_grid = q0_grid + gamma_grid*(erri - alpha)
            
            info_list.append(1)
        else:
            err_t = (scores_test > qt)*1
            info_list.append(0)
        
        Qt_list.append(qt)
        errt_list.append(err_t)

    # Create a DataFrame from the generated data
    dt_Qt = pd.DataFrame({'date': dates, 'value': Qt_list})
    dt_Qt.set_index('date', inplace=True)

    dt_errt = pd.DataFrame({'date': dates, 'value': errt_list})
    dt_errt.set_index('date', inplace=True)
    
    dt_infot = pd.DataFrame({'date': dates, 'value': info_list})
    dt_infot.set_index('date', inplace=True)
    
    return dt_Qt, dt_errt, dt_infot


def infoACI_precompute_q(dt_Xt, dt_Yt, dt_Ypredt, dt_Scorest, info_fun, Error_func, gamma, alpha=.1, q0=1, reac0=0):
    
    T = len(dt_Yt['value'])
    dates = dt_Yt.index
    
    errt_list = [np.nan]
    Qt_list = [q0]
    info_list = [np.nan]
    ll = 0

    # Precompute scores (not allowed in 'real' online)
    # Scores_test = Score_func(dt_Ypredt['value'], dt_Yt['value'], dt_Xt.values)
    Scores_test = dt_Scorest.values
    
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = dt_Xt.value[r]
        Ytest_temp = dt_Yt['value'][r]
        Ypredtest_temp = dt_Ypredt['value'][r]
        scores_test = Scores_test[r]

        err_t = Error_func(Ypredtest_temp, Ytest_temp, Xtest_temp, scores_test, Qt_list[r-1])
        reac_t = err_t*(Qt_list[r-1] >= reac0)*1 + (Qt_list[r-1] < reac0)*1
        
        if info_fun(Ypredtest_temp, Ytest_temp, Xtest_temp, scores_test, Qt_list[r-1]):
            ll += 1
            info_list.append(1)
            qt_new = Qt_list[r-1] + gamma[ll-1]*( reac_t  - alpha)
            # qt_new = Qt_list[r-1] + gamma[r-1]*( err_t  - alpha)
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

def ACI_precompute_q(dt_Xt, dt_Yt, dt_Ypredt, dt_Scorest, info_fun, Error_func, gamma, alpha=.1, q0=1, reac0=0):
    
    T = len(dt_Yt['value'])
    dates = dt_Yt.index
    
    errt_list = [np.nan]
    Qt_list = [q0]
    info_list = [np.nan]
    ll = 0

    # Precompute scores (not allowed in 'real' online)
    # Scores_test = Score_func(dt_Ypredt['value'], dt_Yt['value'], dt_Xt.values)
    Scores_test = dt_Scorest.values
    
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = dt_Xt.value[r]
        Ytest_temp = dt_Yt['value'][r]
        Ypredtest_temp = dt_Ypredt['value'][r]
        scores_test = Scores_test[r]

        err_t = Error_func(Ypredtest_temp, Ytest_temp, Xtest_temp, scores_test, Qt_list[r-1])
        reac_t = err_t*(Qt_list[r-1] >= reac0)*1 + (Qt_list[r-1] < reac0)*1
        
        if info_fun(Ypredtest_temp, Ytest_temp, Xtest_temp, scores_test, Qt_list[r-1]):
            info_list.append(1)
        else:
            info_list.append(0)

        ll += 1
        qt_new = Qt_list[r-1] + gamma[ll-1]*( reac_t  - alpha)

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

def infoDtACI_precompute_q(dt_Xt, dt_Yt, dt_Ypredt, dt_Scorest, info_fun, Error_func, gamma_grid, q0_grid, eta, sg, alpha=.1):
    
    T = len(dt_Yt['value'])
    dates = dt_Yt.index

    k = len(gamma_grid)
    wt = np.repeat(1, k)
    pt = wt/np.sum(wt)
    qt = np.random.choice(q0_grid, p=pt)

    # Precompute scores (not allowed in 'real' online)
    # Scores_test = Score_func(dt_Ypredt['value'], dt_Yt['value'], dt_Xt.values)
    Scores_test = dt_Scorest.values
    
    errt_list = [np.nan]
    Qt_list = [qt]
    info_list = [np.nan]
    ll = 0
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = dt_Xt.value[r]
        Ytest_temp = dt_Yt['value'][r]
        Ypredtest_temp = dt_Ypredt['value'][r]
        scores_test = Scores_test[r]

        if info_fun(Ypredtest_temp, Ytest_temp, Xtest_temp, scores_test, Qt_list[r-1]):
            pt = wt/np.sum(wt)
            qt = np.random.choice(q0_grid, p=pt)
            
            w_bar = wt*np.exp(-eta * pinball_loss(scores_test, q0_grid, alpha))
            W = np.sum(w_bar)
            wt = (1-sg)*w_bar + W * sg/k
            
            erri = Error_func(Ypredtest_temp, Ytest_temp, Xtest_temp, scores_test, q0_grid)
            err_t = Error_func(Ypredtest_temp, Ytest_temp, Xtest_temp, scores_test, qt)
            
            q0_grid = q0_grid + gamma_grid*(erri - alpha)
            
            info_list.append(1)
        else:
            err_t = Error_func(Ypredtest_temp, Ytest_temp, Xtest_temp, scores_test, qt)
            info_list.append(0)
        
        Qt_list.append(qt)
        errt_list.append(err_t)

    # Create a DataFrame from the generated data
    dt_Qt = pd.DataFrame({'date': dates, 'value': Qt_list})
    dt_Qt.set_index('date', inplace=True)

    dt_errt = pd.DataFrame({'date': dates, 'value': errt_list})
    dt_errt.set_index('date', inplace=True)
    
    dt_infot = pd.DataFrame({'date': dates, 'value': info_list})
    dt_infot.set_index('date', inplace=True)
    
    return dt_Qt, dt_errt, dt_infot

# InfoACI for dataframe
def ACI_q_df(dt_Xt, dt_Yt, dt_Ypredt, list_varX, list_varY, gamma, Scores_test, f, info_fun, alpha=.1, q0=1):
    
    T = len(dt_Yt[list_varY])

    # Precompute scores (not allowed in 'real' online)
    # Scores = Score_func(np.ravel(dt_Ypredt[list_varY]), np.ravel(dt_Yt[list_varY].values), dt_Xt[list_varX].values)
    
    errt_list = [np.nan]
    Qt_list = [q0]
    info_list = [np.nan]
    ll = 0
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = np.array(dt_Xt[list_varX].iloc[r]).reshape((-1, 1)).T
        Ytest_temp = np.array(dt_Yt[list_varY].iloc[r])
        Ypred_test_temp = np.array(dt_Ypredt[list_varY].iloc[r])
        scores_test = Scores[r]
        # scores_test = Score_func(fchap(Xtest_temp), Ytest_temp, Xtest_temp)
        
        err_t = 1*(scores_test > Qt_list[r-1])
        if info_fun(Xtest_temp, Ypred_test_temp, Qt_list[r-1]):
            ll += 1
            info_list.append(1)
        else:
            info_list.append(0)

        qt_new = Qt_list[r-1] + gamma[ll-1]*(err_t  - alpha)
        Qt_list.append(qt_new)
        errt_list.append(err_t)

    # Create a DataFrame from the generated data
    index = dt_Xt.index
    
    dt_Qt = pd.DataFrame({'index': index, 'value': Qt_list})
    dt_Qt.set_index('index', inplace=True)

    dt_errt = pd.DataFrame({'index': index, 'value': errt_list})
    dt_errt.set_index('index', inplace=True)
    
    dt_infot = pd.DataFrame({'index': index, 'value': info_list})
    dt_infot.set_index('index', inplace=True)
    
    return dt_Qt, dt_errt, dt_infot

# InfoACI for dataframe
def infoACI_q_df(dt_Xt, dt_Yt, dt_Ypredt, list_varX, list_varY, gamma, Score_func, f, info_fun, alpha=.1, q0=1):
    
    T = len(dt_Yt[list_varY])

    # Precompute scores (not allowed in 'real' online)
    Scores = Score_func(np.ravel(dt_Ypredt[list_varY]), np.ravel(dt_Yt[list_varY].values), dt_Xt[list_varX].values)
    
    errt_list = [np.nan]
    Qt_list = [q0]
    info_list = [np.nan]
    ll = 0
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = np.array(dt_Xt[list_varX].iloc[r]).reshape((-1, 1)).T
        Ytest_temp = np.array(dt_Yt[list_varY].iloc[r])
        Ypred_test_temp = np.array(dt_Ypredt[list_varY].iloc[r])
        scores_test = Scores[r]
        # scores_test = Score_func(fchap(Xtest_temp), Ytest_temp, Xtest_temp)
        
        err_t = 1*(scores_test > Qt_list[r-1])
        if info_fun(Xtest_temp, Ypred_test_temp, Qt_list[r-1]):
            ll += 1
            info_list.append(1)
            qt_new = Qt_list[r-1] + gamma[ll-1]*(err_t  - alpha)
        else:
            info_list.append(0)
            qt_new = Qt_list[r-1]

        Qt_list.append(qt_new)
        errt_list.append(err_t)

    # Create a DataFrame from the generated data
    index = dt_Xt.index
    
    dt_Qt = pd.DataFrame({'index': index, 'value': Qt_list})
    dt_Qt.set_index('index', inplace=True)

    dt_errt = pd.DataFrame({'index': index, 'value': errt_list})
    dt_errt.set_index('index', inplace=True)
    
    dt_infot = pd.DataFrame({'index': index, 'value': info_list})
    dt_infot.set_index('index', inplace=True)
    
    return dt_Qt, dt_errt, dt_infot

def infoDtACI_q_df(dt_Xt, dt_Yt, dt_Ypredt, list_varX, list_varY, gamma_grid, q0_grid, eta, sg, Score_func, f, info_fun, alpha=.1):
    
    T = len(dt_Yt[list_varY])

    # Precompute scores (not allowed in 'real' online)
    Scores = Score_func(np.ravel(dt_Ypredt[list_varY]), np.ravel(dt_Yt[list_varY].values), dt_Xt[list_varX].values)

    k = len(gamma_grid)
    wt = np.repeat(1, k)
    pt = wt/np.sum(wt)
    qt = np.random.choice(q0_grid, p=pt)
    
    errt_list = [np.nan]
    Qt_list = [qt]
    info_list = [np.nan]
    ll = 0
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = np.array(dt_Xt[list_varX].iloc[r]).reshape((-1, 1)).T
        Ytest_temp = np.array(dt_Yt[list_varY].iloc[r])
        Ypred_test_temp = np.array(dt_Ypredt[list_varY].iloc[r])
        scores_test = Scores[r]
        # scores_test = Score_func(fchap(Xtest_temp), Ytest_temp, Xtest_temp)
        if info_fun(Xtest_temp, Ypred_test_temp, Qt_list[r-1]):
            pt = wt/np.sum(wt)
            qt = np.random.choice(q0_grid, p=pt)
            
            w_bar = wt*np.exp(-eta * pinball_loss(scores_test, q0_grid, alpha))
            W = np.sum(w_bar)
            wt = (1-sg)*w_bar + W * sg/k
            
            erri = (scores_test > q0_grid)*1
            err_t = (scores_test > qt)*1
            q0_grid = q0_grid + gamma_grid*(erri - alpha)
            
            info_list.append(1)
        else:
            err_t = (scores_test > qt)*1
            info_list.append(0)
        
        Qt_list.append(qt)
        errt_list.append(err_t)

    # Create a DataFrame from the generated data
    index = dt_Xt.index
    
    dt_Qt = pd.DataFrame({'index': index, 'value': Qt_list})
    dt_Qt.set_index('index', inplace=True)

    dt_errt = pd.DataFrame({'index': index, 'value': errt_list})
    dt_errt.set_index('index', inplace=True)
    
    dt_infot = pd.DataFrame({'index': index, 'value': info_list})
    dt_infot.set_index('index', inplace=True)
    
    return dt_Qt, dt_errt, dt_infot

def DtACI_q_df(dt_Xt, dt_Yt, dt_Ypredt, list_varX, list_varY, gamma_grid, q0_grid, eta, sg, Score_func, f, info_fun, alpha=.1):
    
    T = len(dt_Yt[list_varY])

    # Precompute scores (not allowed in 'real' online)
    Scores = Score_func(np.ravel(dt_Ypredt[list_varY]), np.ravel(dt_Yt[list_varY].values), dt_Xt[list_varX].values)

    k = len(gamma_grid)
    wt = np.repeat(1, k)
    pt = wt/np.sum(wt)
    qt = np.random.choice(q0_grid, p=pt)
    
    errt_list = [np.nan]
    Qt_list = [qt]
    info_list = [np.nan]
    ll = 0
    for r in range(1, T):
        # Get current point and score (at time r)
        Xtest_temp = np.array(dt_Xt[list_varX].iloc[r]).reshape((-1, 1)).T
        Ytest_temp = np.array(dt_Yt[list_varY].iloc[r])
        Ypred_test_temp = np.array(dt_Ypredt[list_varY].iloc[r])
        scores_test = Scores[r]
        # scores_test = Score_func(fchap(Xtest_temp), Ytest_temp, Xtest_temp)

        if info_fun(Xtest_temp, Ypred_test_temp, Qt_list[r-1]):
            info_list.append(1)
        else:
            info_list.append(0)

        pt = wt/np.sum(wt)
        qt = np.random.choice(q0_grid, p=pt)
        
        w_bar = wt*np.exp(-eta * pinball_loss(scores_test, q0_grid, alpha))
        W = np.sum(w_bar)
        wt = (1-sg)*w_bar + W * sg/k
        
        erri = (scores_test > q0_grid)*1
        err_t = (scores_test > qt)*1
        q0_grid = q0_grid + gamma_grid*(erri - alpha)
        
        Qt_list.append(qt)
        errt_list.append(err_t)

    # Create a DataFrame from the generated data
    index = dt_Xt.index
    
    dt_Qt = pd.DataFrame({'index': index, 'value': Qt_list})
    dt_Qt.set_index('index', inplace=True)

    dt_errt = pd.DataFrame({'index': index, 'value': errt_list})
    dt_errt.set_index('index', inplace=True)
    
    dt_infot = pd.DataFrame({'index': index, 'value': info_list})
    dt_infot.set_index('index', inplace=True)
    
    return dt_Qt, dt_errt, dt_infot
