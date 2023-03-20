# @Time     : Jan. 10, 2019 15:15
# @Author   : Veritas YIN
# @FileName : math_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    mask=np.not_equal(v,0)
    mask=mask.astype('float32')
    mask=mask/np.mean(mask)
    mape=np.abs(v_-v)/v
    mape=np.mean(mask*mape)
    return mape


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))


def evaluation(y, y_, x_stats, plot=False, lon=None, lat=None):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param lon: ndarray, Same shape as the spatial dim of y and y_. Used to plot
    :param lat: ndarray, Same shape as the spatial dim of y and y_. Used to plot
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)
    print('y:{},y_:{}'.format(y.shape,y_.shape))

    #======================================================================
    ''' Plot results to help assess model performance '''
    if plot:
        curr_y  = y[-1, -1, ...].ravel()
        curr_y_ = y_[-1, -1, ...].ravel()

        print('matplotlib.__version__', matplotlib.__version__)
        print('~+~+~+~+~+~+~+~+~+~ lon =', np.shape(lon))
        print('~+~+~+~+~+~+~+~+~+~ lat =', np.shape(lat))
        print('~+~+~+~+~+~+~+~+~+~ curr_y =', np.shape(curr_y))
        print('~+~+~+~+~+~+~+~+~+~ curr_y_ =', np.shape(curr_y_))

        min_lat = min(lat.ravel())
        max_lat = max(lat.ravel())
        max_lon = max(lon.ravel())
        min_lon = min(lon.ravel())
        print('1')
        min_pred = min(curr_y_.ravel())
        max_pred = max(curr_y_.ravel())
        print('2')
        min_targ = min(curr_y.ravel())
        max_targ = max(curr_y.ravel())
        print('3')
        norm_pred = Normalize(vmin=min_pred, vmax=max_pred)
        cmap_pred = LinearSegmentedColormap.from_list('custom', ['blue', 'red'], N=200) # Higher N=more smooth
        print('4')
        norm_targ = Normalize(vmin=min_targ, vmax=max_targ)
        cmap_targ = LinearSegmentedColormap.from_list('custom', ['blue', 'red'], N=200) # Higher N=more smooth 
        print('5')
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        fig.tight_layout(pad=3)
        print('6')
        ax[0].set_xlim(min_lon - .1, max_lon + .1)
        ax[1].set_xlim(min_lon - .1, max_lon + .1)
        ax[0].set_ylim(min_lat - .1, max_lat + .1)
        ax[1].set_ylim(min_lat - .1, max_lat + .1)
        print('7')
        ax[0].set_title('Data')
        ax[1].set_title('Prediction')
        print('8')
        targ = ax[0].scatter(x=lon, y=lat, c=curr_y, s=25, marker='s', cmap=cmap_targ, norm=norm_targ, alpha=1)
        pred = ax[1].scatter(x=lon, y=lat, c=curr_y_, s=25, marker='s', cmap=cmap_pred, norm=norm_pred, alpha=1)
        print('9')
        plt.colorbar(targ, ax=ax[0])
        plt.colorbar(pred, ax=ax[1])
        print('10')
        plt.show()
        print('11')
    #======================================================================
    if dim == 3:
        # single_step case
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y=np.expand_dims(y,axis=-1)
        #y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)
