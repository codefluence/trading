import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import beta

import matplotlib.pylab as P
import torch as T

import pandas as pd
import numpy as np

import os
import platform
from tqdm import tqdm

def get_time_series(settings):

    series = []
    vols = []

    targets = pd.read_csv(settings['DATA_DIR'] + settings['ENV'] + '.csv')

    for folder in 'book_'+settings['ENV']+'.parquet', 'trade_'+settings['ENV']+'.parquet': 

        file_paths = []
        path_root = settings['DATA_DIR'] + folder + '/'

        for path, _, files in os.walk(path_root):
            for name in files:
                file_paths.append(os.path.join(path, name))

        for file_path in tqdm(file_paths):

            df = pd.read_parquet(file_path, engine='pyarrow')
            slash = '\\' if platform.system() == 'Windows' else '/'
            stock_id = int(file_path.split(slash)[-2].split('=')[-1])

            for time_id in np.unique(df.time_id):

                df_time = df[df.time_id == time_id].reset_index(drop=True)
                with_changes_len = len(df_time)

                # In kaggle public leaderboard, some books don't start with seconds_in_bucket=0
                # if 'book' in file_path:
                #     assert df_time.seconds_in_bucket[0] == 0

                df_time = df_time.reindex(list(range(600)))

                missing = set(range(600)) - set(df_time.seconds_in_bucket)
                df_time.loc[with_changes_len:,'seconds_in_bucket'] = list(missing)

                df_time = df_time.sort_values(by='seconds_in_bucket').reset_index(drop=True)

                if 'book' in file_path:

                    df_time = df_time.iloc[:,2:].ffill(axis=0)

                    # In kaggle public leaderboard, some books don't start with seconds_in_bucket=0
                    # workaround for https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/251775
                    df_time.bfill(axis=0, inplace=True)

                    # For spans with no trades, values of traded_qty and traded_count are set to 0
                    trades_columns = np.repeat(np.nan, 3*600).reshape(3,600).astype(np.float32)
                    trades_columns[-2:] = 0.

                    series.append(np.vstack((df_time.T.to_numpy(dtype=np.float32), trades_columns)))

                    if 'target' in targets.columns:
                        entry = targets.loc[(targets.stock_id==stock_id) & (targets.time_id==time_id), 'target']
                    else:
                        entry = []

                    vols.append(np.array((  stock_id, time_id, len(vols), 
                                            entry.item() if len(entry)==1 else np.nan), dtype=np.float32))

                elif 'trade' in file_path:

                    if isinstance(vols, list):
                        series = np.stack(series, axis=0)
                        vols = np.stack(vols, axis=0)

                    # Avg trade prices are only forward-filled, nan values will be replaced with WAP later
                    df_time = df_time.iloc[:,2:].fillna({'size':0, 'order_count':0})
                    df_time.ffill(axis=0, inplace=True)

                    tensor_index = vols[(vols[:,0]==stock_id) & (vols[:,1]==time_id), 2].item()
                    series[int(tensor_index),-3:] = df_time.T.to_numpy(dtype=np.float32)

    return series, vols

def h_poly_helper(tt):
  A = T.tensor([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=tt[-1].dtype)
  return [
    sum( A[i, j]*tt[j] for j in range(4) )
    for i in range(4) ]

def h_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = 1
  for i in range(1, 4):
    tt[i] = tt[i-1]*t
  return h_poly_helper(tt)

def H_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = t
  for i in range(1, 4):
    tt[i] = tt[i-1]*t*i/(i+1)
  return h_poly_helper(tt)

def interp(x, y, xs):
  m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
  m = T.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  I = P.searchsorted(x[1:], xs)
  dx = (x[I+1]-x[I])
  hh = h_poly((xs-x[I])/dx)
  return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

def integ(x, y, xs):
  m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
  m = T.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  I = P.searchsorted(x[1:], xs)
  Y = T.zeros_like(y)
  Y[1:] = (x[1:]-x[:-1])*(
      (y[:-1]+y[1:])/2 + (m[:-1] - m[1:])*(x[1:]-x[:-1])/12
      )
  Y = Y.cumsum(0)
  dx = (x[I+1]-x[I])
  hh = H_poly((xs-x[I])/dx)
  return Y[I] + dx*(
      hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
      )

def smooth_bce(logits, targets, smoothing=0.05, weight=None):

    with torch.no_grad():
        smooth_targets = targets * (1.0 - smoothing) + 0.5 * smoothing

    return F.binary_cross_entropy_with_logits(logits.squeeze(), smooth_targets.squeeze(), weight=weight)

# Example
if __name__ == "__main__":
  x = T.linspace(0, 6, 7)
  y = x.sin()
  xs = T.linspace(0, 6, 101)
  ys = interp(x, y, xs)
  Ys = integ(x, y, xs)
  P.scatter(x, y, label='Samples', color='purple')
  P.plot(xs, ys, label='Interpolated curve')
  P.plot(xs, xs.sin(), '--', label='True Curve')
  P.plot(xs, Ys, label='Spline Integral')
  P.plot(xs, 1-xs.cos(), '--', label='True Integral')
  P.legend()
  P.show()

