# @Time     : Mar. 14, 2023
# @Author   : Josh Miller / Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : VSCode / PyCharm
# @Github   : ???

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf
print("tensorflow.__version__", tf.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from utils.distance_matrix import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=350)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')

args = parser.parse_args()
print(f'Training configs: {args}')

n_his, n_pred, batch_size, epoch = args.n_his, args.n_pred, args.batch_size, args.epoch
Ks, Kt = args.ks, args.kt


# blocks: settings of channel size in st_conv_blocks / bottleneck design
#blocks = [[1, 32, 64], [64, 32, 128]]
#blocks=[[64,64],[64,64],[64,64],[64,64]]
blocks=[[64,64]]

''' Read in tempurature data '''
df = pd.read_csv('./data_loader/data/tmax_2016_1_1-2016_12_31_krig_grid.csv')
print(df.head())
df = df.values
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~', np.shape(df))
lon = df[::10, 0]
lat = df[::10, 1]
tmax = df[::10, 2:] # (spatial samples, time)

del(df)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~', np.shape(lat), np.shape(lon), np.shape(tmax))
print("READ CSV")

tmax = pd.DataFrame(tmax.T) # Make time the first axis
tmax.to_csv('./data_loader/data/tmax_only_2016_1_1-2016_12_31_krig_grid.csv', index=False)

''' Load/create distance matrix (if needed) '''
dist_matrix_path = './data_loader/data/distance_matrix.csv'

if os.path.exists(dist_matrix_path):
    print("\"" + dist_matrix_path + "\" exists")
else:
    print("\"" + dist_matrix_path + "\" not found. Creating distance matrix.")

    coords = np.transpose(np.array([lon, lat]))
    #print('~~~~~~~~~~~~~~~~~~~~~~~~~', coords)

    dist_matrix = distance_matrix(coords)
    
    dist_matrix = pd.DataFrame(dist_matrix, header=None)
    dist_matrix.to_csv(dist_matrix_path, index=False)

    del(dist_matrix)

if args.graph == 'default':
    W = weight_matrix(dist_matrix_path)
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('../dataset', args.graph))

# Calculate graph kernel
L = scaled_laplacian(W)
V,U=np.linalg.eig(L)
U_ini=U[:,:32]
print(U_ini.shape)
tf.add_to_collection(name='intial_spatial_embeddings', value=tf.cast(tf.constant(U_ini), tf.float32))

# Alternative approximation method: 1st approx - first_approx(W, n).
n = np.shape(lon)[0]
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
data_file = 'tmax_only_2016_1_1-2016_12_31_krig_grid.csv'
train_ratio, val_ratio, test_ratio = .7, .15, .15

tmax_Dataset = data_gen(pjoin('./data_loader/data/', data_file), (train_ratio, val_ratio, test_ratio), n_his + n_pred)
print(f'>> Loading dataset with Mean: {tmax_Dataset.mean:.2f}, STD: {tmax_Dataset.std:.2f}')

if __name__ == '__main__':
    #model_train(tmax_Dataset, blocks, args)
    print('==============================================================')
    print('==============================================================')
    model_test(tmax_Dataset,args.batch_size, n_his, n_pred, args.inf_mode, plot=False, lon=lon, lat=lat)