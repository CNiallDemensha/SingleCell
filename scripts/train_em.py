import os
import sys
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import torch
import numpy as np
from pprint import pformat, pprint
import sklearn
from sklearn.mixture import GaussianMixture
import random
from hparams import HParams
#from utils.train_utils import run_epoch, get_gap_lr_bs
from PIL import Image
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim

from model import GMM, ConditionalGMM

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# creat exp dir
if not os.path.exists(params.exp_dir):
    os.mkdir(params.exp_dir)
if not os.path.exists(os.path.join(params.exp_dir, 'gen')):
    os.mkdir(os.path.join(params.exp_dir, 'gen'))
if not os.path.exists(os.path.join(params.exp_dir, 'ckpt')):
    os.mkdir(os.path.join(params.exp_dir, 'ckpt'))

train_data = np.load('/playpen1/scribble/ssy/data/Levine_32_matrix_train.npy')
train_data = train_data[:, :-1]
inds = np.random.permutation(train_data.shape[0])
#train_data = torch.Tensor(train_data[inds]).to(device)

test_data = np.load('/playpen1/scribble/ssy/data/Levine_32_matrix_test.npy')
test_label = test_data[:, -1] - 1
test_data = test_data[:, :-1]
#test_data = torch.Tensor(test_data).to(device)

gm = GaussianMixture(n_components=14, covariance_type='diag', random_state=0, max_iter=100000).fit(train_data)
pred = gm.predict(test_data)
print(sklearn.metrics.v_measure_score(test_label, pred))