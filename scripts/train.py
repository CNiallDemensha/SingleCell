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

train_data = np.load('/content/drive/My Drive/COMP 991/Levine_32_matrix_train.npy')
train_data = train_data[:, :-1]
inds = np.random.permutation(train_data.shape[0])
train_data = torch.Tensor(train_data[inds]).to(device)

test_data = np.load('/content/drive/My Drive/COMP 991/Levine_32_matrix_test.npy')
test_label = test_data[:, -1] - 1
test_data = test_data[:, :-1]
test_data = torch.Tensor(test_data).to(device)

# model = ConditionalGMM(params.k, train_data.shape[1], params.tau).to(device)
model = GMM(params.k, train_data.shape[1], params.tau).to(device)
optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
best_loss = 1e7

#labels, clusters = model.kmeans_clustering(train_data.cpu().numpy())
for e in range(params.epochs):
    model.train()
    epoch_losses = []
    inds = np.random.permutation(train_data.shape[0])
    train_data = train_data[inds]
    lam = 0 #if e == 0 else 0.1
    for i in tqdm(range(0, train_data.shape[0], params.batch_size)):
        batch_data = train_data[i:i+params.batch_size]
        nll, _ = model.negative_log_prob(batch_data)
        #l1_loss = torch.norm(torch.sigmoid(10 * model.alpha), 1)
        total_loss = nll #+ lam * l1_loss
        #print(nll, l1_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_losses.append(nll.detach().cpu().item())
        #
    train_epoch_loss = np.mean(epoch_losses)
    #print(torch.sigmoid(model.alpha))
    model.eval()
    epoch_losses = []
    hist = dict()
    pred_labels = []
    for i in tqdm(range(0, test_data.shape[0], params.batch_size)):
        batch_data = test_data[i:i+params.batch_size]
        nll, assignment = model.negative_log_prob(batch_data)
        nll.backward()
        i += params.batch_size
        epoch_losses.append(nll.detach().cpu().item())
        labels = assignment.detach().cpu().numpy()
        pred_labels.append(labels)
        for j in range(params.k):
            if j in hist:
                hist[j] += np.sum(labels == j)
            else:
                hist[j] = np.sum(labels == j)
    pred_labels = np.concatenate(pred_labels)
    test_label = test_label[:pred_labels.shape[0]]
    v = sklearn.metrics.v_measure_score(test_label, pred_labels)

    plt.figure()
    plt.bar(range(params.k), [hist[each] for each in hist])
    plt.savefig('hist.png')
    plt.close()

    plt.figure()
    plt.bar(range(params.k), torch.softmax(params.tau * model.pi, 0).detach().cpu().numpy())
    plt.savefig('dist.png')
    plt.close()

    test_epoch_loss = np.mean(epoch_losses)

    
    
    
    #print(hist)
    print(f"[{e}]/[{params.epochs}], Train Loss: {train_epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}, Test V Score: {v:.4f}")
    #print(f"[{e}]/[{params.epochs}], Train Loss: {train_epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}")
    if test_epoch_loss < best_loss:
        best_test_loss = test_epoch_loss
        torch.save(model.state_dict(), params.exp_dir + '/ckpt/epoch_%d.pt' % e)