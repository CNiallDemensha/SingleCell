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
from scipy.stats import norm
from torch.utils.tensorboard import SummaryWriter
from hparams import HParams
#from utils.train_utils import run_epoch, get_gap_lr_bs
from PIL import Image
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter

from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import GMM, ConditionalGMM, FB_GMM, Variational_GMM_beta
from PIL import Image

def soft_thresholding_operator(x, l):
    idx = torch.where(x>l)
    x[idx] = x[idx] - l
    idx = torch.where(x<-l)
    x[idx] = x[idx] + l
    idx = torch.where((x>=-l).float() * (x<=l).float())
    x[idx] = 0
    return torch.clip(x, 0, 1)


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

writer = SummaryWriter(params.exp_dir)
#train_data = np.load('/playpen1/scribble/ssy/data/Levine_32_matrix_train.npy')
#train_data = train_data[:, :-1]
#inds = np.random.permutation(train_data.shape[0])
#train_data = torch.Tensor(train_data[inds]).to(device)

#test_data = np.load('/playpen1/scribble/ssy/data/Levine_32_matrix_test.npy')
#test_label = test_data[:, -1] - 1
#test_data = test_data[:, :-1]
#test_data = torch.Tensor(test_data).to(device)
orig_test_data = np.load('/playpen1/scribble/ssy/data/Levine_32_matrix_test.npy')
orig_test_label = orig_test_data[:, -1] - 1
orig_test_data = orig_test_data[:, :-1]
inds = np.random.permutation(orig_test_data.shape[0])
orig_test_data = orig_test_data[inds]
orig_test_label = orig_test_label[inds]
train_data = torch.tensor(orig_test_data[:int(0.8*orig_test_data.shape[0])]).to(device)
test_label = orig_test_label[int(0.8*orig_test_data.shape[0]):]
test_data = torch.tensor(orig_test_data[int(0.8*orig_test_data.shape[0]):]).to(device)
#train_data = test_data
#model = ConditionalGMM(params.k, train_data.shape[1], params.tau).to(device)
#centers = np.load('./centers.npy')
#model = GMM(params.k, train_data.shape[1], params.tau).to(device)
model = Variational_GMM_beta(params.k, train_data.shape[1]).to(device)


optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
best_loss = 1e7



#labels, clusters = model.kmeans_clustering(train_data.cpu().numpy())
#beta_dist = torch.distributions.beta.Beta(torch.tensor([0.5]).to(device), torch.tensor([0.5]).to(device))

#mode_0_std = np.linspace(0.2, 0.02, 20)
#mode_1_std = np.linspace(0.5, 0.1, 20)
#threshold = 3
start_epoch = 0
if params.resume_from_epoch > 0:
    start_epoch = params.resume_from_epoch
    checkpoint = torch.load(os.path.join(params.exp_dir, 'ckpt', "epoch_%d.pt" % params.resume_from_epoch))
    model.load_state_dict(checkpoint)
T = np.concatenate([np.linspace(5, 0.1, 200), 0.1 * np.ones(params.epochs - 200)])
for e in range(start_epoch, params.epochs):
    model.train()
    epoch_losses = []
    inds = np.random.permutation(train_data.shape[0])
    train_data = train_data[inds]
    if e < 160:
        model.beta_prior_alpha.requires_grad = False
        model.beta_prior_beta.requires_grad = False
    else:
        model.beta_prior_alpha.requires_grad = True
        model.beta_prior_beta.requires_grad = True
    #model.mu.requires_grad=False
    #model.pi.requires_grad=True
    model.D.requires_grad=False
    #model.b_D.requires_grad=False
    #normal_dist_0 = TruncatedNormal(torch.tensor([0.0]).to(device), torch.tensor([mode_0_std[e]]).to(device), a=torch.tensor([0.0]).to(device), b=torch.tensor([1.0]).to(device))
    #normal_dist_1 = TruncatedNormal(torch.tensor([1.0]).to(device), torch.tensor([mode_1_std[e]]).to(device), a=torch.tensor([0.0]).to(device), b=torch.tensor([1.0]).to(device))
    for i in tqdm(range(0, train_data.shape[0], params.batch_size)):
        batch_data = train_data[i:i+params.batch_size]
        elbo, log_p_x_given_alpha_z, kl_alpha_beta, kl_q_pi, kl_Q_PI, pred, alpha_posterior_prob, z_prob = model(batch_data, T[e])
        #weight = torch.norm(beta_prior_prob, p=1, dim=1).unsqueeze(-1) 
        #weight = weight/weight.max()
        #l1_loss = torch.norm(beta_prior_prob, p=1)
        total_loss = - elbo #+ lam * l1_loss
        #total_loss = nll + lam * beta_loss
        #print(model.mu)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_losses.append(total_loss.detach().cpu().item())
        #print(model.alpha)
        #
    train_epoch_loss = np.mean(epoch_losses)
    writer.add_scalar("log_p_x_given_alpha_z", log_p_x_given_alpha_z, e)
    writer.add_scalar("kl_alpha_beta", kl_alpha_beta, e)
    writer.add_scalar("kl_q_pi", kl_q_pi, e)
    writer.add_scalar("kl_Q_PI", kl_Q_PI, e)
    writer.add_scalar("mean_beta_D_alpha", torch.sigmoid(5*model.beta_prior_alpha).mean(), e)
    writer.add_scalar("mean_beta_D_beta", torch.sigmoid(5*model.beta_prior_beta).mean(), e)
    writer.add_image("alpha_posterior_prob", alpha_posterior_prob[None, :, :], e)
    writer.add_image("z_prior", model.prior_pi[None, :, :], e)
    writer.add_image("z_prob", z_prob[None, :, :], e)
    writer.add_image("beta_D_alpha", torch.sigmoid(5*model.beta_prior_alpha)[None, :, :], e)
    writer.add_image("beta_D_beta", torch.sigmoid(5*model.beta_prior_beta)[None, :, :], e)
    writer.add_histogram("hist_beta_D_alpha", torch.sigmoid(5*model.beta_prior_alpha), e)
    writer.add_histogram("hist_beta_D_beta", torch.sigmoid(5*model.beta_prior_beta), e)
    
    writer.flush()
    model.eval()
    epoch_losses = []
    hist = dict()
    pred_labels = []
    v_scores = []
    for n in tqdm(range(0, test_data.shape[0], params.batch_size)):
        batch_data = test_data[n:n+params.batch_size]
        batch_label = test_label[n:n+params.batch_size]
        elbo, log_p_x_given_alpha_z, kl_alpha_beta, kl_q_pi, kl_Q_PI, pred, alpha_posterior_prob, z_prob = model(batch_data, T[e])
        total_loss = - elbo
        total_loss.backward()
        pred = pred.detach().cpu().numpy()
        v_score = sklearn.metrics.v_measure_score(pred, batch_label)
        epoch_losses.append(total_loss.detach().cpu().item())
        v_scores.append(v_score)
        
    test_epoch_loss = np.mean(epoch_losses)
    writer.add_scalar("V-score", np.mean(v_scores), e)

    
    
    
    #print(hist)
    print(f"[{e}]/[{params.epochs}], Train Loss: {train_epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}, V-score: {np.mean(v_scores):.4f}")
    #print(f"[{e}]/[{params.epochs}], Train Loss: {train_epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}")
    if test_epoch_loss < best_loss:
        best_test_loss = test_epoch_loss
        torch.save(model.state_dict(), params.exp_dir + '/ckpt/epoch_%d.pt' % e)
    if False:
        fg_idx = {
            0 : [13,29], # 0, 5, 11, 12, 13, 17, 19, 23, 24, 25, 28, 29
            1: [19], #
            2: [19, 21], # 19, 21, 22, 29
            3: [7, 23],
            4: [7, 23, 13],
            5: [7],
            6: [25, 5], # 0,  5, 11, 12, 13, 17, 19, 23, 24, 25, 28, 29
            7: [25, 6], # 6, 11, 12, 13, 19, 23, 24, 25, 28, 29
            8: [2],
            9: [15],
            10: [13, 29], # 0,  5, 11, 12, 13, 17, 19, 23, 24, 25, 28, 29,
            11: [2, 23], # 2, 10, 12, 17, 22, 23,
            12: [2, 9], # 0,  2,  3,  9, 14, 23, 24, 29
            13: [2]
        }
        
        learned_mask = beta_prior_prob.detach().cpu().numpy()
        learned_mask = (learned_mask > 0.5).astype(np.int32)
        gt_mask = np.zeros_like(learned_mask, dtype=np.int32)
        for ii in range(14):
            for jj in range(len(fg_idx[ii])):
                gt_mask[ii, fg_idx[ii][jj]] = 1.0
        dist = np.abs(learned_mask[:, None, :] - gt_mask[None, :, :]).sum(-1)
        row_ind, col_ind = linear_sum_assignment(dist)
        for ii in range(14):
            print(f'learned cluster {ii} has foreground features: {np.where(learned_mask[ii]==1)} and matched to ground-truth foreground annotations: {fg_idx[col_ind[ii]]}.')
