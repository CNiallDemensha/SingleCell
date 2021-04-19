import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging

import numpy as np
from pprint import pformat, pprint
import sklearn
from sklearn.cluster import KMeans
from hparams import HParams

import torch
import torch.optim as optim

from model import GMM, ConditionalGMM

logging.getLogger("sklearn").setLevel(level=logging.ERROR)      # Suppress sklearn WARN messages about AMI

data_path = "/nas/longleaf/home/athreya/gmm/data/"
# data_path = ''

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--split_num', type=str)
parser.add_argument('--model_name', type=str, required=False)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device -: {}".format(device))
torch.autograd.set_detect_anomaly(True)

# creat exp dir
if not os.path.exists(params.exp_dir):
    os.mkdir(params.exp_dir)
if not os.path.exists(os.path.join(params.exp_dir, 'gen')):
    os.mkdir(os.path.join(params.exp_dir, 'gen'))
if not os.path.exists(os.path.join(params.exp_dir, 'ckpt')):
    os.mkdir(os.path.join(params.exp_dir, 'ckpt'))

split = args.split_num
train_data = np.load(data_path + 'Levine_32_matrix_train_split{}.npy'.format(split))
train_label = train_data[:, -1] - 1
train_data = train_data[:, :-1]
inds = np.random.permutation(train_data.shape[0])
train_data = torch.Tensor(train_data[inds]).to(device)

test_data = np.load(data_path + 'Levine_32_matrix_test_split{}.npy'.format(split))
test_label = test_data[:, -1] - 1
test_data = test_data[:, :-1]
test_data = torch.Tensor(test_data).to(device)

model = ConditionalGMM(params.k, train_data.shape[1], params.tau)
if(args.model_name):
    print("loading model {}".format(args.model_name))
    model.load_state_dict(torch.load(os.path.join(params.exp_dir, "ckpt", args.model_name)))

model = model.to(device)
#model = GMM(params.k, train_data.shape[1], params.tau).to(device)
optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
best_loss = 1e7

model_string = str(model).lower()
if(model_string.startswith("conditional")):
    model_type = "conditional"
elif(model_string.startswith("independent")):
    model_type = "independent"
else:
    model_type = "vanilla"

train_epoch_loss = []
test_epoch_loss = []
epoch_v_score = []
epoch_ari_score = []
epoch_ami_score = []
print("Starting training")
for e in range(params.epochs):
    model.train()
    iteration_loss = []
    inds = np.random.permutation(train_data.shape[0])
    train_data = train_data[inds]
    lam = 0 if e == 0 else 0.5
    for i in range(0, train_data.shape[0], params.batch_size):
        batch_data = train_data[i:i+params.batch_size]
        nll, _ = model.negative_log_prob(batch_data)
        l1_loss = torch.norm(torch.sigmoid(10 * model.alpha), 1)
        total_loss = nll + lam * l1_loss
        if(i%10000 == 0 and i > 0):
            print("Iteration {} / {} - Loss (nll, l1_loss, total) = {}".format(i, train_data.shape[0], [nll.item(), l1_loss.item(), total_loss.item()]))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # # Update alpha values by applying soft-thresholding
        # for name, param in model.named_parameters():
        #     if(name == 'alpha'):
        #         print("alpha[0] -: {}".format(model.alpha[0]))
        #         print("Applying soft-thresholding")
        #         penalty = lam*params.lr
        #         with torch.no_grad():
        #             param.data.copy_(torch.sign(param.data)*torch.clamp(torch.abs(param.data) - penalty, min=0.0))
        #         print("alpha[0] -: {}".format(model.alpha[0]))

        iteration_loss.append(nll.detach().cpu().item())
        
    train_epoch_loss.append(np.mean(iteration_loss))
    print("Average train epoch loss for epoch {} = {}".format(e+1, train_epoch_loss[-1]))
    model.eval()
    iteration_loss = []
    hist = dict()
    pred_labels = []
    print("Epoch {} - Eval mode".format(e+1))
    with torch.no_grad():
        for i in range(0, test_data.shape[0], params.batch_size):
            batch_data = test_data[i:i+params.batch_size]
            nll, assignment = model.negative_log_prob(batch_data)
            # nll.backward()                                                    # Turned off grad computation in eval mode
            i += params.batch_size
            iteration_loss.append(nll.detach().cpu().item())
            labels = assignment.detach().cpu().numpy()
            pred_labels.append(labels)
            for j in range(params.k):
                if j in hist:
                    hist[j] += np.sum(labels == j)
                else:
                    hist[j] = np.sum(labels == j)

    lr_scheduler.step()

    pred_labels = np.concatenate(pred_labels)
    print("Test label shape = {}, Predicted label shape = {}".format(test_label.shape, pred_labels.shape))
    test_label = test_label[:pred_labels.shape[0]]                                      # Verify if this indexing is needed? test_label and pred_labels should be same size after each epoch

    test_epoch_loss.append(np.mean(iteration_loss))
    v = sklearn.metrics.v_measure_score(test_label, pred_labels)
    ari = sklearn.metrics.adjusted_rand_score(test_label, pred_labels)
    ami = sklearn.metrics.adjusted_mutual_info_score(test_label, pred_labels)
    print("Epoch {} - Test Loss = {}, Vscore = {}, ARI = {}, AMI = {}".format(e+1, test_epoch_loss[-1], v, ari, ami))
    epoch_v_score.append(v)
    epoch_ari_score.append(ari)
    epoch_ami_score.append(ami)

    plt.figure()
    plt.bar(range(params.k), [hist[each] for each in hist])
    plt.savefig('hist.png')
    plt.close()

    plt.figure()
    plt.bar(range(params.k), torch.softmax(params.tau * model.pi, 0).detach().cpu().numpy())
    plt.savefig('dist.png')
    plt.close()

    if test_epoch_loss[-1] < best_loss:
        best_loss = test_epoch_loss[-1]
        torch.save(model.state_dict(), params.exp_dir + '/ckpt/{}_{}_{}_{:.0e}_{}_split{}_epoch_{}.pt'.format(model_type, str((train_data.shape[0] + test_data.shape[0])//1000) + 'k', str(params.epochs) + 'epochs', params.lr, "5e-01reg", split, e+1))
    else:
        print("Test loss {} not better than previous best test loss {}. Skipping saving model".format(test_epoch_loss[-1], best_loss))

print("Training finished")
print("Epoch level Train_Loss, Test_Loss, V_Score, ARI, AMI -:")
print(train_epoch_loss)
print(test_epoch_loss)
print(epoch_v_score)
print(epoch_ari_score)
print(epoch_ami_score)

plt.figure()
plt.plot(range(1, params.epochs + 1), train_epoch_loss, label="train_epoch_loss")
plt.plot(range(1, params.epochs + 1), test_epoch_loss, label="test_epoch_loss")
plt.plot(range(1, params.epochs + 1), epoch_v_score, label="epoch_v_score")
plt.plot(range(1, params.epochs + 1), epoch_ari_score, label="epoch_ari_score")
plt.plot(range(1, params.epochs + 1), epoch_ami_score, label="epoch_ami_score")
plt.xlabel("Epochs")
plt.savefig("metrics.png")


# Comparing with Baseline KMeans
kmeans = KMeans(n_clusters=params.k, max_iter=300).fit(train_data.cpu().numpy())
kmeans_preds = kmeans.predict(test_data.cpu().numpy())
kmeans_v = sklearn.metrics.v_measure_score(test_label, kmeans_preds)
kmeans_ari = sklearn.metrics.adjusted_rand_score(test_label, kmeans_preds)
kmeans_ami = sklearn.metrics.adjusted_mutual_info_score(test_label, kmeans_preds)

print("KMeans V_Score, ARI, AMI -: {}, {}, {}".format(kmeans_v, kmeans_ari, kmeans_ami))
print()
print("Model alpha")
print()
print(torch.sigmoid(model.alpha))
