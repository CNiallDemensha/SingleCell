import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from sklearn.cluster import KMeans
import time

class Timer(object):
    def __init__(self, name='Operation'):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('%s took: %s sec' % (self.name, time.time() - self.tstart))

class GMM(nn.Module):
    ''' A GMM model. '''
    def __init__(
            self, k, d, tau):
        super().__init__()
        self.k = k
        self.d = d
        self.A = nn.Parameter(0.01 * torch.randn(self.k, self.d, self.d), requires_grad=True)
        self.D = nn.Parameter(torch.ones(self.k, self.d), requires_grad=True)
        self.mu = nn.Parameter(torch.zeros(self.k, self.d), requires_grad=True)
        self.pi = nn.Parameter(0.01 * torch.randn(self.k), requires_grad=True)
        self.tau = tau
    def negative_log_prob(self, x):
        sigma = torch.diag_embed(torch.nn.functional.softplus(self.D)) + torch.matmul(self.A, self.A.transpose(1,2))
        sigma = sigma.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # [bs, k, d, d]
        mu = self.mu.unsqueeze(0).repeat(x.shape[0], 1, 1) # [bs, k, d]
        comp = D.MultivariateNormal(mu, sigma)
        x = x.unsqueeze(1).repeat(1, self.k, 1) # [bs, k, d]
        mix_p = torch.log(torch.softmax(self.tau * self.pi, 0).unsqueeze(0)) + comp.log_prob(x)
        assignment = torch.argmax(mix_p, 1)
        nll = - torch.logsumexp(mix_p, dim=1)
        return nll.mean(), assignment
    def kmeans_clustering(self, samples, get_centers=True):
        N, d = samples.shape
        K = self.k
        # Select random d_used coordinates out of d
        d_used = min(d, max(500, d//8))
        d_indices = np.random.choice(d, d_used, replace=False)
        print('Performing k-means clustering to {} components of {} samples in dimension {}/{} ...'.format(K, N, d_used, d))
        with Timer('K-means'):
            clusters = KMeans(n_clusters=K, max_iter=300, n_jobs=-1).fit(samples[:, d_indices])
        labels = clusters.labels_
        if get_centers:
            centers = np.zeros([K, d])
            for i in range(K):
                centers[i, :] = np.mean(samples[labels == i, :], axis=0)
            self.mu = nn.Parameter(torch.Tensor(centers).to(self.mu.device), requires_grad=True)
            return labels, centers
        return labels
    def check_assignment(self, x):
        sigma = torch.diag_embed(torch.nn.functional.softplus(self.D)) + torch.matmul(self.A, self.A.transpose(1,2))
        sigma = sigma.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # [bs, k, d, d]
        mu = self.mu.unsqueeze(0).repeat(x.shape[0], 1, 1) # [bs, k, d]
        comp = D.MultivariateNormal(mu, sigma)
        x = x.unsqueeze(1).repeat(1, self.k, 1) # [bs, k, d]
        labels = torch.argmax(comp.log_prob(x), 1)
        return labels.detach().cpu().numpy()


class ConditionalGMM(nn.Module):
    ''' A GMM model. '''
    def __init__(
            self, k, d, tau):
        super().__init__()
        self.k = k
        self.d = d
        self.f_A = nn.Parameter(0.01 * torch.randn(self.k, self.d, self.d), requires_grad=True)
        self.f_D = nn.Parameter(torch.ones(self.k, self.d), requires_grad=True)                 # Why is D needed for creating positive definite matrix?
        self.f_mu = nn.Parameter(torch.zeros(self.k, self.d), requires_grad=True)
        self.b_A = nn.Parameter(0.01 * torch.randn(1, self.d, self.d), requires_grad=True)
        self.b_D = nn.Parameter(torch.ones(1, self.d), requires_grad=True)
        self.b_mu = nn.Parameter(torch.zeros(1, self.d), requires_grad=True)
        self.pi = nn.Parameter(0.01 * torch.randn(self.k), requires_grad=True)
        self.alpha = nn.Parameter(0.01 * torch.randn(self.k, self.d), requires_grad=True)
        self.tau = tau

    def get_conditional_distribution(self, x, f_mu, f_sigma, b_mu, b_sigma):
        f_d = []
        b_d = []
        f_d.append(D.MultivariateNormal(f_mu[:, :, :1], f_sigma[:,:, :1, :1]))
        b_d.append(D.MultivariateNormal(b_mu[:, :, :1], b_sigma[:,:, :1, :1]))
        for i in range(1, self.d):
            f_inverse_matrix = torch.inverse(f_sigma[:, :, :i, :i])
            f_cond_mu = f_mu[:, :, i:i+1, None] + f_sigma[:, :, i:i+1, :i].matmul(f_inverse_matrix).matmul(x[:, :, :i, None] - f_mu[:, :, :i, None])
            f_cond_mu = f_cond_mu[:, :, :, 0]
            f_cond_sigma = f_sigma[:, :, i:i+1, i:i+1] - f_sigma[:, :, i:i+1, :i].matmul(f_inverse_matrix).matmul(f_sigma[:, :, :i, i:i+1])
            f_d.append(D.MultivariateNormal(f_cond_mu, f_cond_sigma))

            b_inverse_matrix = torch.inverse(b_sigma[:, :, :i, :i])
            b_cond_mu = b_mu[:, :, i:i+1, None] + b_sigma[:, :, i:i+1, :i].matmul(b_inverse_matrix).matmul(x[:, :, :i, None] - b_mu[:, :, :i, None])
            b_cond_mu = b_cond_mu[:, :, :, 0]
            b_cond_sigma = b_sigma[:, :, i:i+1, i:i+1] - b_sigma[:, :, i:i+1, :i].matmul(b_inverse_matrix).matmul(b_sigma[:, :, :i, i:i+1])
            b_d.append(D.MultivariateNormal(b_cond_mu, b_cond_sigma))
        return f_d, b_d

    def negative_log_prob(self, x):
        f_sigma = torch.diag_embed(torch.nn.functional.softplus(self.f_D)) + torch.matmul(self.f_A, self.f_A.transpose(1,2))        # A*A.T (for symmetric) + n*eye (for diagonally dominant) ??f
        f_sigma = f_sigma.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # [bs, k, d, d]
        f_mu = self.f_mu.unsqueeze(0).repeat(x.shape[0], 1, 1) # [bs, k, d]
        b_sigma = torch.diag_embed(torch.nn.functional.softplus(self.b_D)) + torch.matmul(self.b_A, self.b_A.transpose(1,2))
        b_sigma = b_sigma.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # [bs, 1, d, d]
        b_mu = self.b_mu.unsqueeze(0).repeat(x.shape[0], 1, 1) # [bs, 1, d]
        x = x.unsqueeze(1).repeat(1, self.k, 1) # [bs, k, d]
        f_d, b_d = self.get_conditional_distribution(x, f_mu, f_sigma, b_mu, b_sigma)
        #comp = D.MultivariateNormal(mu, sigma)
        
        mix_p = torch.log(torch.softmax(self.tau * self.pi, 0).unsqueeze(0))
        for i in range(len(f_d)):
            alpha = torch.sigmoid(10 * self.alpha)[:, i]
            mix_p = mix_p + alpha.unsqueeze(0) * f_d[i].log_prob(x[:,:,i:i+1]) + (1 - alpha.unsqueeze(0)) * b_d[i].log_prob(x[:,:,i:i+1])
        assignment = torch.argmax(mix_p, 1)
        nll = - torch.logsumexp(mix_p, dim=1)
        return nll.mean(), assignment
    def kmeans_clustering(self, samples, get_centers=True):
        N, d = samples.shape
        K = self.k
        # Select random d_used coordinates out of d
        d_used = min(d, max(500, d//8))
        d_indices = np.random.choice(d, d_used, replace=False)
        print('Performing k-means clustering to {} components of {} samples in dimension {}/{} ...'.format(K, N, d_used, d))
        with Timer('K-means'):
            clusters = KMeans(n_clusters=K, max_iter=300, n_jobs=-1).fit(samples[:, d_indices])
        labels = clusters.labels_
        if get_centers:
            centers = np.zeros([K, d])
            for i in range(K):
                centers[i, :] = np.mean(samples[labels == i, :], axis=0)
            self.mu = nn.Parameter(torch.Tensor(centers).to(self.mu.device), requires_grad=True)
            return labels, centers
        return labels
    def check_assignment(self, x):
        sigma = torch.diag_embed(torch.nn.functional.softplus(self.D)) + torch.matmul(self.A, self.A.transpose(1,2))
        sigma = sigma.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # [bs, k, d, d]
        mu = self.mu.unsqueeze(0).repeat(x.shape[0], 1, 1) # [bs, k, d]
        comp = D.MultivariateNormal(mu, sigma)
        x = x.unsqueeze(1).repeat(1, self.k, 1) # [bs, k, d]
        labels = torch.argmax(comp.log_prob(x), 1)
        return labels.detach().cpu().numpy()



class IndependentGMM(nn.Module):
    ''' A GMM model. '''
    def __init__(
            self, k, d, tau):
        super().__init__()
        self.k = k
        self.d = d
        self.f_mu = nn.Parameter(torch.zeros(self.k, self.d), requires_grad=True)
        self.b_mu = nn.Parameter(torch.zeros(1, self.d), requires_grad=True)
        self.pi = nn.Parameter(0.01 * torch.randn(self.k), requires_grad=True)
        self.alpha = nn.Parameter(0.01 * torch.randn(self.k, self.d), requires_grad=True)
        self.tau = tau

    def negative_log_prob(self, x):
        # sigma = torch.diag_embed(torch.nn.functional.softplus(self.D)) #+ torch.matmul(self.A, self.A.transpose(1,2))
        # sigma = sigma.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # [bs, k, d, d]
        f_mu = self.f_mu.unsqueeze(0).repeat(x.shape[0], 1, 1)  # [bs, k, d]
        b_mu = self.b_mu.unsqueeze(0).repeat(x.shape[0], 1, 1)  # [bs, 1, d]
        f_comp = D.Normal(f_mu, 1.0)
        b_comp = D.Normal(b_mu, 1.0)
        x = x.unsqueeze(1).repeat(1, self.k, 1)  # [bs, k, d]
        alpha = torch.sigmoid(10 * self.alpha)
        mix_p = torch.log(torch.softmax(self.tau * self.pi, 0).unsqueeze(0)) + \
                torch.sum(alpha.unsqueeze(0) * f_comp.log_prob(x), -1) + \
                torch.sum((1 - alpha.unsqueeze(0)) * b_comp.log_prob(x), -1)
        assignment = torch.argmax(mix_p, 1)
        nll = - torch.logsumexp(mix_p, dim=1)
        return nll.mean(), assignment


if __name__ == "__main__":
    model = GMM(10, 64)
    data = torch.zeros(32, 64)
    print(model.negative_log_prob(data))

