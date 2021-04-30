import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributions as D
import numpy as np
from sklearn.cluster import KMeans
import time
import pdb
import torch.nn.functional as F

class Timer(object):
    def __init__(self, name='Operation'):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('%s took: %s sec' % (self.name, time.time() - self.tstart))
class FB_GMM(nn.Module):
    ''' A GMM model. '''
    def __init__(
            self, k, d, tau, mu=None):
        super().__init__()
        self.k = k
        self.d = d
        self.A = nn.Parameter(0.01 * torch.randn(self.k, self.d, self.d), requires_grad=True)
        self.alpha = nn.Parameter(0.01 * torch.randn(self.k, self.d), requires_grad=True)
        self.D = nn.Parameter(torch.ones(self.k, self.d), requires_grad=True)
        self.f_mu = nn.Parameter(torch.zeros(self.k, self.d), requires_grad=True)
        self.b_mu = nn.Parameter(torch.zeros(1, self.d), requires_grad=True)
        self.pi = nn.Parameter(0.01 * torch.randn(self.k), requires_grad=True)
        self.tau = tau
    def negative_log_prob(self, x):
        #sigma = torch.diag_embed(torch.nn.functional.softplus(self.D)) #+ torch.matmul(self.A, self.A.transpose(1,2))
        #sigma = sigma.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # [bs, k, d, d]
        f_mu = self.f_mu.unsqueeze(0).repeat(x.shape[0], 1, 1) # [bs, k, d]
        b_mu = self.b_mu.unsqueeze(0).repeat(x.shape[0], 1, 1) # [bs, 1, d]
        f_comp = D.Normal(f_mu, 1.0)
        b_comp = D.Normal(b_mu, 1.0)
        x = x.unsqueeze(1).repeat(1, self.k, 1) # [bs, k, d]
        alpha = torch.sigmoid(5*self.alpha)
        #alpha = self.alpha
        mix_p = torch.log(torch.softmax(self.tau * self.pi, 0).unsqueeze(0)) +  \
                torch.sum(alpha.unsqueeze(0) * f_comp.log_prob(x), -1) + \
                torch.sum((1 - alpha.unsqueeze(0)) * b_comp.log_prob(x), -1)
        assignment = torch.argmax(mix_p, 1)
        nll = - torch.logsumexp(mix_p, dim=1)
        return nll.mean(), assignment
    def kmeans_clustering(self, samples, get_centers=True):
        N, d = samples.shape
        K = self.k
        # Select random d_used coordinates out of d
        #d_used = min(d, max(500, d//8))
        #d_indices = np.random.choice(d, d_used, replace=False)
        print('Performing k-means clustering to {} components of {} samples  ...'.format(K, N))
        with Timer('K-means'):
            clusters = KMeans(n_clusters=K, max_iter=300, n_jobs=-1).fit(samples)
        labels = clusters.labels_
        if get_centers:
            centers = np.zeros([K, d])
            for i in range(K):
                centers[i, :] = np.mean(samples[labels == i, :], axis=0)
            #self.mu = nn.Parameter(torch.Tensor(centers).to(self.mu.device), requires_grad=True)
            #pdb.set_trace()
            return labels, centers
        return labels
    def check_assignment(self, x):
        sigma = torch.diag_embed(torch.nn.functional.softplus(self.D)) #+ torch.matmul(self.A, self.A.transpose(1,2))
        sigma = sigma.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # [bs, k, d, d]
        mu = self.mu.unsqueeze(0).repeat(x.shape[0], 1, 1) # [bs, k, d]
        comp = D.MultivariateNormal(mu, sigma)
        x = x.unsqueeze(1).repeat(1, self.k, 1) # [bs, k, d]
        labels = torch.argmax(comp.log_prob(x), 1)
        return labels.detach().cpu().numpy()

class Variational_GMM_beta(nn.Module):
    def __init__(self, k, d, hidden_dims=128, num_layers=2):
        super().__init__()
        self.k = k
        self.d = d
        self.D = nn.Parameter(torch.ones(self.k, self.d), requires_grad=True)
        self.f_mu = nn.Parameter(torch.zeros(self.k, self.d), requires_grad=True)
        self.b_mu = nn.Parameter(torch.zeros(1, self.d), requires_grad=True)
        #self.prior_pi = nn.Parameter(0.1 * torch.randn(1, self.k), requires_grad=True)
        self.prior_pi = nn.Parameter(torch.ones(1, self.k) / self.k, requires_grad=True)
        self.q_net = [nn.Linear(d, hidden_dims), nn.ELU()]
        for i in range(num_layers):
            self.q_net += [nn.Linear(hidden_dims, hidden_dims), nn.ELU()]
        self.q_net += [nn.Linear(hidden_dims, k)]
        self.q_net = nn.Sequential(*self.q_net)
        self.r_net = [nn.Linear(d+k, hidden_dims), nn.ELU()]
        for i in range(num_layers):
            self.r_net += [nn.Linear(hidden_dims, hidden_dims), nn.ELU()]
        self.r_net += [nn.Linear(hidden_dims, d)]
        self.r_net = nn.Sequential(*self.r_net)
        self.beta_prior_alpha = nn.Parameter(0.01 * torch.randn(k, d), requires_grad=True)
        self.beta_prior_beta = nn.Parameter(0.01 * torch.randn(k, d), requires_grad=True)
        
        
    
    def categorical_kl_divergence(self, p_logits, q_logits):
        p_softmax = torch.softmax(p_logits, -1)
        kl = torch.sum(p_softmax * F.log_softmax(p_logits, dim = 1) - p_softmax * F.log_softmax(q_logits, dim = 1), -1)
        return kl
    
    def bernoulli_kl_divergence(self, p_prob, q_prob, eps=1e-4):
        # q_prob has shape [B, k ,d]
        # p_prob has shape [B, k, d]
        kl = p_prob * (torch.log(eps + p_prob) - torch.log(eps + q_prob)) + (1 - p_prob) * (torch.log(eps + 1 - p_prob) - torch.log(eps + 1 - q_prob))
        return torch.sum(kl, -1)
    
    def bernoulli_sample(self, p, T, eps=1e-4):
        noise = torch.rand_like(p)
        return torch.sigmoid((torch.log(eps + noise) - torch.log(eps + 1 - noise) + torch.log(eps + p) - torch.log(eps + 1 - p)) / T)

    def elbo(self, inputs, T): # inputs [B, d]
        z_logits = self.q_net(inputs) # [B, k]
        z_prob = torch.softmax(z_logits, -1) # [B, K]
        pred = torch.argmax(z_prob, -1)
        alpha_posterior_logits = self.r_net(torch.cat([inputs, z_logits], -1)) # [B, d]
        alpha_posterior_prob = torch.sigmoid(alpha_posterior_logits)
        f_mu = self.f_mu.unsqueeze(0).repeat(inputs.shape[0], 1, 1) # [B, k, d]
        b_mu = self.b_mu.unsqueeze(0).repeat(inputs.shape[0], 1, 1) # [B, 1, d]
        f_comp = D.Normal(f_mu, 1.0)
        b_comp = D.Normal(b_mu, 1.0)
        inputs_tiled = inputs.unsqueeze(1).repeat(1, self.k, 1) # [B, k, d]
        logp_f = f_comp.log_prob(inputs_tiled) #[B, k, d]
        logp_b = b_comp.log_prob(inputs.unsqueeze(1)) #[B, 1, d]
        E_log_p_x_given_alpha_z = torch.sum(logp_f * alpha_posterior_prob.unsqueeze(1), -1) + \
                                  torch.sum(logp_b * (1 - alpha_posterior_prob.unsqueeze(1)), -1) # [B, k]
        beta_prior_D = D.beta.Beta(torch.sigmoid(5*self.beta_prior_alpha.unsqueeze(0)), torch.sigmoid(5*self.beta_prior_beta.unsqueeze(0)))
        beta = beta_prior_D.rsample()
        kl_alpha_beta = self.bernoulli_kl_divergence(alpha_posterior_prob.unsqueeze(1), beta) # [B, k, d]
        kl_alpha_beta_z = z_prob * kl_alpha_beta
        E_z = torch.sum(z_prob * E_log_p_x_given_alpha_z - kl_alpha_beta_z, -1) # [B]
        kl_q_pi = self.categorical_kl_divergence(z_logits, self.prior_pi) # [b]
        beta_prior_D = D.beta.Beta(torch.sigmoid(5*self.beta_prior_alpha), torch.sigmoid(5*self.beta_prior_beta))
        PI_D = D.beta.Beta(0.5*torch.ones(self.k, self.d).to(inputs.device), 0.5*torch.ones(self.k, self.d).to(inputs.device))
        #D_beta_prior = D.Normal(torch.sigmoid(5 * self.beta_prior), 0.1*torch.ones(self.k,self.d).to(inputs.device))
        #D_PI = D.Normal(torch.zeros(self.k, self.d).to(inputs.device), 0.1*torch.ones(self.k,self.d).to(inputs.device))
        kl_Q_PI = D.kl.kl_divergence(beta_prior_D, PI_D).sum()
        return torch.mean(E_z - kl_q_pi) - kl_Q_PI, torch.mean(E_z), torch.mean(kl_alpha_beta_z), torch.mean(kl_q_pi), kl_Q_PI, pred, alpha_posterior_prob, z_prob
    
    def forward(self, inputs, T):
        return self.elbo(inputs.float(), T)

class Variational_GMM(nn.Module):
    def __init__(self, k, d, hidden_dims=128, num_layers=2):
        super().__init__()
        self.k = k
        self.d = d
        self.D = nn.Parameter(torch.ones(self.k, self.d), requires_grad=True)
        self.f_mu = nn.Parameter(torch.zeros(self.k, self.d), requires_grad=True)
        self.b_mu = nn.Parameter(torch.zeros(1, self.d), requires_grad=True)
        #self.prior_pi = nn.Parameter(0.1 * torch.randn(1, self.k), requires_grad=True)
        self.prior_pi = nn.Parameter(torch.ones(1, self.k) / self.k, requires_grad=True)
        self.q_net = [nn.Linear(d, hidden_dims), nn.ELU()]
        for i in range(num_layers):
            self.q_net += [nn.Linear(hidden_dims, hidden_dims), nn.ELU()]
        self.q_net += [nn.Linear(hidden_dims, k)]
        self.q_net = nn.Sequential(*self.q_net)
        self.r_net = [nn.Linear(d+k, hidden_dims), nn.ELU()]
        for i in range(num_layers):
            self.r_net += [nn.Linear(hidden_dims, hidden_dims), nn.ELU()]
        self.r_net += [nn.Linear(hidden_dims, d)]
        self.r_net = nn.Sequential(*self.r_net)
        self.beta_prior = nn.Parameter(0.01 * torch.randn(k, d), requires_grad=True)
    
    def categorical_kl_divergence(self, p_logits, q_logits):
        p_softmax = torch.softmax(p_logits, -1)
        kl = torch.sum(p_softmax * F.log_softmax(p_logits, dim = 1) - p_softmax * F.log_softmax(q_logits, dim = 1), -1)
        return kl
    
    def bernoulli_kl_divergence(self, p_prob, q_prob, eps=1e-4):
        # q_prob has shape [B, k ,d]
        # p_prob has shape [B, k, d]
        kl = p_prob * (torch.log(eps + p_prob) - torch.log(eps + q_prob)) + (1 - p_prob) * (torch.log(eps + 1 - p_prob) - torch.log(eps + 1 - q_prob))
        return torch.sum(kl, -1)
    
    def bernoulli_sample(self, p, T, eps=1e-4):
        noise = torch.rand_like(p)
        return torch.sigmoid((torch.log(eps + noise) - torch.log(eps + 1 - noise) + torch.log(eps + p) - torch.log(eps + 1 - p)) / T)

    def elbo(self, inputs, T): # inputs [B, d]
        z_logits = self.q_net(inputs) # [B, k]
        z_prob = torch.softmax(z_logits, -1) # [B, K]
        pred = torch.argmax(z_prob, -1)
        alpha_posterior_logits = self.r_net(torch.cat([inputs, z_logits], -1)) # [B, d]
        #beta_prior_logits = z_prob.matmul(self.beta_prior) # [B, d]
        alpha_posterior_prob = torch.sigmoid(alpha_posterior_logits) # [B, d]
        #alpha_posterior_prob_sample = self.bernoulli_sample(alpha_posterior_prob, T)
        alpha_posterior_prob_sample = alpha_posterior_prob
        f_mu = self.f_mu.unsqueeze(0).repeat(inputs.shape[0], 1, 1) # [B, k, d]
        b_mu = self.b_mu.unsqueeze(0).repeat(inputs.shape[0], 1, 1) # [B, 1, d]
        f_comp = D.Normal(f_mu, 1.0)
        b_comp = D.Normal(b_mu, 1.0)
        inputs_tiled = inputs.unsqueeze(1).repeat(1, self.k, 1) # [B, k, d]
        logp_f = f_comp.log_prob(inputs_tiled) #[B, k, d]
        logp_b = b_comp.log_prob(inputs.unsqueeze(1)) #[B, 1, d]
        E_log_p_x_given_alpha_z = torch.sum(logp_f * alpha_posterior_prob_sample.unsqueeze(1), -1) + \
                                  torch.sum(logp_b * (1 - alpha_posterior_prob_sample.unsqueeze(1)), -1) # [B, k]
        p_beta = torch.sigmoid(5 * self.beta_prior.unsqueeze(0))
        kl_alpha_beta = self.bernoulli_kl_divergence(alpha_posterior_prob.unsqueeze(1), p_beta) # [B, k]
        kl_alpha_beta_z = z_prob * kl_alpha_beta
        E_z = torch.sum(z_prob * E_log_p_x_given_alpha_z - kl_alpha_beta_z, -1) # [B]
        kl_q_pi = self.categorical_kl_divergence(z_logits, self.prior_pi) # [b]
        D_beta_prior = D.Normal(torch.sigmoid(5 * self.beta_prior), 0.1*torch.ones(self.k,self.d).to(inputs.device))
        #D_PI = D.Normal(torch.zeros(self.k, self.d).to(inputs.device), 0.1*torch.ones(self.k,self.d).to(inputs.device))
        D_PI = D.Uniform(0, 1)
        kl_Q_PI = torch.mean(D.kl.kl_divergence(D_beta_prior, D_PI))
        kl_Q_PI = 0
        return torch.mean(E_z - kl_q_pi) - kl_Q_PI, torch.mean(E_z), torch.mean(kl_alpha_beta_z), torch.mean(kl_q_pi), kl_Q_PI, pred, alpha_posterior_prob, alpha_posterior_prob_sample, z_prob
    
    def forward(self, inputs, T):
        return self.elbo(inputs.float(), T)


class GMM(nn.Module):
    ''' A GMM model. '''
    def __init__(
            self, k, d, tau, mu=None):
        super().__init__()
        self.k = k
        self.d = d
        self.A = nn.Parameter(0.01 * torch.randn(self.k, self.d, self.d), requires_grad=True)
        self.D = nn.Parameter(torch.ones(self.k, self.d), requires_grad=True)
        if mu is None:
            self.mu = nn.Parameter(torch.zeros(self.k, self.d), requires_grad=True)
        else:
            self.mu = nn.Parameter(torch.Tensor(mu), requires_grad=True)
        self.pi = nn.Parameter(0.01 * torch.randn(self.k), requires_grad=True)
        self.tau = tau
    def negative_log_prob(self, x):
        sigma = torch.diag_embed(torch.nn.functional.softplus(self.D)) #+ torch.matmul(self.A, self.A.transpose(1,2))
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
        #d_used = min(d, max(500, d//8))
        #d_indices = np.random.choice(d, d_used, replace=False)
        print('Performing k-means clustering to {} components of {} samples  ...'.format(K, N))
        with Timer('K-means'):
            clusters = KMeans(n_clusters=K, max_iter=300, n_jobs=-1).fit(samples)
        labels = clusters.labels_
        if get_centers:
            centers = np.zeros([K, d])
            for i in range(K):
                centers[i, :] = np.mean(samples[labels == i, :], axis=0)
            #self.mu = nn.Parameter(torch.Tensor(centers).to(self.mu.device), requires_grad=True)
            #pdb.set_trace()
            return labels, centers
        return labels
    def check_assignment(self, x):
        sigma = torch.diag_embed(torch.nn.functional.softplus(self.D)) #+ torch.matmul(self.A, self.A.transpose(1,2))
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
        self.f_D = nn.Parameter(torch.ones(self.k, self.d), requires_grad=True)
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
        f_sigma = torch.diag_embed(torch.nn.functional.softplus(self.f_D)) #+ torch.matmul(self.f_A, self.f_A.transpose(1,2))
        f_sigma = f_sigma.unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # [bs, k, d, d]
        f_mu = self.f_mu.unsqueeze(0).repeat(x.shape[0], 1, 1) # [bs, k, d]
        b_sigma = torch.diag_embed(torch.nn.functional.softplus(self.b_D)) #+ torch.matmul(self.b_A, self.b_A.transpose(1,2))
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

if __name__ == "__main__":
    model = GMM(10, 64)
    data = torch.zeros(32, 64)
    print(model.negative_log_prob(data))

