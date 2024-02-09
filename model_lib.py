import torch
from numpy import log, pi
from utils_lib import * 
import time
from gmm_lib import GaussianMixtureTorch as GMM


class KernelDensityEstimator(torch.nn.Module):
    def __init__(self, mu, sigma=1e-1, learn_weights=False):
        
        super(KernelDensityEstimator, self).__init__()

        self.N, self.dim = mu.shape
        self.mu = mu.unsqueeze(0)
        self.learn_weights = learn_weights

        self.sigmatilde = from_sigma(torch.ones(1,self.N,1)*sigma)

        self.vicinity_matrix = torch.ones(self.N,self.N)

        self.wtilde = torch.ones(1,self.N)
        self.weights = torch.ones(1,self.N)/self.N

        self.loo_mask = 1.0-torch.eye(self.N)

    def autograd(self):
        self.sigmatilde = torch.nn.Parameter(self.sigmatilde)
        if self.learn_weights: self.wtilde = torch.nn.Parameter(self.wtilde)

    def forward(self, x, leave_one_out=False, batch_idx=None):

        mu, sigma, weights = self.mu, to_sigma(self.sigmatilde), self.weights

        kernel_matrix_exponent = self.gaussian_kernel_exponent(x, mu, sigma) + torch.log(weights)

        if leave_one_out:
            if batch_idx is not None: 
                mask = self.loo_mask[batch_idx]
            else:
                mask = self.loo_mask
            kernel_matrix_exponent_masked = kernel_matrix_exponent*mask
            kernel_matrix_exponent_masked[mask==0] = -1e32
            log_likelihood = torch.logsumexp(kernel_matrix_exponent_masked,1)
        else:
            log_likelihood = torch.logsumexp(kernel_matrix_exponent,1)
            
        return log_likelihood
    
    def gaussian_kernel_exponent(self, x, mu, sigma):
        D = self.dim
        x_mu = x.unsqueeze(1)-mu
        mah_dist = torch.norm(x_mu,dim=-1)/sigma.squeeze()
        log_det = 2*D*torch.log(sigma).squeeze()
        return -0.5*(mah_dist**2 + log_det + D*log(2*pi))
    
    def sampling(self, num_samples=1, respect_zeros=False, respect_idx=None, data_mean=None, data_std=None):
        mu, sigma, weights = self.mu, to_sigma(self.sigmatilde), self.weights
        kernel_idx = torch.multinomial(weights.squeeze(), num_samples, replacement=True)
        mu_selected, sigma_selected = mu[0,kernel_idx,:], sigma[0,kernel_idx]
        samples = sigma_selected * torch.randn(num_samples, self.dim) + mu_selected
        if respect_zeros:
            if respect_idx is None: raise ValueError('Respect idx is required!')
            samples_unnorm = samples*data_std + data_mean
            respected_columns = samples_unnorm[:,respect_idx]
            respected_columns[respected_columns<0.0]=0.0
            samples_unnorm[:,respect_idx]=respected_columns
            samples = (samples_unnorm-data_mean)/data_std
        return samples
    
    def expectation_step(self, x, leave_one_out=True):
        mu, sigma, weights = self.mu, to_sigma(self.sigmatilde), self.weights
        kernel_matrix = torch.exp(self.gaussian_kernel_exponent(x, mu, sigma) + torch.log(weights)) + 1e-32
        if leave_one_out: kernel_matrix = kernel_matrix*self.loo_mask
        self.responsibility_matrix = kernel_matrix/kernel_matrix.sum(1,keepdims=True)
    
    def maximization_step(self, x):
        x_mu = x.unsqueeze(1)-self.mu
        responsibility_matrix = self.responsibility_matrix
        sigma = torch.sqrt(torch.sum(responsibility_matrix*(x_mu**2).mean(-1),0)/(responsibility_matrix.sum(0))).unsqueeze(-1)
        self.sigmatilde = from_sigma(sigma+1e-6).unsqueeze(0)
        if self.learn_weights: self.weights = responsibility_matrix.sum(0,keepdim=True)/responsibility_matrix.sum()

    def gradient_step(self, dataloader, optimizer, num_epochs=1):
        for _ in range(num_epochs):
            for x, idx in dataloader:
                optimizer.zero_grad()
                self.weights = self.wtilde.softmax(dim=1)
                log_likelihood_loo = self(x, leave_one_out=True, batch_idx=idx)
                loss = -log_likelihood_loo.mean()
                loss.backward()
                optimizer.step()

    def train(self, 
                x, 
                x_val=None,
                modified_em = False,
                dataloader=None,
                optimizer=None,
                prune=False,
                prune_threshold_factor=1e-2,
                adaptive_threshold=True,
                wait_convergence=True, 
                num_iterations=50, 
                loss_threshold=1e-4,
                verbose=False):
        #########################################
        if prune and not self.learn_weights: raise ValueError('Pruning requires learning weights!')
        if prune and not modified_em: raise ValueError('Pruning is supported only with modified EM!')
        
        train_flag, itx, logs, last_loss = True, 0, create_logger_dict(), 0.0
        #########################################
        if prune:
            prune_acc = 0
            prune_threshold = torch.tensor(1.0)/self.N*prune_threshold_factor
        #########################################
        start_time = time.time()
        while train_flag:
            if modified_em:
                self.expectation_step(x, leave_one_out=True)
                self.maximization_step(x)
                log_likelihood_loo = self(x, leave_one_out=True)
            else:
                self.gradient_step(dataloader, optimizer, num_epochs=1)
                log_likelihood_loo = self(dataloader.dataset.data, leave_one_out=True)
            loss = -log_likelihood_loo.mean()
            if loss.isnan(): raise ValueError('Loss is NaN!')
            itx+=1
            #########################################
            #########################################
            if prune: 
                if adaptive_threshold: prune_threshold = torch.tensor(1.0)/self.N*prune_threshold_factor
                filter_idx = (self.weights>prune_threshold).squeeze().nonzero().squeeze()
                prune_idx = (self.weights<=prune_threshold).squeeze().nonzero().squeeze()
                if prune_idx.numel()>0:
                    self.mu, self.sigmatilde = self.mu[:,filter_idx], self.sigmatilde[:,filter_idx]
                    self.weights = self.weights[:,filter_idx]/self.weights[:,filter_idx].sum()
                    self.wtilde = self.wtilde[:,filter_idx]
                    self.loo_mask = self.loo_mask[:,filter_idx]
                    self.N = self.weights.shape[1]
                    prune_acc += prune_idx.numel()
            #########################################
            if verbose: 
                print('Iteration: %d, Log-likelihood: %.5f'%(itx, -loss))
                if prune: print(f"Number of discarded kernels: {prune_acc}")
            #########################################
            if wait_convergence:
                delta_loss = torch.abs(loss-last_loss)
                last_loss = loss
                if delta_loss<loss_threshold: train_flag=False
            else:
                if itx>=num_iterations: train_flag=False
            #########################################
        end_time = time.time()
        logs['log_likelihood_loo'] += [-loss.detach().numpy()]
        logs['log_likelihood'] += [self(x, leave_one_out=False).mean().detach().numpy()]
        logs['log_likelihood_val'] += [self(x_val, leave_one_out=False).mean().detach().numpy()]
        logs['log10_weights'] += [self.weights.squeeze().log10().detach().numpy()]
        logs['log10_sigma'] += [to_sigma(self.sigmatilde).squeeze().log10().detach().numpy()]
        logs['num_kernels'] += [self.weights.squeeze().numel()]
        logs['num_params'] += [self.weights.squeeze().numel()*(1+self.learn_weights)]
        logs['itx'] += [itx]
        logs['time'] = end_time-start_time
        if prune: logs['log10_filter_threshold'] += [torch.log10(prune_threshold)]
        return logs

class GaussianMixtureModelBenchmark(GMM):
    def __init__(self, num_kernels, n_features):
        super(GaussianMixtureModelBenchmark, self).__init__(n_components=num_kernels, n_features=n_features, verbose=False)
        self.num_kernels = num_kernels

    def train(self, x, x_val=None):
        logs = create_logger_dict()

        start_time = time.time()
        self.fit(x)
        end_time = time.time()

        num_dims = x.shape[1]
        logs['log_likelihood_val'] = self.score(x_val).mean()
        logs['num_kernels'] = self.num_kernels
        logs['num_params'] = ((num_dims + num_dims * (num_dims + 1) / 2 + 1) * self.num_kernels).astype(int)
        logs['time'] = end_time-start_time

        return logs
    
    def sampling(self, num_samples, respect_zeros=False, respect_idx=None, data_mean=None, data_std=None):
        samples = self.sample(num_samples)[0]
        if respect_zeros:
            if respect_idx is None: raise ValueError('Respect idx is required!')
            samples_unnorm = samples*data_std + data_mean
            respected_columns = samples_unnorm[:,respect_idx]
            respected_columns[respected_columns<0.0]=0.0
            samples_unnorm[:,respect_idx]=respected_columns
            samples = (samples_unnorm-data_mean)/data_std
        return samples
    
    def score(self, x):
        return self.score_samples(x)