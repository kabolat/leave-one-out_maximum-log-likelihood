import torch

def to_sigma(sigmatilde):
    return torch.log2(torch.pow(2,sigmatilde)+1)

def from_sigma(sigma):
    return torch.log2(torch.pow(2,sigma)-1)

def create_logger_dict():
    logs = {'log_likelihood':[],
            'log_likelihood_loo':[], 
            'log_likelihood_val':[], 
            'log10_weights':[], 
            'num_kernels':[],
            'num_params':[],
            'log10_sigma': [], 
            'log10_filter_threshold':[],
            'itx':[]}
    return logs
