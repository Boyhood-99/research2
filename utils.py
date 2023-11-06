from threading import Thread
from typing import Any
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss


class TrainThread(Thread):
    def __init__(self, func,
                #  args,
                 ):
        Thread.__init__(self)
        self.func = func
        # self.args = args
        self.result = None
    def run(self, ):
        self.diff, self.loss_dic, self.model, self.client_id = self.func
    def getresult(self):
        return self.diff, self.loss_dic,self.model, self.client_id
    
class TrainThread_FedDyn(Thread):
    def __init__(self, func,
                #  args,
                 ):
        Thread.__init__(self)
        self.func = func
        # self.args = args
        self.result = None
    def run(self, ):
        self.loss, self.prev_grads = self.func

    def getresult(self):
        return self.loss, self.prev_grads


class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction = 'none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_memory_allocated(device, inplace = False):
    '''
    Function measures allocated memory before and after the ReLU function call.
    INPUT:
      - device: gpu device to run the operation
      - inplace: True - to run ReLU in-place, False - for normal ReLU call
    '''
 
 
    # Create a large tensor
    t = torch.randn(10000, 10000, device=device)
 
 
    # Measure allocated memory
    torch.cuda.synchronize()
    start_max_memory = torch.cuda.max_memory_allocated() / 1024**2
    start_memory = torch.cuda.memory_allocated() / 1024**2
 
 
    # Call in-place or normal ReLU
    if inplace:
        F.relu_(t)
    else:
        output = F.relu(t)
 
 
    # Measure allocated memory after the call
    torch.cuda.synchronize()
    end_max_memory = torch.cuda.max_memory_allocated() / 1024**2
    end_memory = torch.cuda.memory_allocated() / 1024**2
 
 
    # Return amount of memory allocated for ReLU call
    return end_memory - start_memory, end_max_memory - start_max_memory




if __name__ == 'main':
    #内存分配实验
    # setup the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    
    # call the function to measure allocated memory out of place
    memory_allocated, max_memory_allocated = get_memory_allocated(device, inplace = False)
    print('Allocated memory: {}'.format(memory_allocated))
    print('Allocated max memory: {}'.format(max_memory_allocated))

    #in place
    memory_allocated_inplace, max_memory_allocated_inplace = get_memory_allocated(device, inplace = True)
    print('Allocated memory: {}'.format(memory_allocated_inplace))
    print('Allocated max memory: {}'.format(max_memory_allocated_inplace))