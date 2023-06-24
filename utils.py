
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 
 
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