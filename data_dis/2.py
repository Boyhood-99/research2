import torch

def sqrt(x, epsilon = 0.001):
    assert x>=0
    l, r = 0, x
    
    while l<=r:
        middle = l +(r-l)/2
        if middle*middle ==x:
            return middle
        elif middle*middle <x:
            l =middle
        else:
            r =middle
        if abs(middle*middle - x) < epsilon:
            return middle   

def solutin(num, n):
    i, j =0,0
    sum=0
    
    for j in range(len(num)):
        sum+=num[j]
        while sum>n:
            len = j-i+1
            sum-=num[i]
            i+=1
            if j-i+1<len:
                len = j-i+1
    return len


import torch

# 输入张量 x，假设其为一个 (3, 3) 的张量
x = torch.tensor([[1, 2, 3], [7, 8, 9], [5, 6, 7]], dtype=torch.float32)

# 计算每个特征的均值和标准差
mean = x.mean(dim=0)
std = x.std(dim=0)

# 假设缩放因子和偏移项为以下值
gamma = torch.tensor([1.2, 1.2, 1.2])
beta = torch.tensor([0.5, 0.5, 0.5])

# 计算归一化输出
x_normalized = (x - mean) / std
y = gamma * x_normalized + beta

print("Input x:\n", x)
print("Normalized x:\n", x_normalized)
print("Output y:\n", y)
