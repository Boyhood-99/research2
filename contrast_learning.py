import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 加载预训练的ResNet作为特征提取器
resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层

# 对比学习任务
class ContrastiveLearning(nn.Module):
    def __init__(self, base_encoder):
        super(ContrastiveLearning, self).__init__()
        self.base_encoder = base_encoder

    def forward(self, x1, x2):
        z1 = self.base_encoder(x1)
        z2 = self.base_encoder(x2)
        return z1, z2

# 训练对比学习
model = ContrastiveLearning(resnet)
criterion = nn.CosineSimilarity(dim=1)  # 余弦相似度作为相似性度量
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10
batch_size = 32

data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(data_loader):
        data1, data2 = data[0], data[1]  # 将每个batch的数据分为两个部分，形成样本对
        data1, data2 = data1.cuda(), data2.cuda()

        optimizer.zero_grad()
        z1, z2 = model(data1, data2)
        loss = 1 - criterion(z1, z2).mean()  # 最大化相似性
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# 使用训练好的模型进行特征提取或其他任务
