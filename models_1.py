import torch 
from torchvision import models
import torch.nn.functional as F
#The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. 
# You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.



def get_model(name="resnet50",):
	if name == "resnet18":
		model = models.resnet18(weights =models.ResNet18_Weights.DEFAULT)
	elif name == "resnet50":
		model = models.resnet50(weights =None)	
		# model = models.resnet50(weights =models.ResNet50_Weights.DEFAULT)
	elif name == "densenet121":
		model = models.densenet121()		
	elif name == "alexnet":
		model = models.alexnet()
	elif name == "vgg16":
		model = models.vgg16()
	elif name == "vgg19":
		model = models.vgg19()
	elif name == "inception_v3":
		model = models.inception_v3()
	elif name == "googlenet":		
		model = models.googlenet()
	elif name == "cnn":		
		# model = models.googlenet()
		model = CNN_NET()
		

	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 



class CNN_NET(torch.nn.Module):
    def __init__(self, ):
        super(CNN_NET,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3,
                                     out_channels = 6,
                                     kernel_size = 5,
                                     stride = 1,
                                     padding = 0)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2,
                                       stride = 2)
        self.conv2 = torch.nn.Conv2d(6,16,5)
		# self.pool = torch.nn.MaxPool2d(kernel_size = 2,
        #                                stride = 2)
        self.fc1 = torch.nn.Linear(16*5*5,120)
        # self.fc2 = torch.nn.Linear(120,84)
        self.fc3 = torch.nn.Linear(120,10)

    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5) #卷积结束后将多层图片平铺batchsize行16*5*5列，每行为一个sample，16*5*5个特征
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# ======================resnet18
# Total params: 23,379,024
# Trainable params: 23,379,024
# Non-trainable params: 0
# Params size (MB): 93.52
# ======================resnet18

# =========================res50
# Total params: 51,114,064
# Trainable params: 51,114,064
# Non-trainable params: 0
# Params size (MB): 204.46
# =========================res50