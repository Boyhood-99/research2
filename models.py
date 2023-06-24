import torch 
from torchvision import models
#The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. 
# You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.

def get_model(name="resnet50",):
	if name == "resnet18":
		model = models.resnet18(weights =None)
	elif name == "resnet50":
		model = models.resnet50(weights =None)	
		model = models.resnet50(weights =models.ResNet50_Weights.DEFAULT)
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
		model = models.googlenet()
		
	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 