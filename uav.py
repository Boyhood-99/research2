from torchvision import models
import torch.utils.data as DATA
import torch
# from torchinfo import summary
from torchsummary import summary
from torchstat import stat
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import mmdet.models.losses.focal_loss as focal_loss

class Server(object):
	
	def __init__(self, conf, eval_dataset, compile):
	
		self.conf = conf 
		#-------------------------------------------------------------
		# self.global_model = models_1.get_model(self.conf["model_name"]) 
		# input = torch.tensor
		# summary(self.global_model, input_size=(3, 32, 32))
		#-------------------------------------------
		self.global_model = models.resnet18(
			# weights = None,
			weights =models.ResNet18_Weights.DEFAULT,
			)
		inchannel = self.global_model.fc.in_features
		self.global_model.fc = nn.Linear(inchannel, 10)
		if torch.cuda.is_available():
			self.global_model.cuda()
		#----------------------------------------
		if compile:
			self.global_model = torch.compile(self.global_model) 
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
		
	#模型聚合，不需保留梯度
	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)
				
	def model_eval(self):
		
		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch 
			dataset_size += data.size()[0]
			
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
				
			
			output = self.global_model(data)
			
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			# print(output,target)
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))
		total_l = total_loss / dataset_size

		return acc, total_l
	

class Client(object):

	def __init__(self, conf, train_dataset, dataset_indice, id = -1, compile = False):

		# self.location = torch.zeros(size=(3))
		
		self.conf = conf
		
		###
		# self.local_model = models_1.get_model(self.conf["model_name"]) 
		# summary(self.local_model, input_size=(3, 32, 32))
		#-----------------------------------------------------
		self.local_model = models.resnet18(
						# weights =None,
				     weights= models.ResNet18_Weights.DEFAULT,
					 	)
		
		# for param in self.local_model.parameters():
		# 	param.requires_grad = False

		inchannel = self.local_model.fc.in_features
		self.local_model.fc = nn.Linear(inchannel, 10)
		if torch.cuda.is_available():
			self.local_model.cuda()
		#------------------------------------------

		if compile:
			self.local_model = torch.compile(self.local_model)

		self.client_id = id
		
		self.train_dataset = train_dataset

		####------------------------------------
		#自定义样本数量
		# self.num_sample = np.random.randint(800,1000)
		# self.ls = list(np.random.choice(dataset_indice, self.num_sample, replace=False))
		
		# self.train_loader = DATA.DataLoader(self.train_dataset, batch_size = conf["batch_size"],  
		# 		      		num_workers=2, drop_last =True, pin_memory=True,
		# 					sampler = DATA.sampler.SubsetRandomSampler(self.ls),
		# 					# shuffle=True,
		# 					)
            
        ###----------------------------------
		
		self.train_loader = DATA.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
				      		num_workers=2, drop_last =True, pin_memory=True, 
							sampler=DATA.sampler.SubsetRandomSampler(dataset_indice),
							)
				      		
		###----------------------------------
		self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'],
									weight_decay = 1e-3,#1e-4,1e-3
									)
		
		# self.optimizer = torch.optim.SGD([
		# 	{'params': self.local_model.fc.parameters()},
        # 	{'params': self.local_model.layer4.parameters()},
		# 	{'params': self.local_model.layer3.parameters()},
		# 	{'params': self.local_model.layer2.parameters()},
		# 	{'params': self.local_model.layer1.parameters()},
		# 	{'params': self.local_model.conv1.parameters()},
		# 	], 
		# 	lr=self.conf['lr'],
		# 	momentum=self.conf['momentum'],
		# 	# weight_decay = 1e-4,
		# 	)
		#####--------------------------
		# self.criterion =  torch.nn.functional.cross_entropy(label_smoothing=0.001)
		self.criterion = torch.nn.CrossEntropyLoss(
			# label_smoothing=0.2,
					     )
		
		self.criterion = FocalLoss()

		# self.criterion = focal_loss()

		print(f'client {self.client_id}   dataset_size:{len(dataset_indice)}')
		# print(f'dataset_indice:{dataset_indice}')
		dataset_indice_dic = {}
		for i in dataset_indice:
			key = self.train_dataset.targets[i]
			dataset_indice_dic[key] = dataset_indice_dic.get(key, 0) + 1	
		f = sorted(zip(dataset_indice_dic.keys(), dataset_indice_dic.values()), key=lambda x: x[0], reverse=True)
		print(f'{f}')

	def get_indice(self,):
		print(self.num_sample)
		print(self.ls)
		return self.num_sample, self.ls
	def local_train(self, global_model, global_epoch, local_epochs, name = None):

		for pm_name, param in global_model.state_dict().items():
			self.local_model.state_dict()[pm_name].copy_(param.clone())
		
		self.local_model.train()
		loss_dic = {}
		# for local_epoch in range(self.conf["local_epochs"]):
		for local_epoch in range(local_epochs):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				self.optimizer.zero_grad()
				#前向传播
				output = self.local_model(data)
				# loss = torch.nn.functional.cross_entropy(output, target)
				loss = self.criterion(output, target)
				if name == 'FedProx':
					proximal_term = 0.0
					for w, w_t in zip(self.local_model.parameters(), global_model.parameters()):
						proximal_term += (w - w_t).norm(2)
					loss = loss + (self.conf['mu'] / 2) * proximal_term

				#反向传播
				loss.backward()
				self.optimizer.step()
				
				
			# print(f"L_UAV_{self.client_id} complete the {local_epoch+1}-th local iteration ")
			
			loss_dic[f'local epoch{local_epoch} loss'] = loss.item()
		
		# print(f'local train loss for L_UAV_{self.client_id} in the {global_epoch }-th global epoch:{loss}')

		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - global_model.state_dict()[name])
			# diff[name] = data.sub_(global_model.state_dict()[name])
			
		return diff , loss_dic
	

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
