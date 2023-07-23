from torchvision import models
import torch.utils.data as DATA
import torch
# from torchinfo import summary
from torchsummary import summary
from torchstat import stat
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Server(object):
	
	def __init__(self, conf, eval_dataset, compile):
	
		self.conf = conf 
		#-------------------------------------------------------------
		# self.global_model = models_1.get_model(self.conf["model_name"]) 
		# input = torch.tensor
		# summary(self.global_model, input_size=(3, 32, 32))
		#-------------------------------------------
		self.global_model = models.resnet18(weights =models.ResNet18_Weights.DEFAULT)
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
		self.local_model = models.resnet18(weights =models.ResNet18_Weights.DEFAULT)
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
		# num_sample = np.random.randint(800,1000)
		# self.train_loader = DATA.DataLoader(self.train_dataset, batch_size = conf["batch_size"],  
		# 		      		num_workers=2, drop_last =True, pin_memory=True,
		# 					sampler = DATA.sampler.SubsetRandomSampler(
		# 					list(np.random.choice(dataset_indice, num_sample)))
		# 					# shuffle=True,
		# 					)
            
        ###----------------------------------
		
		self.train_loader = DATA.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
				      		num_workers=2, drop_last =True, pin_memory=True, 
							sampler=DATA.sampler.SubsetRandomSampler(dataset_indice),
							)
				      		

		self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'],
									# weight_decay = 1e-4,
									)
		# self.lossfun =  torch.nn.functional.cross_entropy()

	def local_train(self, global_model, global_epoch, local_epochs, name = None):

		for name, param in global_model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
		
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
				loss = torch.nn.functional.cross_entropy(output, target)
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
		