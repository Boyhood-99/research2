import numpy as np
import models, torch, copy
from tqdm import tqdm
import torch.utils.data as DATA

class Client(object):

	def __init__(self, conf, model, train_dataset, id = -1, compile = False):
		
		self.conf = conf
		
		self.local_model = models.get_model(self.conf["model_name"]) 
		if compile:
			self.local_model = torch.compile(self.local_model)
		
		self.client_id = id
		
		self.train_dataset = train_dataset
		
		#客户端平分数据集
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / self.conf['num_f_uav'])

		train_indices = all_range[id * data_len: (id + 1) * data_len]

		#完全平分
		# self.train_loader = DATA.DataLoader(self.train_dataset, batch_size=conf["batch_size"], num_workers=2, 
		# 					drop_last =True, pin_memory=True, sampler=DATA.sampler.SubsetRandomSampler(train_indices),
		#  					shuffle=True,
		# )
		##自定义样本数量
		num_sample = np.random.randint(80,100)
		self.train_loader = DATA.DataLoader(self.train_dataset, batch_size = conf["batch_size"],  num_workers=2, 
							drop_last =True, pin_memory=True,sampler = DATA.sampler.SubsetRandomSampler(
							list(np.random.choice(train_indices, num_sample)))
							# shuffle=True,
							)
									
		
	def local_train(self, global_model, global_epoch):

		for name, param in global_model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	
		#print(id(model))
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		#print(id(self.local_model))
		
		self.local_model.train()
		loss_dic = {}
		for local_epoch in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				optimizer.zero_grad()
				#前向传播
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				#反向传播
				loss.backward()
				optimizer.step()
				
			# print("Epoch %d done." % e)	
			# print(f"L_UAV_{self.client_id} complete the {local_epoch+1}-th local iteration ")
			
			loss_dic[f'local epoch{local_epoch} loss'] = loss.item()
		print('\n')
		print(f'local train loss for L_UAV_{self.client_id} in the {global_epoch }-th global epoch:{loss}')

		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - global_model.state_dict()[name])
			
			# diff[name] = data.sub_(global_model.state_dict()[name])
			
			
			
		return diff , loss_dic
		