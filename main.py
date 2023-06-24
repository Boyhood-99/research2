import argparse, json
import datetime
import os
import logging
import torch
import random
import pandas as pd
from server import *
from client import *
import models, datasets
from tqdm import tqdm

#python main.py -c ./utils/conf.json
#global Epoch: 4, acc: 72.61999999999999, loss: 0.8394354427337647

def main():

	# parser = argparse.ArgumentParser(description='Federated Learning')
	# parser.add_argument('-c', '--config', dest='conf')
	# args = parser.parse_args()

	# with open(args.conf, 'r') as f:
	# 	conf = json.load(f)	
	
	with open('./utils/conf.json', 'r') as f:
		conf = json.load(f)	
	
	

	
	# print(type(conf))
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	
	server = Server(conf, eval_datasets, compile=conf['compile'])
	clients = []
	
	for uav_id in range(conf['num_f_uav']):
		clients.append(Client(conf, server.global_model, train_datasets, uav_id, conf['compile']))


	df_list = []
	for global_epoch in tqdm(range(conf["global_epochs"])):
	
		# candidates = random.sample(clients, conf["k"])
		# candidates = clients
		candidates = []
		for i in conf['candidates']:
			candidates.append(clients[i])

		
		weight_accumulator = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)

		global_epoch_dic = {}
		global_epoch_dic['global Epoch'] = global_epoch
		for candidate in candidates:
			
			diff, loss_dic = candidate.local_train(server.global_model, global_epoch)
			
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
			global_epoch_dic[f'f_uav{candidate.client_id}'] = loss_dic
			# print(f"L_UAV_{self.client_id} complete the {local_epoch+1}-th local iteration ")
		server.model_aggregate(weight_accumulator)
		
		acc, loss = server.model_eval()
		print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
		
		global_epoch_dic['global_accuracy'] = acc
		global_epoch_dic['global_loss'] = loss
		
		
		df_list.append(global_epoch_dic)
		


	df = pd.DataFrame(df_list)
	df.to_csv('log.csv')
	return df
if __name__ == '__main__':
	print(torch.cuda.is_available())
	print(torch.__version__)


	main()
				
			
		
		
	
		
		
	