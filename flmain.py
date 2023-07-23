import argparse, json
import datetime
import torch
import pandas as pd
from uav import *
from datasets import Dataset
from tqdm import tqdm
import time
from utils import TrainThread
from configuration import ConfigDraw, ConfigTrain
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from visualization import flvisual
writer = SummaryWriter('./tensorboard/log/')

#global Epoch: 4, acc: 72.61999999999999, loss: 0.8394354427337647

def main(conf):
	# torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
	# torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
	torch.set_float32_matmul_precision('high')
	
	date = datetime.datetime.now().strftime('%m-%d')

	# parser = argparse.ArgumentParser(description='Federated Learning')
	# parser.add_argument('-c', '--config', dest='conf')
	# args = parser.parse_args()
	
	dataset = Dataset(conf)
	eval_datasets = dataset.eval_dataset
	server = Server(conf, eval_datasets, compile = conf['compile'])
	num_clients = conf['f_uav_num']
	clients = []
	for i in range(num_clients):  
		clients.append(Client(conf,  dataset.train_dataset, dataset.dataset_indice_list[i], id=i, compile = conf['compile']))
	
	df_list = []

	acc, loss = server.model_eval()
	
	print(f'global Epoch: 0, acc: {acc}, loss: {loss}')
	for global_epoch in tqdm(range(conf["global_epochs"])):
		#-----------------------------------------------------
		# candidates = random.sample(clients, conf["k"])
		# candidates = clients
		candidates = []
		for i in conf['candidates']:
			candidates.append(clients[i])		
		weight_accumulator = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)

		global_epoch_dic = {}#for print log
		global_epoch_dic['global Epoch'] = global_epoch
		#---------------------------------------------

		#---------------------------------------单线程，主要是local_train耗时
		# for candidate in candidates:
		# 	diff, loss_dic = candidate.local_train(server.global_model, global_epoch)
			
		# 	for name, params in server.global_model.state_dict().items():
		# 		weight_accumulator[name].add_(diff[name])
		# 	global_epoch_dic[f'f_uav{candidate.client_id}'] = loss_dic
		# 	# print(f"L_UAV_{self.client_id} complete the {local_epoch+1}-th local iteration ")
		#---------------------------------------单线程
		#---------------------------------------多线程
		num_candidate = len(candidates)
		threads = []
		for i in range(num_candidate):
			thread = TrainThread(candidates[i].local_train(server.global_model, global_epoch, conf['local_epochs']))
			# thread.setDaemon(True)
			threads.append(thread)
		# for thread in threads:
			threads[-1].start()

		for thread in threads:
			thread.join()
		
			# diff, loss_dic = thread.getresult()
			# for name, params in server.global_model.state_dict().items():
			# 	weight_accumulator[name].add_(diff[name])
			# 	global_epoch_dic[f'f_uav{candidates[i].client_id}'] = loss_dic
		
		for i in range(num_candidate):
			diff, loss_dic = threads[i].getresult()
			
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
				global_epoch_dic[f'f_uav{candidates[i].client_id}'] = loss_dic
			# print(f"L_UAV_{self.client_id} complete the {local_epoch+1}-th local iteration ")
		#--------------------------------------------多线程
		
		server.model_aggregate(weight_accumulator)

		#-------------------------------------
		if global_epoch < 10 or global_epoch == conf["global_epochs"] - 1:
			acc, loss = server.model_eval()#耗时
			print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
			
		elif global_epoch >10 and global_epoch%10 == 0:
			acc, loss = server.model_eval()#耗时
			print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
		#---------------------------------------
		# acc, loss = server.model_eval()#耗时

		

		# print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
			

		global_epoch_dic['global_accuracy'] = acc
		global_epoch_dic['global_loss'] = loss
		df_list.append(global_epoch_dic)

	df = pd.DataFrame(df_list)
	df.to_csv(f'log{date}.csv')
	date = datetime.datetime.now().strftime('%m-%d')
	flvisual(df, date, path = f'./FL_main/fig_7_16.png')
	return df

if __name__ == '__main__':
	# date = datetime.datetime.now().strftime('%H:%M:%S')
	# print(date)
    print(torch.cuda.is_available())
    print(torch.__version__)
    # torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # torch.set_float32_matmul_precision('high')
    with open('./conf.json' , 'r') as f:
        conf = json.load(f)
    config_train = ConfigTrain
    config_draw = ConfigDraw
    conf['config_train'] = config_train
    conf["config_draw"] = config_draw 
   
    main(conf=conf)
        
			
		
		
	
		
		
	