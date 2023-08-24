import copy, json
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
from visualization import flvisual, data_dis_visual
import random
writer = SummaryWriter('./tensorboard/log/')

#global Epoch: 4, acc: 72.61999999999999, loss: 0.8394354427337647

def main(conf, dir_alpha = 0.3):
	# torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
	# torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
	torch.set_float32_matmul_precision('high')
	
	date = datetime.datetime.now().strftime('%m-%d')
	torch.manual_seed(2023)
	np.random.seed(2023)
	random.seed(2023)
	

	# parser = argparse.ArgumentParser(description='Federated Learning')
	# parser.add_argument('-c', '--config', dest='conf')
	# args = parser.parse_args()
	
	dataset = Dataset(conf, dir_alpha=dir_alpha)
	eval_dataset = dataset.eval_dataset
	server = Server(conf, eval_dataset, compile = conf['compile'])
	
	num_clients = conf['f_uav_num']
	clients = []
	np.random.seed(2023)
	for i in range(num_clients):  
		_ = Client(conf,  dataset.train_dataset, dataset.dataset_indice_list[i], id=i, compile = conf['compile'])
		clients.append(_)
		# if i == 1 or i == 2:
		# 	_, _ = _.get_indice()
			# print(dataset.dataset_indice_list[i])
	
	df_list = []

	acc, loss = server.model_eval()

	# global_epoch_dic['global Epoch'] = global_epoch
	global_epoch_dic = {}
	global_epoch_dic['global_accuracy'] = acc
	global_epoch_dic['global_loss'] = loss
	df_list.append(global_epoch_dic)
	
	print(f'global Epoch: 0, acc: {acc}, loss: {loss}')
	for global_epoch in tqdm(range(conf["global_epochs"])):
		#-----------------------------------------------------
		# candidates = random.sample(clients, conf["k"])
		# candidates = clients
		candidates = []
		for i in conf['candidates']:
			candidates.append(clients[i])	

		local_epochs = random.randint(2, 11)	
		
		# local_epochs = conf['local_epochs']
		print(local_epochs)
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
		for candidate in candidates:
			assert isinstance(candidate, Client)
			thread = TrainThread(candidate.local_train(server.global_model, global_epoch, 
                                                           local_epochs, 
														#   name='FedProx',
															) )
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
		# if global_epoch < 10 or global_epoch == conf["global_epochs"] - 1:
		# 	acc, loss = server.model_eval()#耗时
		# 	print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
			
		# elif global_epoch >10 and global_epoch%10 == 0:
		# 	acc, loss = server.model_eval()#耗时
		# 	print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
		#---------------------------------------
		acc, loss = server.model_eval()#耗时
		print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
		

		# print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
			

		global_epoch_dic['global_accuracy'] = acc
		global_epoch_dic['global_loss'] = loss
		df_list.append(global_epoch_dic)

	df = pd.DataFrame(df_list)
	# df.to_csv(f'./log/log{date}.csv')
	# date = datetime.datetime.now().strftime('%m-%d')
	# ts = time.time()
	# flvisual(df, ts, )
	return df

if __name__ == '__main__':
    # for test data_dis_visual
    # ls1 = [{'global_accuracy':1, 'global_loss':2}, {'global_accuracy':3, 'global_loss':5},]
    # ls2 = [{'global_accuracy':5, 'global_loss':6}, {'global_accuracy':4, 'global_loss':5},]
    # df1 = pd.DataFrame(ls1)
    # df2 = pd.DataFrame(ls2)
    # df_list = [(df1, 1), (df2, 2)]
    # data_dis_visual(df_list=df_list)
    
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
    

    # df1 = main(conf=conf, dir_alpha=0.3)
    # df2 = main(conf=conf, dir_alpha=0.6)
    # df3 = main(conf=conf, dir_alpha=1)
    df4 = main(conf=conf, dir_alpha=10)
    df_list = [(df1, 0.3), 
	       (df2, 0.6), 
		   (df3, 1), 
		   (df4, 10),
		   ]
    
    data_dis_visual(df_list=df_list)
        
		
	
		
		
	