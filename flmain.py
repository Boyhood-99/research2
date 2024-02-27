import copy, json
import datetime
import torch
import pandas as pd
from uav import *
from datasets import Dataset
from tqdm import tqdm
import time
import numpy as np
from utils import TrainThread
from configuration import ConfigDraw, ConfigTrain
from fed_alg import FedAvg, FedProx, FedDyn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from visualization import flvisual, data_dis_visual
import random
writer = SummaryWriter('./tensorboard/log/')

#for FL framework training

def main(conf, dir_alpha = 0.3, fl_name = 'FedAvg'):
	# torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
	# torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
	torch.set_float32_matmul_precision('high')
	torch.manual_seed(2023)
	np.random.seed(2023)
	random.seed(2023)
	is_beam = True
	ula_num = 3
	
	if fl_name == 'FedAvg':
		fl = FedAvg(conf = conf, dir_alpha=dir_alpha, feddecorr = False)
	elif fl_name == 'FedProx':
		fl = FedProx(conf = conf, dir_alpha=dir_alpha)
	elif fl_name == 'FedDyn':
		fl = FedDyn(conf = conf, dir_alpha=dir_alpha)
	else:
		fl = FedAvg(conf = conf, dir_alpha=dir_alpha)
	df_list = []
	global_epoch_dic = fl.reset()
	df_list.append(global_epoch_dic)

	df = pd.read_csv(f'./output/main_output/SAC/actions{is_beam}{ula_num}.csv')
	local_epochs_ls = df['iteration']
	# local_epochs = np.random.randint(2, 11)
	for global_epoch in tqdm(range(conf["global_epochs"])):
		local_epochs = int(local_epochs_ls[global_epoch])
		global_epoch_dic, acc, diff_acc, diff_loss, avg_local_loss = \
		fl.iteration(global_epoch, local_epochs, candidate_index=conf["config_train"].CONDIDATE)
		df_list.append(global_epoch_dic)

	
	df = pd.DataFrame(df_list)
	df.to_csv(f'./output/FL_main_output/{fl_name}{dir_alpha}.csv')
	
	ts = datetime.datetime.now().strftime('%m-%d')
	# flvisual(df, ts, )
	return df

if __name__ == '__main__':
	a = 'fed'
	print(f'mmlm/{a}.csv')
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


	#### fl alg
	fl_name_ls = ['FedAvg', 'FedProx', 'FedDyn', 'FedDecorr']
	for fl_name in fl_name_ls:
		main(conf=conf, dir_alpha=0.3)

	####data  distribution
	# main(conf=conf, dir_alpha=0.3)
	# main(conf=conf, dir_alpha=0.6)
	# main(conf=conf, dir_alpha=1)
	# main(conf=conf, dir_alpha=10)
	# df1 = pd.read_csv(f'./output/FL_main_output/FedAvg{0.3}.csv')
	# df2 = pd.read_csv(f'./output/FL_main_output/FedAvg{0.6}.csv')
	# df3 = pd.read_csv(f'./output/FL_main_output/FedAvg{1}.csv')
	# df4 = pd.read_csv(f'./output/FL_main_output/FedAvg{10}.csv')

	# df_list = [
	# 		(df1, 0.3), 
	# 		(df2, 0.6), 
	# 		(df3, 1), 
	# 		(df4, 10),
	# 		]
	# data_dis_visual(df_list=df_list)
		
	## iid, noniid
	
		
		
	
		
		
	