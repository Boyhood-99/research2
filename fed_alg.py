import datetime
import time
import copy
import torch
from uav import *
from utils import TrainThread, TrainThread_FedDyn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./tensorboard/log/')
from datasets import Dataset
import numpy as np

'''
implementation of different FL algorithms
'''
class FedAvg():
    def __init__(self, conf,  dir_alpha=0.3) -> None:
            np.random.seed(2023)
            torch.manual_seed(2023)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.name = 'FedAvg'
            self.conf = conf
            self.dataset = Dataset(self.conf, dir_alpha=dir_alpha)

            self.eval_dataset = self.dataset.eval_dataset
            self.server = Server(self.conf, self.eval_dataset, compile=self.conf['compile'])
            
            self.global_model_init = copy.deepcopy(self.server.global_model.state_dict())
            
            num_clients = self.conf['f_uav_num']
            self.clients = []
            for i in range(num_clients):  
                self.clients.append(Client(self.conf,  self.dataset.train_dataset, self.dataset.dataset_indice_list[i], id=i, compile = self.conf['compile']))

            acc, loss = self.server.model_eval()
            self.acc = acc
            self.loss = loss
            
    def reset(self, ):
        # self.server.global_model.__init__() 
        # self.server.global_model.cuda()
        
        self.server.global_model.load_state_dict(self.global_model_init)

        acc, loss = self.server.model_eval()
        self.acc = acc
        self.loss = loss
        print(f'global Epoch: 0, acc: {acc}, loss: {loss}')

        self.global_epoch_dic = {}  ##for print log
        self.global_epoch_dic['global Epoch'] = 0
        self.global_epoch_dic['global_accuracy'] = acc
        self.global_epoch_dic['global_loss'] = loss
        # df_list.append(self.global_epoch_dic)
        return self.global_epoch_dic
    def iteration(self, global_epoch, local_epochs, auto_lr=None):
        lr = self.conf['lr'] if auto_lr is None else auto_lr
        begin = time.time()
        date = datetime.datetime.now().strftime('%m-%d')
        
        self.candidates = []
        self.datasize_total = 0
        for i in self.conf['candidates']:
            can = self.clients[i]
            assert isinstance(can, Client)
            self.datasize_total += can.datasize
            self.candidates.append(can)		

        self.global_epoch_dic = {} #for print log
        # self.global_epoch_dic['global Epoch'] = global_epoch
        ####

        #---------------------------------------单线程，主要是local_train耗时
        # for candidate in candidates:
        # 	diff, loss_dic = candidate.local_train(server.global_model, global_epoch)
            
        # 	for name, params in server.global_model.state_dict().items():
        # 		weight_accumulator[name].add_(diff[name])
        # 	global_epoch_dic[f'f_uav{candidate.client_id}'] = loss_dic
        # 	# print(f"L_UAV_{self.client_id} complete the {local_epoch+1}-th local iteration ")
        #---------------------------------------单线程
        #---------------------------------------多线程
        num_candidate = len(self.candidates)
        threads = []
        for candidate in self.candidates:
            assert isinstance(candidate, Client)
            thread = TrainThread(candidate.local_train(self.server.global_model, global_epoch, 
                                                           local_epochs, 
                                                           name = self.name,
                                                             ))
            
            threads.append(thread)
            threads[-1].start()

        for thread in threads:
            thread.join()
        
            # diff, loss_dic = thread.getresult()
            # for name, params in server.global_model.state_dict().items():
            # 	weight_accumulator[name].add_(diff[name])
            # 	global_epoch_dic[f'f_uav{candidates[i].client_id}'] = loss_dic
        loss_list = []
        
        weight_accumulator = {}
        for name, params in self.server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        local_models_ls = {}
        weights = {}
        for i in range(num_candidate):
            thread = threads[i]
            diff, loss_dic, local_model, can_id = thread.getresult()
            local_models_ls[f'{i}'] = local_model
            weights[f'{can_id}'] = self.clients[can_id].datasize/self.datasize_total

            loss_list.append(loss_dic[f'local epoch{local_epochs - 1} loss'])
            
            for name, params in self.server.global_model.state_dict().items():
                # weight_accumulator[name].add_(diff[name])
                weight_accumulator[name].add_(diff[name])
                # self.global_epoch_dic[f'f_uav{candidates[i].client_id}'] = loss_dic
            # print(f"L_UAV_{self.client_id} complete the {local_epoch+1}-th local iteration ")
        #--------------------------------------------多线程
        avg_local_loss = np.array(loss_list).mean()
        
        # self.server.model_aggregate(weight_accumulator)
        self.server.model_agg(local_models_ls, weights)
        #-------------------------------------
        # if global_epoch <10:
        # 	acc, loss = server.model_eval()#耗时
        # 	print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
            
        # elif global_epoch >10 and global_epoch%10 == 0:
        # 	acc, loss = server.model_eval()#耗时
        # 	print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
        #---------------------------------------
        acc, loss = self.server.model_eval()    ###耗时

        diff_acc = acc - self.acc
        self.acc = acc
        diff_loss = loss - self.loss
        self.loss = loss

        diff_time = time.time() - begin
        print(f'local_Epochs: {local_epochs},\t global Epoch: {global_epoch + 1},\t acc: {acc:.2f},\t loss: {loss:.4f},\t  \
                 time: {diff_time:.2f},\t avg local loss: {avg_local_loss:.4f}')
        
        
        self.global_epoch_dic['global_accuracy'] = acc
        self.global_epoch_dic['global_loss'] = loss
        
        return self.global_epoch_dic, acc, diff_acc, diff_loss, avg_local_loss

    def get_datasize(self, ):
        datasize = []
        for i in range(len(self.clients)):
            client = self.clients[i]
            assert isinstance(client, Client)
            datasize.append(client.datasize)
        return datasize

    def update_model(self, ):
        for candidate in self.candidates:
            assert isinstance(candidate, Client)
            candidate.local_model.load_state_dict(self.server.global_model.state_dict())

class FedProx(FedAvg):
    def __init__(self, conf, ) -> None:
        super().__init__(conf, )
        self.name = 'FedProx'

    def iteration(self, global_epoch, local_epochs):
        return super().iteration(global_epoch, local_epochs)
    

class FedDyn(FedAvg):
    def __init__(self, conf, dir_alpha=0.3) -> None:
        super().__init__(conf, dir_alpha)
        self.h = {
            key: torch.zeros(params.shape, device=self.device)
            for key, params in self.server.global_model.state_dict().items()
            }
    
    def reset(self, ):
        self.h = {
            key: torch.zeros(params.shape, device=self.device)
            for key, params in self.server.global_model.state_dict().items()
            }
        return super().reset()
    def iteration(self, global_epoch, local_epochs, auto_lr=None ): 
        begin = time.time()   
        self.candidates = []
        for i in self.conf['candidates']:
            self.candidates.append(self.clients[i])		

        self.global_epoch_dic = {} #for print log
       
        num_candidate = len(self.candidates)
        #####多线程训练
        threads = []
        for candidate in self.candidates:
            assert isinstance(candidate, Client)
            prev_grads = candidate.prev_grads
            thread = TrainThread_FedDyn(candidate.train_FedDyn(self.server.global_model, prev_grads, 
                                                           local_epochs, 
                                                           name = self.name,
                                                             ))
            threads.append(thread)
            threads[-1].start()
        for thread in threads:
            thread.join()
        ########更新prev_grads
        loss_list = []
        for i in range(len(self.candidates)):
            candidate = self.candidates[i]
            thread = threads[i]
            assert isinstance(thread, TrainThread_FedDyn)
            loss, prev_grads = thread.getresult()
            assert isinstance(candidate, Client)
            candidate.prev_grads = prev_grads
            loss_list.append(loss)
        avg_local_loss = np.array(loss_list).mean()

        ####模型聚合
        h = {
            key: prev_h
            - self.conf['feddyn_alpha'] * 1 / len(self.clients) * sum(candidate.local_model.state_dict()[key] - old_params for candidate in self.candidates)
            for (key, prev_h), old_params in zip(self.h.items(), self.server.global_model.state_dict().values())
        }
        new_parameters = {
            key: (1 / len(self.candidates)) * sum(candidate.local_model.state_dict()[key] for candidate in self.candidates)
            for key in self.server.global_model.state_dict().keys()
        }
        new_parameters = {
            key: params - (1 / self.conf['feddyn_alpha']) * h_params
            for (key, params), h_params in zip(new_parameters.items(), h.values())
        }
        self.server.global_model.load_state_dict(new_parameters)
        self.h = copy.deepcopy(h)
        ###下发更新客户端模型
        self.update_model()

        ####模型评估
        acc, loss = self.server.model_eval()    ###耗时

        diff_acc = acc - self.acc
        self.acc = acc
        diff_loss = loss - self.loss
        self.loss = loss

        diff_time = time.time() - begin
        print(f'local_Epochs: {local_epochs},\t global Epoch: {global_epoch + 1},\t acc: {acc:.2f},\t loss: {loss:.4f},\t  \
                 time: {diff_time:.2f},\t avg local loss: {avg_local_loss:.4f}')
        
        self.global_epoch_dic['global_accuracy'] = acc
        self.global_epoch_dic['global_loss'] = loss
            
        return self.global_epoch_dic, acc, diff_acc, diff_loss, avg_local_loss