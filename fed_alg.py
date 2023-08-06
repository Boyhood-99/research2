import datetime
import time
import copy
import torch
from uav import *
from utils import TrainThread
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./tensorboard/log/')
from datasets import Dataset

class FedAvg():
    def __init__(self, conf,  dir_alpha=0.3) -> None:
            self.name = 'FedAvg'
            self.conf = conf
            self.dataset = Dataset(self.conf, dir_alpha=dir_alpha)

            self.eval_datasets = self.dataset.eval_datasets
            self.server = Server(self.conf, self.eval_datasets, compile=self.conf['compile'])
            global global_model_init
            global_model_init = copy.deepcopy(self.server.global_model)
            
            
            num_clients = self.conf['f_uav_num']
            self.clients = []
            np.random.seed(2023)
            for i in range(num_clients):  
                self.clients.append(Client(self.conf,  self.dataset.train_datasets, self.dataset.dataset_indice_list[i], id=i, compile = self.conf['compile']))

            acc, loss = self.server.model_eval()
            self.acc = acc
            self.loss = loss
            
    def reset(self, ):
        # self.server.global_model.__init__() 
        # self.server.global_model.cuda()
        
        self.server.global_model = global_model_init
        acc, loss = self.server.model_eval()
        print(f'global Epoch: 0, acc: {self.acc}, loss: {self.loss}')

        self.acc = acc
        self.loss = loss
        # df_list = []
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

        
        candidates = []
        for i in self.conf['candidates']:
            candidates.append(self.clients[i])		
        weight_accumulator = {}
        for name, params in self.server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        self.global_epoch_dic = {}#for print log
        self.global_epoch_dic['global Epoch'] = global_epoch
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
        num_candidate = len(candidates)
        threads = []
        for i in range(num_candidate):
            thread = TrainThread(candidates[i].local_train(self.server.global_model, global_epoch, 
                                                           local_epochs, name = self.name, ))
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
        loss_list = []
        for i in range(num_candidate):
            diff, loss_dic = threads[i].getresult()

            loss_list.append(loss_dic[f'local epoch{local_epochs - 1} loss'])
            
            for name, params in self.server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])
                # self.global_epoch_dic[f'f_uav{candidates[i].client_id}'] = loss_dic
            # print(f"L_UAV_{self.client_id} complete the {local_epoch+1}-th local iteration ")
        #--------------------------------------------多线程

        avg_local_loss = np.array(loss_list).mean()
        
        self.server.model_aggregate(weight_accumulator)
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
        print(f'global Epoch: {global_epoch +1}, acc: {acc}, loss: {loss}, time: {diff_time}')
        
        
        self.global_epoch_dic['global_accuracy'] = acc
        self.global_epoch_dic['global_loss'] = loss
        
        return self.global_epoch_dic, acc, diff_acc, diff_loss, avg_local_loss


class FedProx(FedAvg):
    def __init__(self, conf, ) -> None:
        super().__init__(conf, )
        self.name = 'FedProx'

    def iteration(self, global_epoch, local_epochs):
        return super().iteration(global_epoch, local_epochs)