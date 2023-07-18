import datetime
import time
import torch
from uav import *
from utils import TrainThread
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./tensorboard/log/')


class FedAvg():
    def __init__(self, conf, train_datasets, eval_datasets, ) -> None:
            self.conf = conf
            self.train_datasets = train_datasets
            self.eval_datasets = eval_datasets
            self.name = 'FedAvg'

            self.server = Server(self.conf, self.eval_datasets, compile=self.conf['compile'])

            ###
            num_clients = self.conf['f_uav_num']
            num_classes = 10
            seed = 2023
            
            
           
            hetero_dir_part = CIFAR10Partitioner(self.train_dataset.targets,
                                                num_clients,
                                                balance=None,
                                                partition="dirichlet",
                                                dir_alpha=0.3,
                                                seed=seed)

            csv_file = f"./partition-reports/cifar10_hetero_dir_0.3_{num_clients}clients.csv"
            partition_report(self.train_dataset.targets, hetero_dir_part.client_dict,
                            class_num=num_classes,
                            verbose=False, file=csv_file)
            

            self.clients = []
            for uav_id in range(conf['f_uav_num']):
                dataset_indice = hetero_dir_part.client_dict[uav_id]
                self.clients.append(Client(self.conf,  self.train_datasets, dataset_indice, uav_id, self.conf['compile']))

            acc, loss = self.server.model_eval()
            self.acc = acc
            self.loss = loss
    def reset(self,):
        # self.server.global_model.__init__() 
        # self.server.global_model.cuda()
        self.server = Server(self.conf, self.eval_datasets, compile=self.conf['compile'])
        acc, loss = self.server.model_eval()
        self.acc = acc
        self.loss = loss
        print(f'global Epoch: 0, acc: {self.acc}, loss: {self.loss}')
    def iteration(self, global_epoch, local_epochs):
        begin = time.time()
        date = datetime.datetime.now().strftime('%m-%d')

        
        candidates = []
        for i in self.conf['candidates']:
            candidates.append(self.clients[i])		
        weight_accumulator = {}
        for name, params in self.server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        global_epoch_dic = {}#for print log
        global_epoch_dic['global Epoch'] = global_epoch
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
            thread = TrainThread(candidates[i].local_train(self.server.global_model, global_epoch, local_epochs, name = self.name))
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
                global_epoch_dic[f'f_uav{candidates[i].client_id}'] = loss_dic
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
        
        
        global_epoch_dic['global_accuracy'] = acc
        global_epoch_dic['global_loss'] = loss
        
        return global_epoch_dic, acc, diff_acc, diff_loss, avg_local_loss


class FedProx(FedAvg):
    def __init__(self, conf, train_datasets, eval_datasets) -> None:
        super().__init__(conf, train_datasets, eval_datasets)
        self.name = 'FedProx'

    def iteration(self, global_epoch, local_epochs):
        return super().iteration(global_epoch, local_epochs)