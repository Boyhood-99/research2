import datetime
import torch
from uav import *
from client import *
from utils import TrainThread
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./tensorboard/log/')


class FedAvg():
    def __init__(self, conf, train_datasets, eval_datasets, ) -> None:
            self.conf = conf
            self.server = Server(conf, eval_datasets, compile=conf['compile'])
            self.clients = []
            
            for uav_id in range(conf['f_uav_num']):
                self.clients.append(Client(conf, self.server.global_model, train_datasets, uav_id, conf['compile']))
	

    def globaliter(self, global_epoch):
        date = datetime.datetime.now().strftime('%m-%d')

        # parser = argparse.ArgumentParser(description='Federated Learning')
        # parser.add_argument('-c', '--config', dest='conf')
        # args = parser.parse_args()

       
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
            thread = TrainThread(candidates[i].local_train(self.server.global_model, global_epoch))
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
            
            for name, params in self.server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])
                global_epoch_dic[f'f_uav{candidates[i].client_id}'] = loss_dic
            # print(f"L_UAV_{self.client_id} complete the {local_epoch+1}-th local iteration ")
        #--------------------------------------------多线程
        
        
        self.server.model_aggregate(weight_accumulator)
        #-------------------------------------
        # if global_epoch <10:
        # 	acc, loss = server.model_eval()#耗时
        # 	print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
            
        # elif global_epoch >10 and global_epoch%10 == 0:
        # 	acc, loss = server.model_eval()#耗时
        # 	print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
        #---------------------------------------
        acc, loss = self.server.model_eval()#耗时

        

        print(f'global Epoch: {global_epoch}, acc: {acc}, loss: {loss}')
        

        global_epoch_dic['global_accuracy'] = acc
        global_epoch_dic['global_loss'] = loss
        
        return global_epoch_dic, acc