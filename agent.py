import torch
import numpy as np
from env import Environment,Environment1,Environment0, Environment2
from rl_alg import SAC,DDPG
from copy import deepcopy

from normalization import ZFilter
import math
from configuration import ConfigTrain
import pandas as pd

from fed_alg import FedAvg
import datasets

class AgentSAC():
    def __init__(self, conf, policy_lr = 3e-6, path='./SAC/policy_sac_model',env=Environment2(),capacity=1000,size=300000) -> None:
        self.conf = conf
        self.config_train = conf['config_train']

        self.env = env
        REPLAYBUFFER=size
        self.MEMORY_WARMUP_CAPACITY=capacity
        F_UAV_NUM=self.env.f_uav_num
        END_REWARD=self.env.end_reward
        self.s_dim = F_UAV_NUM*2+4
        self.a_dim = 6
        self.step_sum = 0
        self.ep_reward = 0
        if type(env) == Environment0:
            self.s_dim = F_UAV_NUM*2 + 4
            self.a_dim = 3

        if type(env) == Environment1:
            self.s_dim = F_UAV_NUM*2 + 2
            self.a_dim = 3

        if type(env) == Environment2:
            self.s_dim = F_UAV_NUM*6 + 5
            # self.a_dim = 3

        self.z=ZFilter(self.s_dim)
        
        self.sac = SAC(state_dim=self.s_dim,action_dim=self.a_dim,device=self.config_train.DEVICE,
                        batch_size=self.config_train.BATCHSIZE,
                        replay_buffer_size=REPLAYBUFFER,policy_lr = policy_lr)
        self.train_datasets, self.eval_datasets = datasets.get_dataset("./data/", conf["type"])
        self.fl = FedAvg(conf = conf, train_datasets = self.train_datasets, eval_datasets = self.eval_datasets)

    def reset(self, ):
        self.step_sum = 0
        self.ep_reward = 0
        s = self.env.reset()
        return s

    def episode(self, i, global_epochs, test = False): 
        s = self.reset()
        s = self.z(s)
        actions = []
        # for i in range(global_epochs):
        while True:
            self.step_sum+=1
            if test:
                a = self.sac.test_choose_action(s) 
            else:
                a = self.sac.choose_action(s)             
            # s_, r, done ,x,y= env.step(a)
            action=deepcopy(a)          
            s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,_= self.env.step(action)
            ###
            # global_epoch_dic, acc = self.fl.globaliter(global_epoch = self.step_sum)
            ###
            
            s_=self.z(s_)
            self.sac.replay_buffer.store_transition(s_,a,r,s_,done)

            if test:
                action_dic = {}
                action_dic['velocity'] = action[0]
                action_dic['direction'] = action[1]
                action_dic['iteration'] = action[2]
                if self.a_dim > 3:
                    action_dic['alpha']  = action[3:]
                actions.append(action_dic)

            self.ep_reward += r
            s=s_
            if done:
                break
        print('Episode:', i, ' Reward: %.4f' % self.ep_reward, 'Step_sum: %i' % self.step_sum,' timetotal: %.4f' % self.env.time_total )
        if test:
            return -self.ep_reward, actions
        else:
            return -self.ep_reward
        

    def update(self, update_times = 200):
        if self.sac.replay_buffer.__len__() > self.MEMORY_WARMUP_CAPACITY : 
            for i in range(update_times):   #50-200np.clip((200-i)*2,50,200)
                q_,p_,alpha_=self.sac.learn()
            
                
        
        
            
            
