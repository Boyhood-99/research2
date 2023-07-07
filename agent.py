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
    def __init__(self, conf, policy_lr = 3e-6, path='./SAC/policy_sac_model',capacity=1000,size=300000) -> None:
        self.conf = conf
        self.config_train = conf['config_train']

        self.env = Environment2(self.conf)
        REPLAYBUFFER=size
        self.MEMORY_WARMUP_CAPACITY=capacity
        F_UAV_NUM=self.env.f_uav_num
        END_REWARD=self.env.end_reward
        self.s_dim = F_UAV_NUM*2+4
        self.a_dim = 6
        self.step_num = 0
        self.ep_reward = 0
        if type(self.env) == Environment0:
            self.s_dim = F_UAV_NUM*2 + 4
            self.a_dim = 3

        if type(self.env) == Environment1:
            self.s_dim = F_UAV_NUM*2 + 2
            self.a_dim = 3

        if type(self.env) == Environment2:
            self.s_dim = F_UAV_NUM*6 + 5
            # self.a_dim = 3

        self.z=ZFilter(self.s_dim)
        
        self.sac = SAC(state_dim=self.s_dim,action_dim=self.a_dim,device=self.config_train.DEVICE,
                        batch_size=self.config_train.BATCHSIZE,
                        replay_buffer_size=REPLAYBUFFER,policy_lr = policy_lr)

    def reset(self, ):
        self.step_num = 0
        self.ep_reward = 0
        s = self.env.reset()
        return s

    def episode(self, i, global_epochs, test = False): 
        s = self.reset()
        s = self.z(s)
        actions = []
        while True:
            
            if test:
                a = self.sac.test_choose_action(s) 
            else:
                a = self.sac.choose_action(s)             
            # s_, r, done ,x,y= env.step(a)
            action=deepcopy(a)          
            s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,_= self.env.step(self.step_num, action)
            ###

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
            self.step_num+=1
            self.ep_reward += r
            s=s_
            if done:
                break
        print('Episode:', i, ' Reward: %.4f' % self.ep_reward, 'Step_sum: %i' % self.step_num,' timetotal: %.4f' % self.env.time_total )
        if test:
            return -self.ep_reward, actions
        else:
            return -self.ep_reward
        

    def update(self, update_times = 200):
        if self.sac.replay_buffer.__len__() > self.MEMORY_WARMUP_CAPACITY : 
            for i in range(update_times):   #50-200np.clip((200-i)*2,50,200)
                q_,p_,alpha_=self.sac.learn()
            
                
        
        
            
            
