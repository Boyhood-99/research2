import torch
import numpy as np
import os
from env import Environment,Environment1,Environment0, Environment2
from rl_alg import SAC, DDPG, PPO
from copy import deepcopy
from configuration import ConfigTrain, ConfigDraw
from dataclasses import dataclass
from normalization import ZFilter
import gym

class AgentSAC():
    def __init__(self, conf, policy_lr = 3e-6, dir = './SAC') -> None:#capacity = 1000
        np.random.seed(2023)
        torch.manual_seed(2023)
        self.dir = dir
        assert isinstance(conf, dict)
        self.conf = conf
        self.config_train = conf['config_train']
        assert isinstance(self.config_train, ConfigTrain)
        self.env = Environment2(self.conf)
        self.buffer_size = self.config_train.BUFFER_SIZE
        self.warmup_capacity = self.config_train.WARM_UP
        F_UAV_NUM = self.env.f_uav_num
        
        self.s_dim = F_UAV_NUM*2+4
        self.a_dim = 6
        self.step_num = 0
        self.ep_reward = 0
        if type(self.env) ==  Environment0:
            self.s_dim = F_UAV_NUM*2 + 4
            self.a_dim = 3

        if type(self.env) ==  Environment1:
            self.s_dim = F_UAV_NUM*2 + 2
            self.a_dim = 3

        if type(self.env) ==  Environment2:
            self.s_dim = F_UAV_NUM*6 + 3
            # self.a_dim = 3

        self.z = ZFilter(self.s_dim)
        
        self.agent = SAC(state_dim = self.s_dim, action_dim = self.a_dim, device = self.config_train.DEVICE,
                        batch_size = self.config_train.BATCHSIZE,
                        replay_buffer_size = self.buffer_size, policy_lr = policy_lr)

    def reset(self, ):
        self.step_num = 0
        self.ep_reward = 0
        s = self.env.reset()
        return s

    def episode(self, i, global_epochs, ): 
        s = self.reset()
        s = self.z(s)
        
        energy_consum = 0 
        acc_increase = 0
        loss_decrease = 0
        while True:  
            
            # a = self.agent.choose_action(s)   
            a = self.choose_action(s)          
            action = deepcopy(a)          
            s_, r, energy_consum_, acc_increase_, loss_decrease_, done ,l, f , _, _, _, = self.env.step(self.step_num, action)
            
            s_ = self.z(s_)
            self.agent.replay_buffer.store_transition(s_,a,r,s_,done)

            self.step_num += 1
            self.ep_reward +=  r
            energy_consum +=   energy_consum_
            acc_increase +=  acc_increase_
            loss_decrease +=  loss_decrease_
            s = s_
            if done:
                break

        print(f'Episode: {i}  Reward: {self.ep_reward:.4f} Step_sum:  {self.step_num} \
              timetotal: {self.env.time_total:.2f} energy_consum:{energy_consum:.2f}')
        
        return self.ep_reward, energy_consum

    def choose_action(self, s):
        a = self.agent.choose_action(s)  
        return a    
    def episode_test(self, i, global_epochs,):
        s = self.reset()
        s = self.z(s)
        actions = []
        tra_ls = []
        
        energy_consum = 0 
        acc_increase = 0
        loss_decrease = 0
        while True:  
            a = self.agent.test_choose_action(s) 
                      
            action = deepcopy(a)          
            s_, r, energy_consum_, acc_increase_, loss_decrease_, done ,l, f , _, _, _, = self.env.step(self.step_num, action)
           
            s_ = self.z(s_)
            # self.agent.replay_buffer.store_transition(s_,a,r,s_,done)

            ###trajectory and action
            tra_ls.append({'l_x': l[0], 'l_y' :l[1], 'l_z':l[2], 'f_x': f[0], 'f_y' :f[1], 'f_z' :f[2]})
            action_dic = {}
            action_dic['velocity'] = action[0]
            action_dic['direction'] = action[1]
            action_dic['iteration'] = action[2]
            if self.a_dim > 3:
                action_dic['alpha']  = action[3:]
            actions.append(action_dic)

            ###
            self.step_num +=  1
            self.ep_reward +=  r
            energy_consum +=   energy_consum_
            acc_increase +=  acc_increase_
            loss_decrease +=  loss_decrease_
            s = s_
            if done:
                break
        
        print(f'Episode: {i}  Reward: {self.ep_reward} Step_sum:  {self.step_num} \
              timetotal: {self.env.time_total} energy_consum:{energy_consum}')
        
        return self.ep_reward, energy_consum, actions, self.env.get_dflist(), tra_ls

    def update(self, update_times = 200):
        if self.agent.replay_buffer.__len__() > self.warmup_capacity : 
            for i in range(update_times):   #50-200np.clip((200-i)*2,50,200)
                q_, p_, alpha_ = self.agent.learn()
    
    def save_model(self, ):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        torch.save(self.agent.actor.state_dict(), self.dir + '/actor.pth')
        torch.save(self.agent.critic.state_dict(), self.dir + '/critic.pth')
        print('SAC actor and critic is updated')
class AgentDDPG(AgentSAC):
    def __init__(self, conf, lr_a = 3e-6, dir = './DDPG') -> None:
        super().__init__(conf, policy_lr=lr_a, dir=dir)         
        self.noise = 0.5 #0.8
        self.replacement = [dict(name = 'soft', tau = 0.001),dict(name = 'hard', rep_iter = 600)][0] 
        self.agent = DDPG(state_dim = self.s_dim, action_dim = self.a_dim, device = self.config_train.DEVICE,
                          replacement = self.replacement, replay_buffer_size = self.buffer_size, 
                          batch_size = self.config_train.BATCHSIZE,  lr_a = lr_a,
                )

    def reset(self):
        return super().reset()
    
    def choose_action(self, s):
        a = self.agent.choose_action(s)
        a = np.clip(np.random.normal(a, self.noise), -1,1)
        return a    
    
    def episode(self, i, global_epochs):
        self.ep_reward, energy_consum = super().episode(i, global_epochs)
        if self.agent.replay_buffer.__len__() > self.warmup_capacity :
            self.noise *= 0.99
        return self.ep_reward, energy_consum

    def episode_test(self, i, global_epochs):
        return super().episode_test(i, global_epochs)
    

    ##回合更新
    def update(self, update_times = 200):
        if self.agent.replay_buffer.__len__() > self.warmup_capacity : 
            for i in range(update_times):   #50-200np.clip((200-i)*2,50,200)
                # _, _ = self.agent.learn()
                self.agent.learn()
    def save_model(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        torch.save(self.agent.net_target.state_dict(), self.dir + '/net_target.pth')
        print('DDPG net is updated')
        
        
class AgentPPO(AgentSAC):
    def __init__(self, conf, policy_lr=0.0003, dir='./PPO') -> None:
        super().__init__(conf, policy_lr, dir)
        self.gym = True
        if self.gym:
            env_name = "Pendulum-v1"
            self.env = gym.make(env_name)
            self.s_dim = self.env.observation_space.shape[0]
            self.a_dim = self.env.action_space.shape[0]
            self.z = ZFilter(self.s_dim)
        self.has_continuous_action_space = True
        self.action_std_decay_rate = 0.05        
        self.min_action_std = 0.1    
        self.agent = PPO(state_dim = self.s_dim, action_dim = self.a_dim, 
                         device = self.config_train.DEVICE, lr_a = policy_lr,
                         K_epochs = 10
                )
        
        

    def reset(self):
        return super().reset()
    
    def episode(self, i, global_epochs):
        s = self.reset()[0]
        # print(s)
        if not self.gym:
            s = self.z(s)
        
        energy_consum = 0 
        acc_increase = 0
        loss_decrease = 0
        while True:  
            
            # a = self.agent.choose_action(s)   
            a = self.choose_action(s)          
            action = deepcopy(a)  
            if self.gym:
                info = self.env.step(action)  
                s_, r, done = info[0], info[1], info[2]
                
            else:
                s_, r, energy_consum_, acc_increase_, loss_decrease_, done ,l, f , _, _, _, = self.env.step(self.step_num, action)
                energy_consum +=   energy_consum_
                acc_increase +=  acc_increase_
                loss_decrease +=  loss_decrease_
                s_ = self.z(s_)
            self.agent.replay_buffer.rewards.append(r)
            self.agent.replay_buffer.is_terminals.append(done)
            
                
            self.step_num += 1
            self.ep_reward +=  r
            
            s = s_
            if done or self.step_num%1000 == 0:
                break

        print(f'Episode: {i}  Reward: {self.ep_reward:.4f} Step_sum:  {self.step_num}'+
              #f'timetotal: {self.env.time_total:.2f}' +
              f'energy_consum:{energy_consum:.2f}')
        
        return self.ep_reward, energy_consum
    
    def episode_test(self, i, global_epochs):
        return super().episode_test(i, global_epochs)
        
    
    def update(self, update_times = 200):
        self.agent.learn()
    
    def save_model(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        torch.save(self.agent.policy.state_dict(), self.dir + '/net_target.pth')
        print('PPO net is updated')
    
    def std_decay(self, ):
        if self.has_continuous_action_space:
            self.agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)


#单步更新
# if ddpg.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY :                              
#     a_loss_,td_error_ = ddpg.learn()
#     a_loss.append(math.fabs(a_loss_.cpu().detach().numpy()))
#     td_error.append(td_error_.cpu().detach().numpy())  
# if done:
#     NOISE*= 0.99
#     print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Explore: %.2f' % NOISE,
#     'Step_sum: %i' % step_sum ,' timetotal: %.4f' % env.time_total)  
#     break   

            

        
  
    
