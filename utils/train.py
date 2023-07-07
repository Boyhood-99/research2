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
#accuracy and number 
# MAX_EPISODES=200
# BATCHSIZE=128 #256,32
# # MEMORY_WARMUP_CAPACITY=5000#10000#20000
# # REPLAYBUFFER=150000#200000#100000


class AgentTrain():
    def __init__(self, config_train) -> None:
        self.config_train = config_train
    def stoaction(self, ):

        env = Environment() 
        episode_reward=[]
        a_dim = 3
        for i in range(self.config_train.MAX_EPISODES):
        
            s = env.reset() 
            ep_reward = 0
            step_sum=0
            #test
            while True:
                    #传输时间、一轮时间和无人机位置
                
                    t_comm=[]
                    t_total=[]
                    l_uav_location=[]
                    f_uav_location=[]
                    d=[]

                    step_sum+=1
                    
                    a = np.random.uniform(low=-1,high=1, size=(a_dim,))
                    s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,d_= env.step(a)
                

                    l_uav_location.append(l_uav_location_)
                    f_uav_location.append(f_uav_location_)
                    t_comm.append(t_comm_)
                    t_total.append(t_total_)
                    d.append(d_)
                

                    ep_reward += r
                    if done :
                        print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total )
                        
                        break
                    

            episode_reward.append(-(ep_reward-7))#减去最终奖励
    
        return episode_reward,t_comm,t_total,l_uav_location,f_uav_location,d

    def fixaction(self, ):
        env = Environment() 
        episode_reward=[]

        for i in range(self.config_train.MAX_EPISODES):
            s = env.reset() 
            ep_reward = 0
            step_sum=0
            l_uav_location=[]
            f_uav_location=[]
            while True:
                    #传输时间、一轮时间和无人机位置
                
                    t_comm=[]
                    t_total=[]
                    
                    d=[]

                    step_sum+=1
                    
                    # a = np.array([-0.7,0,1]) #I=30，
                    # a = np.array([-0.7,0,-1]) #I=1，
                    # a = np.array([-0.7,0,0.30]) #I=20，
                    # a = np.array([-0.7,0,-0.04])  #I=15，
                    # a = np.array([-0.7,0,-0.38]) #I=10，
                    # a = np.array([-0.6,0,-0.73])  #I=5
                    # a = np.array([-0.3,0,-0.93])  #I=2，
                    a = np.array([-0.4,0,-0.38])  #I=2，

                    s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,d_= env.step(a)
                
                    l_uav_location.append(l_uav_location_)
                    f_uav_location.append(f_uav_location_)
                    t_comm.append(t_comm_)
                    t_total.append(t_total_)
                    d.append(d_)

                    ep_reward += r
                    if done :
                        print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total )
                        break
                    

            episode_reward.append(-ep_reward)#减去最终奖励
    
        return episode_reward,t_comm,t_total,l_uav_location,f_uav_location,d

    def ddpg_train(self, lr_a=3e-7, lr_c=1e-5, path='./ddpg_model',env=Environment(),capacity=10000,size=300000):
        NOISE=0.8
        REPLACEMENT = [dict(name='soft', tau=0.001),dict(name='hard', rep_iter=600)][0] 
        env = env
        REPLAYBUFFER=size
        MEMORY_WARMUP_CAPACITY=capacity
        F_UAV_NUM=env.f_uav_num
        END_REWARD=env.end_reward
    
        s_dim = F_UAV_NUM*2+4
        a_dim = 6
        if type(env) == Environment0:
            s_dim = F_UAV_NUM*2 + 4
            a_dim = 3

        if type(env) == Environment1:
            s_dim = F_UAV_NUM*2 + 2
            a_dim = 3

        if type(env) == Environment2:
            s_dim = F_UAV_NUM*6 + 5
            # a_dim = 3

        z=ZFilter(s_dim)
        ddpg = DDPG(state_dim=s_dim,action_dim=a_dim,device=self.config_train.DEVICE,replacement=REPLACEMENT,
                    replay_buffer_size=REPLAYBUFFER,batch_size=self.config_train.BATCHSIZE,lr_a=lr_a, lr_c=lr_c
                    )

        t_comm=[]
        t_total=[]
        l_uav_location=[]
        f_uav_location=[]
        d=[]

        episode_reward=[]
        a_loss=[]
        td_error=[]
    
        for i in range(self.config_train.MAX_EPISODES):
            
            a_=[]
            s = env.reset() 
            s=z(s)
            ep_reward = 0
            step_sum=0
            #for test
            if i == self.config_train.MAX_EPISODES-1:  
                while True:
                    step_sum+=1
                    a = ddpg.test_choose_action(s)     
                    s_, r, done ,t_comm_,t_total_,l,f,d_= env.step(a)
                    action=deepcopy(a)

                    l_uav_location.append(l)
                    f_uav_location.append(f)
                    t_comm.append(t_comm_)
                    t_total.append(t_total_)
                    d.append(d_)
                    a_.append(action)

                    s_=z(s_)
                    ep_reward += r
                    if done :
                        print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Explore: %.2f' % NOISE,'Step_sum: %i' % step_sum )
                        torch.save(ddpg.net_target,path)
                        print(a_)
                        break
                    s = s_
        #for train
            else:
                while True:
                    step_sum+=1
                    a = ddpg.choose_action(s)
                    a = np.clip(np.random.normal(a, NOISE), -1,1) 
                    action=deepcopy(a) 
                    s_, r, done ,_,_,_,_,d_= env.step(action)

                    s_=z(s_)
                    ep_reward += r
                    ddpg.replay_buffer.store_transition(s,a,r,s_,done)
                                        
                    if ddpg.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY and done:
                        NOISE*=0.99

                    #回合更新
                    if  done  :
                        if ddpg.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY :
                            for j in range(200):                
                                a_loss_,td_error_=ddpg.learn()
                                a_loss.append(math.fabs(a_loss_.cpu().detach().numpy()))
                                td_error.append(td_error_.cpu().detach().numpy())  
                        
                        print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Explore: %.2f' % NOISE,
                        'Step_sum: %i' % step_sum ,' timetotal: %.4f' % env.time_total)  
                        break      
                    
                    #单步更新
                    # if ddpg.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY :                              
                    #     a_loss_,td_error_=ddpg.learn()
                    #     a_loss.append(math.fabs(a_loss_.cpu().detach().numpy()))
                    #     td_error.append(td_error_.cpu().detach().numpy())  
                    # if done:
                    #     NOISE*=0.99
                    #     print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Explore: %.2f' % NOISE,
                    #     'Step_sum: %i' % step_sum ,' timetotal: %.4f' % env.time_total)  
                    #     break   

                    s = s_

            episode_reward.append(-(ep_reward-END_REWARD))#减去最终奖励
    
        return episode_reward,a_loss,td_error,t_comm,t_total,l_uav_location,f_uav_location,d

    #for trajectory sac

    def sac_train_trajectory(self, policy_lr = 3e-7,path='./SAC/policy_sac_model',I=np.array(-1/29),env=Environment(),capacity=10000,size=300000):
        env = env 
        F_UAV_NUM=env.f_uav_num
        REPLAYBUFFER=size
        MEMORY_WARMUP_CAPACITY=capacity
        END_REWARD=env.end_reward
        # NET_TARGET=torch.load('./2561281281024/ddpg_model2')
        
        s_dim = F_UAV_NUM*2+2
        a_dim = 2
        
        z=ZFilter(s_dim)
        #a_bound = env.action_space.high
        sac = SAC(state_dim=s_dim,action_dim=a_dim,device=self.config_train.DEVICE,batch_size=self.config_train.BATCHSIZE,
        replay_buffer_size=REPLAYBUFFER,policy_lr = policy_lr)
        episode_reward=[]
    
        t_comm=[]
        t_total=[]
        l_uav_location=[]
        f_uav_location=[]
        d=[]

        entropy=[]
        q=[]
        p=[]
        alpha=[]
        for i in range(self.config_train.MAX_EPISODES):
            a_=[]
            s = env.reset() 
            s=z(s)
            ep_reward = 0
            step_sum=0
            # for test
            if i == self.config_train.MAX_EPISODES-1:
                while True:
                    
                    step_sum+=1
                    a = sac.test_choose_action(s) 
                    # I=np.random.randint(1,30)
                    I=I
                    action=np.append(a,I)
                    
                    s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,d_= env.step(action)
                    s_=z(s_)

                    l_uav_location.append(l_uav_location_)
                    f_uav_location.append(f_uav_location_)
                    t_comm.append(t_comm_)
                    t_total.append(t_total_)
                    d.append(d_)
                    a_.append(action)

                    ep_reward += r
                    if done :
                        print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum )
                        torch.save(sac.policy_net,path)
                        print(a_)
                        break
                    s=s_
        #for train   
            else:  
                while True:
                    step_sum+=1
                    a = sac.choose_action(s)
                    # I=np.random.randint(1,20)
                    I=I
                    action=np.append(a,I)
                    s_, r, done ,_,_,_,_,_= env.step(action)           
                    s_=z(s_)
                    sac.replay_buffer.store_transition(s,a,r,s_,done)
                    ep_reward += r
        
                    if done :
                        if sac.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY : 
                            for j in range(200) :
                                q_,p_,alpha_=sac.learn()
                                # entropy.append(log)
                                q.append(q_)
                                p.append(p_)
                                alpha.append(alpha_)
                        print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total ) 
                        break  
                    s = s_    

            episode_reward.append(-(ep_reward-END_REWARD))#减去最终奖励
    
        return episode_reward,q,p,alpha,t_comm,t_total,l_uav_location,f_uav_location,d


    def sac_train_federated(self, policy_lr = 3e-7,path='./SAC/policy_sac_model',env=Environment(),capacity=10000,size=300000):
        env = env 
        F_UAV_NUM=env.f_uav_num
        REPLAYBUFFER=size
        MEMORY_WARMUP_CAPACITY=capacity
        END_REWARD=env.end_reward

        s_dim = F_UAV_NUM*2+2
        a_dim = 1
        z=ZFilter(s_dim)
        #a_bound = env.action_space.high
        sac = SAC(state_dim=s_dim,action_dim=a_dim,device=self.config_train.DEVICE,batch_size=self.config_train.BATCHSIZE,
        replay_buffer_size=REPLAYBUFFER,policy_lr = policy_lr)
        episode_reward=[]
    
        t_comm=[]
        t_total=[]
        l_uav_location=[]
        f_uav_location=[]
        d=[]

        entropy=[]
        q=[]
        p=[]
        alpha=[]
        for i in range(self.config_train.MAX_EPISODES):
            a_=[]
            s = env.reset() 
            s=z(s)
            ep_reward = 0
            step_sum=0
            #for test
            if i == self.config_train.MAX_EPISODES-1:
                while True:
                    step_sum+=1
                    a = sac.test_choose_action(s) 
                    #随机或者固定
                    # v_theta=np.random.uniform(size=(2))
                    v_theta=[-2/7,0]
                    action=np.concatenate([v_theta,a])
                
                    s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,d_= env.step(action)

                    s_=z(s_)

                    l_uav_location.append(l_uav_location_)
                    f_uav_location.append(f_uav_location_)
                    t_comm.append(t_comm_)
                    t_total.append(t_total_)
                    # d.append(d_)
                    a_.append(action)

                    ep_reward += r
                    if done :
                        print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum )
                        torch.save(sac.policy_net,path)
                        print(a_)
                        break
                    s=s_
        #for train   
            else:  
                while True:
                    step_sum+=1
                    a = sac.choose_action(s)
                    # v_theta=np.random.uniform(size=(2))
                    v_theta=[-2/7,0]
                    action=np.concatenate([v_theta,a])
                    
                    s_, r, done ,_,_,_,_,_= env.step(action)
                
                    s_=z(s_)
                    sac.replay_buffer.store_transition(s,a,r,s_,done)
                    ep_reward += r
        
                    if done :
                        if sac.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY : 
                            for j in range(200) :
                                q_,p_,alpha_=sac.learn()
                                # entropy.append(log)
                                q.append(q_)
                                p.append(p_)
                                alpha.append(alpha_)
                        print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total ) 
                        break  
                    s = s_    

            episode_reward.append(-(ep_reward-END_REWARD))#减去最终奖励
    
        return episode_reward,q,p,alpha,t_comm,t_total,l_uav_location,f_uav_location,d


    #for sac trajectory and federated optimization, policy_lr = 3e-7, soft_tau=0.005,soft_q_lr = 1e-5,policy_lr = 1e-4,
    def sac_train(self, policy_lr = 3e-6, path='./SAC/policy_sac_model',env=Environment(),capacity=1000,size=300000):#3e-7
        env = env
        REPLAYBUFFER=size
        MEMORY_WARMUP_CAPACITY=capacity
        F_UAV_NUM=env.f_uav_num
        END_REWARD=env.end_reward
        s_dim = F_UAV_NUM*2+4
        a_dim = 6
        if type(env) == Environment0:
            s_dim = F_UAV_NUM*2 + 4
            a_dim = 3

        if type(env) == Environment1:
            s_dim = F_UAV_NUM*2 + 2
            a_dim = 3

        if type(env) == Environment2:
            s_dim = F_UAV_NUM*6 + 5
            # a_dim = 3

        z=ZFilter(s_dim)
        #a_bound = env.action_space.high
        sac = SAC(state_dim=s_dim,action_dim=a_dim,device=self.config_train.DEVICE,batch_size=self.config_train.BATCHSIZE,
        replay_buffer_size=REPLAYBUFFER,policy_lr = policy_lr)
        episode_reward=[]
    
        t_comm=[]
        t_total=[]
        l_uav_location=[]
        f_uav_location=[]
        # d=[]
        max_distance=[]
        total_time=[]

        entropy=[]
        q=[]
        p=[]
        alpha=[]
        for i in range(self.config_train.MAX_EPISODES):
            a_=[]
            s = env.reset() 
            s=z(s)
            ep_reward = 0
            step_sum=0
            
            #for test

            if i == self.config_train.MAX_EPISODES-1:
                while True:
                    step_sum+=1
                    a = sac.test_choose_action(s) 
                    # a = sac.choose_action(s)             
                    # s_, r, done ,x,y= env.step(a)
                                
                    s_, r, done ,t_comm_,t_total_,l_uav_location_,f_uav_location_,_= env.step(a)
                    action=deepcopy(a)
                    s_=z(s_)

                    l_uav_location.append(l_uav_location_)
                    f_uav_location.append(f_uav_location_)
                    t_comm.append(t_comm_)
                    t_total.append(t_total_)
                    # d.append(d_)
                    action_dic = {}
                    action_dic['velocity'] = action[0]
                    action_dic['direction'] = action[1]
                    action_dic['iteration'] = action[2]
                    if a_dim > 3:
                        action_dic['alpha']  = action[3:]

                    a_.append(action_dic)

                    ep_reward += r
                    if done :
                        print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum )
                        torch.save(sac.policy_net,path)
                        
                        break
                    s=s_
        #for train   
            else:  
                while True:
                    d=[]
                    
                    step_sum+=1
                    a = sac.choose_action(s)
                    action=deepcopy(a)
                    s_, r, done ,_,_,_,_,d_= env.step(action)
                    ###
                    ###
                    # s_, r, done ,x,y= env.step(a)
                    s_=z(s_)
                    sac.replay_buffer.store_transition(s,a,r,s_,done)

                    ep_reward += r
                    d.append(d_)

                    # #回合更新
                    if done :
                        if sac.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY : 
                            for j in range(200):   #50-200np.clip((200-i)*2,50,200)
                                q_,p_,alpha_=sac.learn()
                                # entropy.append(log)
                                q.append(q_)
                                p.append(p_)
                                alpha.append(alpha_)
                        print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total ) 
                        max_distance.append(np.max(d))
                        total_time.append(env.time_total)
                        # print(env.flytime)
                        break  


                    #单步更新
                    
                    # if sac.replay_buffer.__len__() > MEMORY_WARMUP_CAPACITY : 
                    #     q_,p_,alpha_=sac.learn()
                    #     # entropy.append(log)
                    #     q.append(q_)
                    #     p.append(p_)
                    #     alpha.append(alpha_)
                    # if done:
                    #     print('Episode:', i, ' Reward: %.4f' % ep_reward, 'Step_sum: %i' % step_sum,' timetotal: %.4f' % env.time_total )  
                    #     max_distance.append(np.max(d))
                    #     break  

                    s = s_    
            
            if env.time_total > env.time_max:
                ep_reward += env.time_penalty
            episode_reward.append(-(ep_reward-END_REWARD))#减去最终奖励
            
            

            # episode_reward.append(-ep_reward)
        # print(a_[:200])
        # print(a_[-50:-1])
        pd.DataFrame(a_).to_csv("./output/test_action.csv")

        return episode_reward,q,p,alpha,t_comm,t_total,l_uav_location,f_uav_location,max_distance,total_time

