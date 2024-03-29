import math
import numpy as np
import time
from systemmodel import SystemModel, SystemModel0, SystemModel1, SystemModel2
from copy import deepcopy
from fed_alg import FedAvg, FedDyn
from datasets import Dataset 
from configuration import *
'''
every environment represents a different design 
for performance comparison
'''


#state space based on location, state excludes new phi parameter
class Environment(object):
    #x_range = [-20, 2000], y_range = [-100, 100]
    def __init__(self, conf, f_uav_num = 5, epsilon = 0.001, end_reward = 0, time_max = 80, 
                #  x_range = [-20, 2000], y_range = [-100, 100],
                #  x_range = [-20, 1000], y_range = [-500, 500],
                #  x_range = [-20, 4000], y_range = [-500, 500],
                 x_range = [-20, 1000], y_range = [-200, 200],
                 ):
        self.conf = conf
        self.f_uav_num = self.conf['config_train'].UAV_NUM
        self.l_uav_num = 1
        self.f_uav_H = 140               # 无人机的飞行高度
        self.l_uav_H = 150

        self.distance_max = 200
        self.time_max = time_max
        self.end_reward = end_reward
        self.time_penalty = 0
       
        #动作约束
        self.theta_min  = -math.pi/2
        self.theta_max  = math.pi/2
        self.v_min = 10
        self.v_max = 10

        self.a_min = 0
        self.a_max = 2#20
        # self.v_min = 20
        # self.v_max = 30
        
        self.x_range = x_range
        self.y_range = y_range
        
        self.l_v_min = 5
        self.l_v_max = 20
        self.a_v_min = 0
        self.a_v_max = 5
        # self.l_v_min = 5
        # self.l_v_max = 40

        #analog beamforming
        self.alpha_z_min = -1
        self.alpha_z_max = 0

        self.t_uav_f = 2*10**9

        #FL parameters, γ, L, ξ are set as 2, 4, and 0.1 respectively. The learning rate is λ = 0.25 
        self.gamma = 2
        self.L = 4
        self.ksi = 0.1#  0.1
        self.lamda = 0.25#0.1
        self.epsilon = epsilon           #0.1#0.001
        
        # self.eta = 0.5#0.01
        # self.eta_min = 0.001
        # self.eta_max = 0.78#1

        self.iteration = 5
        self.iteration_min = 1
        self.iteration_max = 30

        self.local_epochs= 5
        self.local_epochs_min = 2
        self.local_epochs_max = 10

        self.auto_lr= 5
        self.auto_lr_min = 0.01
        self.auto_lr_max = 0.1

        self.step_num = 0
        
        
        self.num_episode = 0
        # self.l_uav_location = np.zeros(2)
        self.outofdistance = 0

        self.direction_flag_x = np.ones(self.f_uav_num)#方向控制
        self.direction_flag_y = np.ones(self.f_uav_num)#方向控制
        
    # 定义每一个episode的初始状态，实现随机初始化
    def reset(self):
        #随机初始化底层无人机位置，范围{0，1000}
        #self.f_uav_location = np.random.uniform(0, 1000, size = (self.f_uav_num, 2))
        self.f_uav_location = np.zeros((self.f_uav_num, 2))

        #随机初始化顶层无人机位置
        self.l_uav_location = np.zeros(2)

        # 创建初始观察值state, 包括底层无人机的位置(f_uav_num, 2)、顶层无人机的位置(f_uav_num, 2)、飞行时间

        self.local_accuracy_sum = np.zeros(1)
        self.time_total = 0
        self.state = np.hstack((self.f_uav_location.reshape(self.f_uav_num*2, ), self.l_uav_location, self.time_total, self.local_accuracy_sum))


        return self.state

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        state = self.state
        
        f_uav_location = state[:self.f_uav_num*2]
        f_uav_location = f_uav_location.reshape((self.f_uav_num, 2))
        
        l_uav_location = state[self.f_uav_num*2:self.f_uav_num*2 + 2]
        
     
        time_total = state[self.f_uav_num*2 + 2:self.f_uav_num*2 + 3]
        local_accuracy_sum = state[self.f_uav_num*2 + 3:self.f_uav_num*2 + 4]
        
      
        action = action
        action[0] = (self.l_v_max-self.l_v_min)/2*action[0] + (self.l_v_max + self.l_v_min)/2
        action[1] = math.pi/2*action[1]
        action[2] = math.ceil((self.iteration_max-self.iteration_min)/2*action[2] + (self.iteration_max + self.iteration_min)/2)
        # action[3:6] = action[3:6]
        action[5] = (self.alpha_z_max-self.alpha_z_min)/2*action[5] + (self.alpha_z_max + self.alpha_z_min)/2 
        alpha = action[3:]
       
        distance = self.systemmodel.Distance(f_uav_location, l_uav_location) 
        gain, index = self.systemmodel.Gain(f_uav_location, l_uav_location, alpha, distance, self.num_episode)

        # episode_distance = []
        # for i in index:
        #     episode_distance.append(distance[i])
        

        t_down_ = self.systemmodel.t_down(index, distance, gain)
        t_up_ = self.systemmodel.ofdma_t_up(index, distance, gain)

        t_comp = self.systemmodel.t_comp(index, action[2])
        t_agg = self.systemmodel.t_agg(self.t_uav_f)
        
        #此处无人机飞行时间为当前状态下，底层无人机计算时间(与本次决策相关)和上下行传输时间(只与当前位置有关)
        fly_time = np.max(t_comp + t_down_ + t_up_) #+ t_agg
        next_time_total = time_total + fly_time
        self.time_total = next_time_total


        #根据action，环境进行下一步, 

        
        #底层无人机速度和方向是随机的
        f_uav_v = np.random.uniform(self.v_min, self.v_max, size = (self.f_uav_num, 1))
        f_uav_theta = np.random.uniform(self.theta_min, self.theta_max, size = (self.f_uav_num, 1))

        #底层无人机新坐标
        next_f_uav_location = []
        for i in range(self.f_uav_num):
            for j in range(2):
                if j == 0:
                    next_f_uav_location_x = f_uav_location[i][0] + fly_time*f_uav_v[i]*math.cos(f_uav_theta[i])
                    next_f_uav_location_x = np.array(max(min(
                            next_f_uav_location_x, self.x_range[1]), self.x_range[0])).reshape(1,)
                    next_f_uav_location.append(np.array(next_f_uav_location_x).reshape(1,))

                else:
                    next_f_uav_location_y = f_uav_location[i][1] + fly_time*f_uav_v[i]*math.sin(f_uav_theta[i])
                    next_f_uav_location_y = np.array(max(min(
                        next_f_uav_location_y, self.y_range[1]), self.y_range[0])).reshape(1,)
                    next_f_uav_location.append(np.array(next_f_uav_location_y).reshape(1,))
                
        next_f_uav_location = np.array(next_f_uav_location)
        next_f_uav_location = next_f_uav_location.reshape(5 ,2)

        #顶层无人机新坐标
     
        l_uav_location[0] = l_uav_location[0] + fly_time*action[0]*math.cos(action[1])
        l_uav_location[1] = l_uav_location[1] + fly_time*action[0]*math.sin(action[1]) 

        l_uav_location[0] = max(min(
            l_uav_location[0], self.x_range[1]), self.x_range[0])
        l_uav_location[1] = max(min(
            l_uav_location[1], self.y_range[1]), self.y_range[0])
        
        self.l_uav_location = deepcopy(l_uav_location)
#-----------------------------------------------------------------------下一个状态带来的时隙时间长度,涉及reward
        distance = self.systemmodel.Distance(next_f_uav_location, l_uav_location) 
        gain, index = self.systemmodel.Gain(next_f_uav_location, l_uav_location, alpha, distance, self.num_episode)

        t_down_ = self.systemmodel.t_down(index, distance, gain)
        t_up_ = self.systemmodel.ofdma_t_up(index, distance, gain)

        t_comp = self.systemmodel.t_comp(index, action[2])
        t_agg = self.systemmodel.t_agg(self.t_uav_f)
        
        #此处无人机飞行时间为当前状态下，底层无人机计算时间(与本次决策相关)和上下行传输时间(只与当前位置有关)
        next_time = np.max(t_comp + t_down_ + t_up_) #+ t_agg
#-------------------------------------------------------------------------------------
        #局部精度公式求和
        yita = math.exp(-action[2]*(2-self.L*self.lamda)*self.lamda*self.gamma/2)
        local_accuracy = math.log(1-((1-yita)*self.gamma**2*self.ksi)/(2*self.L**2))
        next_local_accuracy_sum = local_accuracy_sum + local_accuracy
#----------------------------------------------------------------------------------------       
        #判断该观察状态下，本次episode是否结束

        done = 1 if next_local_accuracy_sum < math.log(self.epsilon) else 0
        done = np.array(done)
        

        #环境状态改变, 下一个state
        next_state = np.hstack((next_f_uav_location.reshape(self.f_uav_num*2, ), l_uav_location.reshape(2, ), self.time_total, 
                              next_local_accuracy_sum))
        
       
        self.state = next_state.astype(float)
#------------------------------------------------------------------------------------
        #奖励函数求解
        # reward1 = np.min(gain)
        reward1 = 0
        reward2 = np.array(( -next_time) * self.systemmodel.p_fly(action[0])/ 1000) 
      

        #结算奖励
        if done :
            reward2 += self.end_reward
            self.num_episode += 1
            if self.time_total > self.time_max:
                reward2 -= self.time_penalty
#--------------------------------------------------------------------------------
        l = deepcopy(self.l_uav_location)
        f = deepcopy(next_f_uav_location)
        d = deepcopy(np.max(distance))
        reward = reward1 + reward2
        
        return self.state, reward, done, np.max(t_up_ + t_down_), np.max(t_comp + t_up_ + t_down_), l, f, d

#state space based on location, state includes new phi parameter
class Environment2(Environment):
    def __init__(self, conf, ):
        super().__init__(conf=conf)
        self.conf = conf
        assert isinstance(self.conf['config_train'], ConfigTrain)
        self.is_beam = self.conf['config_train'].IS_BEAM
        self.is_norm_man = self.conf['config_train'].IS_NORM_MAN
        self.fl = FedAvg(conf = self.conf, dir_alpha=0.3, feddecorr=True)
        # self.fl = FedDyn(conf = self.conf, dir_alpha=0.3)
        datasize = self.fl.get_datasize()
        self.systemmodel = SystemModel2(datasize, f_uav_num = self.f_uav_num, 
                                        ula_num=self.conf['config_train'].ULA_NUM,
                                        m_num = self.conf['config_train'].M_NUM,
                                       )
        if not self.is_beam:
            self.systemmodel = SystemModel(datasize, f_uav_num = self.f_uav_num, 
                                        is_beam=False,
                                        m_num = self.conf['config_train'].M_NUM,
                                       )
    def generate_position(self, init_position, time_max):
        '''
        @todo
        '''
        # f_uav_v = np.random.uniform(self.v_min, self.v_max, size = (self.f_uav_num, 1))
        # f_uav_theta = np.random.uniform(self.theta_min, self.theta_max, size = (self.f_uav_num, 1))

        # #底层无人机新坐标
        # # next_f_uav_location = []
        # for i in range(self.f_uav_num):
        #     for j in range(2):
        #         if j == 0:
        #             f_uav_location[i][0] += fly_time*f_uav_v[i]*math.cos(f_uav_theta[i])
        #             f_uav_location[i][0] = max(min(
        #                     f_uav_location[i][0], self.x_range[1]), self.x_range[0])

        #         else:
        #             f_uav_location[i][1] += fly_time*f_uav_v[i]*math.sin(f_uav_theta[i])
        #             f_uav_location[i][1] = max(min(
        #                 f_uav_location[i][1], self.y_range[1]), self.y_range[0])
            
        
        # next_f_uav_location = np.array(f_uav_location)#.reshape(self.f_uav_num, 3)
        pass

    def reset(self):
        self.df_list = []
        self.step_num = 0
        self.time_total = 0
        ####fl
        global_epoch_dic = self.fl.reset()
        self.df_list.append(global_epoch_dic)
        
        #随机初始化底层无人机位置，范围{0，1000}
        # np.random.seed(1)
        self.f_uav_location = np.random.uniform(0, 50, size = (self.f_uav_num, 3))
        # self.f_uav_location = np.zeros((self.f_uav_num, 3))
        for i in range(self.f_uav_num):
            self.f_uav_location[i, 2] = self.f_uav_H
        #随机初始化顶层无人机位置
        self.l_uav_location = np.zeros(3)
        self.l_uav_location[2] = self.l_uav_H
        # self.gain = np.zeros(())
        if self.is_beam:
            #如果初始化不当，当智能体选择边界动作或者其他动作时，导致phi为0，天线增益为0，引发错误, 因此建议根据位置算 
            init_distance = self.systemmodel.Distance(f_uav_location=self.f_uav_location, l_uav_location=self.l_uav_location)
            init_phi = self.systemmodel.Phi(f_uav_location=self.f_uav_location, l_uav_location=self.l_uav_location, d=init_distance)
            # 创建初始观察值state, 包括底层无人机的位置(f_uav_num, 3)、顶层无人机的位置(3)、飞行时间
            self.state = np.hstack((self.f_uav_location.reshape(self.f_uav_num*3, ),init_phi.reshape(self.f_uav_num*3, ), self.l_uav_location, 
                                    # self.time_total, self.local_accuracy_sum, 
                                    ))
            
        else:
            self.state = np.hstack((self.f_uav_location.reshape(self.f_uav_num*3, ), self.l_uav_location, 
                                    # self.time_total, self.local_accuracy_sum, 
                                    ))
        state = deepcopy(self.state).reshape(self.state.size)
        if self.is_norm_man:
            for i in range(3*self.f_uav_num):
                if i%3 == 2:
                    state[i] /= self.f_uav_H
                elif i%3 == 1:
                    state[i] /= self.y_range[1]
                else:
                    state[i] == state[i]/self.x_range[1]
            state[-1] /= self.l_uav_H
            state[-2] /= self.y_range[1]
            state[-3] /= self.x_range[1]
        return state
    def step(self, step_num, action):
        
        state = self.state
        time_total = self.time_total
        if self.is_beam:
            f_uav_location = state[:self.f_uav_num*3]
            f_uav_location = f_uav_location.reshape((self.f_uav_num, 3))
            phi = state[self.f_uav_num*3:self.f_uav_num*6]
            phi = phi.reshape((self.f_uav_num, 3))
            l_uav_location = state[self.f_uav_num*6:self.f_uav_num*6 + 3]
            
            action = action
            action[0] = (self.l_v_max-self.l_v_min)/2*action[0] + (self.l_v_max + self.l_v_min)/2
            action[1] = math.pi/2*action[1]
            action[2] = math.ceil((self.local_epochs_max-self.local_epochs_min)/2*action[2] + (self.local_epochs_max + self.local_epochs_min)/2)
            action[5] = (self.alpha_z_max-self.alpha_z_min)/2*action[5] + (self.alpha_z_max + self.alpha_z_min)/2 
            alpha = action[3:6]
        else:
            f_uav_location = state[:self.f_uav_num*3]
            f_uav_location = f_uav_location.reshape((self.f_uav_num, 3))
            l_uav_location = state[self.f_uav_num*3:self.f_uav_num*3 + 3]
            
            action = action
            action[0] = (self.l_v_max-self.l_v_min)/2*action[0] + (self.l_v_max + self.l_v_min)/2
            action[1] = math.pi/2*action[1]
            action[2] = math.ceil((self.local_epochs_max-self.local_epochs_min)/2*action[2] + (self.local_epochs_max + self.local_epochs_min)/2)
        ########## fl
        time_begin = time.time()
        local_epochs = int(action[2])
        self.step_num += 1

        global_epoch_dic = {}
        diff_acc = 0
        diff_loss = 0
        avg_local_loss = 0

        global_epoch_dic, acc, diff_acc, diff_loss, avg_local_loss = self.fl.iteration(step_num, \
                                                        local_epochs, candidate_index=self.conf["config_train"].CONDIDATE)
        
        self.df_list.append(global_epoch_dic)
        self.fl_time = time.time() - time_begin
        #########  fl
        if self.is_beam:
            distance = self.systemmodel.Distance(f_uav_location, l_uav_location) 
            gain, index = self.systemmodel.Gain(phi, alpha, self.num_episode)
            if len(index) == 0:
                print(action)
            t_down_ = self.systemmodel.t_down(index, distance, gain)
            t_up_ = self.systemmodel.ofdma_t_up(index, distance, gain)
            t_comp = self.systemmodel.t_comp(index, action[2])
            t_agg = self.systemmodel.t_agg(self.t_uav_f)
        else:
            index = np.arange(self.l_uav_num)
            distance = self.systemmodel.Distance(f_uav_location, l_uav_location)
            t_down_ = self.systemmodel.t_down(index, distance, )
            t_up_ = self.systemmodel.ofdma_t_up(index, distance, )
            t_comp = self.systemmodel.t_comp(index, action[2])
            t_agg = self.systemmodel.t_agg(self.t_uav_f)
        uav_num = len(index)
        #此处无人机飞行时间为当前状态下，底层无人机计算时间(与本次决策相关)和上下行传输时间(只与当前位置有关)
        fly_time = np.max(t_comp + t_down_ + t_up_) #+ t_agg
        next_time_total = time_total + fly_time
        self.time_total = deepcopy(next_time_total)

        #根据action，环境进行下一步, 底层无人机速度和方向是随机的  
        f_uav_v = np.random.uniform(self.v_min, self.v_max, size = (self.f_uav_num, 1))
        f_uav_theta = np.random.uniform(self.theta_min, self.theta_max, size = (self.f_uav_num, 1))

        #底层无人机新坐标
        # next_f_uav_location = []
        for i in range(self.f_uav_num):
            for j in range(2):
                if j == 0:
                    f_uav_location[i][0] += fly_time*f_uav_v[i]*math.cos(f_uav_theta[i])
                    f_uav_location[i][0] = max(min(
                            f_uav_location[i][0], self.x_range[1]), self.x_range[0])
                else:
                    f_uav_location[i][1] += fly_time*f_uav_v[i]*math.sin(f_uav_theta[i])
                    f_uav_location[i][1] = max(min(
                        f_uav_location[i][1], self.y_range[1]), self.y_range[0])
            
        next_f_uav_location = np.array(f_uav_location)#.reshape(self.f_uav_num, 3)

        #顶层无人机新坐标
        l_uav_location[0] = l_uav_location[0] + fly_time*action[0]*math.cos(action[1])
        l_uav_location[1] = l_uav_location[1] + fly_time*action[0]*math.sin(action[1]) 

        l_uav_location[0] = max(min(
            l_uav_location[0], self.x_range[1]), self.x_range[0])
        l_uav_location[1] = max(min(
            l_uav_location[1], self.y_range[1]), self.y_range[0])
        
        next_l_uav_location = np.array(l_uav_location)
        self.l_uav_location = deepcopy(next_l_uav_location)

        # is terminal
        done = 1 if self.step_num > self.conf["global_epochs"] - 1 else 0  #30
        done = np.array(done)
        
        #  next state
        if self.is_beam:
            next_distance = self.systemmodel.Distance(f_uav_location=next_f_uav_location, l_uav_location=next_l_uav_location)
            next_phi = self.systemmodel.Phi(f_uav_location=next_f_uav_location, l_uav_location=next_l_uav_location, d=next_distance)
            next_state = np.hstack((next_f_uav_location.reshape(self.f_uav_num*3, ), next_phi.reshape(self.f_uav_num*3, ), next_l_uav_location.reshape(3, ),
                                    #  self.time_total, self.local_accuracy_sum,
                                    ))
        else:
            next_state = np.hstack((next_f_uav_location.reshape(self.f_uav_num*3, ), next_l_uav_location.reshape(3, ),
                                    
                                    ))
        self.state = next_state.astype(float)
        #奖励函数求解
    
        # reward1 = np.min(gain)
        reward_settle = 0
        energy_consum = np.array(( fly_time) * self.systemmodel.p_fly(action[0])) 
        
        acc_increase = diff_acc
        acc_increase = 0
        loss_decrease = -diff_loss 
        

        reward5 = - avg_local_loss
        

        #结算奖励
        ##已设置为0
        if done :
            reward_settle += self.end_reward
            self.num_episode += 1
            if self.time_total > self.time_max:
                reward_settle -= self.time_penalty

        l = deepcopy(self.l_uav_location)
        f = deepcopy(next_f_uav_location)
        d = deepcopy(np.max(distance))
        
        reward = reward_settle + (- energy_consum*0.001) + acc_increase + loss_decrease*5 + reward5

        # print('energy:', energy_consum, 'loss_decrease:', loss_decrease )
        # print('acc_increase', acc_increase)

        ##
        state = deepcopy(self.state)
        if self.is_norm_man:
            for i in range(3*self.f_uav_num):
                if i%3 == 2:
                    state[i] /= self.f_uav_H
                elif i%3 == 1:
                    state[i] /= self.y_range[1]
                else:
                    state[i] == state[i]/self.x_range[1]
            state[-1] /= self.l_uav_H
            state[-2] /= self.y_range[1]
            state[-3] /= self.x_range[1]

        return state, reward, energy_consum, acc_increase, loss_decrease, done, uav_num, \
            l, f, d, np.max(t_up_ + t_down_), np.max(t_comp + t_up_ + t_down_)
    
    def get_dflist(self, ):
        return self.df_list
    def get_fl_time(self, ):
        return self.fl_time




'''the following below is old version, deprecated'''
#env1, state space based on distance, with DANE FL algorithm
class Environment1(Environment):

    def __init__(self,):
        super().__init__()
        self.systemmodel = SystemModel1(f_uav_num=self.f_uav_num)
        
    # 定义每一个episode的初始状态，实现随机初始化
    def reset(self):
        

        # self.distance = np.zeros((self.f_uav_num,1))
        # self.direction = np.zeros((self.f_uav_num,1))
        self.distance = 20*np.random.random(size = (self.f_uav_num,1))
        self.direction = 2*math.pi*np.random.random(size = (self.f_uav_num,1))
        self.distance_direction = np.concatenate((self.distance,self.direction),axis = 1)#使f_uav的距离和方向相邻

        self.local_accuracy_sum = np.zeros(1)
        self.time_total = np.zeros(1)
    
        self.l_uav_location = np.zeros(2)
        self.state = np.hstack((self.distance_direction.reshape(self.f_uav_num*2,),self.local_accuracy_sum,self.time_total))


        return self.state

    def step(self, action):
        
        
        state = self.state
        self.action = action
        l_uav_location = deepcopy(self.l_uav_location)
        time_total = self.time_total
        
        distance_direction = np.array(state[:self.f_uav_num*2]).reshape(self.f_uav_num,2)
        distance = distance_direction[:,0]
        direction = distance_direction[:,1]

        local_accuracy_sum = state[self.f_uav_num*2:self.f_uav_num*2+1]
       
        
        self.action[0] = (self.l_v_max-self.l_v_min)/2*self.action[0]+(self.l_v_max+self.l_v_min)/2
        # self.action[1] = (self.theta_max-self.theta_min)/2*self.action[1]+(self.theta_max+self.theta_min)/2
        self.action[1] = math.pi/2*self.action[1]
        self.action[2] = math.ceil((self.iteration_max-self.iteration_min)/2*self.action[2]+(self.iteration_max+self.iteration_min)/2)
        
        # local_iteration = math.ceil(2/((2-self.L*self.lamda)*self.lamda*self.gamma)*math.log(1/action[2]))
        t_down_ = self.systemmodel.t_down(distance)
        t_up_ = self.systemmodel.ofdma_t_up(distance)
        t_comp = self.systemmodel.t_comp(self.action[2])
        t_agg = self.systemmodel.t_agg(self.t_uav_f)
        #此处无人机飞行时间为当前状态下，底层无人机计算时间(与本次决策相关)和上下行传输时间(只与当前位置有关)
        fly_time = np.max(t_comp+t_down_+t_up_)#+t_agg
        next_time_total = time_total+fly_time
        self.time_total = next_time_total

        #环境进行下一步
        #无范围限制，随机方向和速度
        '''
        #底层无人机当前随机的速度和方向
        f_uav_v = np.random.uniform(self.v_min,self.v_max,size = (self.f_uav_num,1))
        f_uav_theta = np.random.uniform(self.theta_min,self.theta_max,size = (self.f_uav_num,1)) 

        # f_uav_theta_norm = np.clip(np.random.normal(size = (self.f_uav_num,1)),-1,1)      
        # f_uav_theta = f_uav_theta_norm*self.theta_max
        
        #底层无人机下一轮绝对坐标
        next_f_uav_location = []
        for i in range(self.f_uav_num):
            for j in range(2):
                if j == 0:
                    next_f_uav_location_x = l_uav_location[0]+distance[i]*math.cos(direction[i])+fly_time*f_uav_v[i]*math.cos(f_uav_theta[i])
                    next_f_uav_location.append(next_f_uav_location_x)
                else:
                    next_f_uav_location_y = l_uav_location[1]+distance[i]*math.sin(direction[i])+fly_time*f_uav_v[i]*math.sin(f_uav_theta[i])
                    next_f_uav_location.append(next_f_uav_location_y)               
        
        next_f_uav_location = np.array(next_f_uav_location).reshape(self.f_uav_num, 2)'''

                
        #限制范围
        f_uav_v = np.random.uniform(self.v_min,self.v_max,size = (self.f_uav_num,1))
        # f_uav_theta = np.random.uniform(self.theta_min,self.theta_max,size = (self.f_uav_num,1))
    
        f_uav_theta = []
        #x，y方向反向.
        # # for i in range(self.f_uav_num):
        #     if self.direction_flag_x[i]:
        #         f_uav_theta_i = np.random.uniform(self.theta_min,self.theta_max)
        #         if self.direction_flag_y[i]:
        #             f_uav_theta_i = -f_uav_theta_i if f_uav_theta_i>0 else f_uav_theta_i
        #         else:
        #             f_uav_theta_i = -f_uav_theta_i if f_uav_theta_i<0 else f_uav_theta_i
        #         f_uav_theta.append(f_uav_theta_i)
        #     else:
        #         f_uav_theta_i = np.random.uniform(self.theta_min,self.theta_max)+math.pi
        #         if self.direction_flag_y[i]:
        #             f_uav_theta_i = -f_uav_theta_i if f_uav_theta_i>0 else f_uav_theta_i
        #         else:
        #             f_uav_theta_i = -f_uav_theta_i if f_uav_theta_i<0 else f_uav_theta_i
        #         f_uav_theta.append(f_uav_theta_i)
        
        #x反向，y不变      
        for i in range(self.f_uav_num):
            if self.direction_flag_x[i]:
                f_uav_theta_i = np.random.uniform(self.theta_min,self.theta_max)
                f_uav_theta.append(f_uav_theta_i)
            else:
                f_uav_theta_i = np.random.uniform(self.theta_min,self.theta_max)+math.pi
                f_uav_theta.append(f_uav_theta_i)
             
        f_uav_theta = np.array(f_uav_theta)
        

        #底层无人机下一轮绝对坐标
        next_f_uav_location = []
        for i in range(self.f_uav_num):
            for j in range(2):
                if j == 0:
                    next_f_uav_location_x = l_uav_location[0]+distance[i]*math.cos(direction[i])+fly_time*f_uav_v[i]*math.cos(f_uav_theta[i])
                    # if next_f_uav_location_x>self.x_range[1]:
                    #     self.direction_flag_x[i] = 0
                    #     next_f_uav_location_x = self.x_range[1]
                    #     # next_f_uav_location_x = max(min(
                    #     #     l_uav_location[0]+distance[i]*math.cos(direction[i])+fly_time*f_uav_v[i]*math.cos(f_uav_theta[i]),self.x_range[1]),self.x_range[0])
                    # elif next_f_uav_location_x<self.x_range[0]:
                    #     self.direction_flag_x[i] = 1
                    #     next_f_uav_location_x = self.x_range[0]
                    # else:
                    #     pass
                    next_f_uav_location_x = max(min(
                            next_f_uav_location_x,self.x_range[1]),self.x_range[0])
                    next_f_uav_location.append(next_f_uav_location_x)
                else:
                    next_f_uav_location_y = l_uav_location[1]+distance[i]*math.sin(direction[i])+fly_time*f_uav_v[i]*math.sin(f_uav_theta[i])
                    # if next_f_uav_location_y>self.y_range[1]:
                    #     self.direction_flag_y[i] = 1
                    #     next_f_uav_location_y = self.y_range[1]                      
                    # elif next_f_uav_location_y<self.y_range[0]:
                    #     self.direction_flag_y[i] = 0
                    #     next_f_uav_location_y = self.y_range[0]
                    # else:
                    #     pass
        
                    next_f_uav_location_y = max(min(
                        next_f_uav_location_y,self.y_range[1]),self.y_range[0])
                    next_f_uav_location.append(next_f_uav_location_y)

        next_f_uav_location = np.array(next_f_uav_location).reshape(self.f_uav_num, 2)
        


        #顶层无人机新坐标

        l_uav_location[0] = l_uav_location[0]+self.action[0]*math.cos(self.action[1])*fly_time
        l_uav_location[1] = l_uav_location[1]+self.action[0]*math.sin(self.action[1])*fly_time
        
        l_uav_location[0] = max(min(
            l_uav_location[0],self.x_range[1]),self.x_range[0])
        l_uav_location[1] = max(min(
            l_uav_location[1],self.y_range[1]),self.y_range[0])
        
        self.l_uav_location = deepcopy(l_uav_location)

        next_distance = []
        next_direction = []
        next_true_distance = []

        #当前相对水平距离和方位角
        for i in range(self.f_uav_num):
            d_ = math.sqrt((next_f_uav_location[i][0]-l_uav_location[0])**2+(next_f_uav_location[i][1]-l_uav_location[1])**2)

            di_ = math.atan2((next_f_uav_location[i][1]-l_uav_location[1]),(next_f_uav_location[i][0]-l_uav_location[0]))

            next_true_distance.append(math.sqrt(d_**2+(self.f_uav_H-self.l_uav_H )**2))
            next_distance.append(d_)
            next_direction.append(di_)

        next_true_distance = np.array(next_true_distance).reshape(self.f_uav_num,1)
        next_distance = np.array(next_distance).reshape(self.f_uav_num,1)
        next_direction = np.array(next_direction).reshape(self.f_uav_num,1)
        
        next_distance_direction = np.concatenate((next_distance,next_direction),axis = 1)
        #当前距离
      
      
        #局部精度公式求和
        yita = math.exp(-self.action[2]*(2-self.L*self.lamda)*self.lamda*self.gamma/2)
        local_accuracy = math.log(1-((1-yita)*self.gamma**2*self.ksi)/(2*self.L**2))
        next_local_accuracy_sum = local_accuracy_sum+local_accuracy
      
        
        #判断下一个观察状态下，本次episode是否结束

        done = 1 if next_local_accuracy_sum<math.log(self.epsilon) else 0
        done = np.array(done)

        #环境状态改变,下一个state
        next_state = np.hstack((next_distance_direction.reshape(self.f_uav_num*2,),next_local_accuracy_sum,self.time_total))

        self.state = next_state
        #奖励函数求解
        
        #日常奖励
        reward1 = 0
        reward2 = np.array(( -fly_time) * self.systemmodel.p_fly(action[0])/ 1000)

        # if l_uav_location[0] == self.x_range[1] or l_uav_location[0] == self.x_range[0]:
        #     reward-= self.penalty
        # if l_uav_location[1] == self.y_range[1] or l_uav_location[1] == self.y_range[0]:
        #     reward-= self.penalty

        # if np.sum(next_true_distance>self.distance_max):
        #     reward-= self.distance_penalty
            
        # if np.sum(next_true_distance>self.distance_max):
        #     self.outofdistance += 1

        #结算奖励
        # if done :
        #     reward2 += self.end_reward
        #     if self.time_total>self.time_max:
        #         reward2-= self.time_penalty

            # if self.outofdistance:
            #     reward-= 1
            #     self.outofdistance = 0

        l = deepcopy(self.l_uav_location)
        f = deepcopy(next_f_uav_location)
        d = deepcopy(np.max(next_true_distance))
        reward = reward1+reward2
        
        return self.state, reward, done,np.max(t_up_+t_down_),np.max(t_comp+t_up_+t_down_),l, f,d,


#env0, state space based on location, with DANE FL algorithm
class Environment0(Environment):

    def __init__(self,):
        super().__init__()
        self.systemmodel = SystemModel0()
        
    # 定义每一个episode的初始状态，实现随机初始化
    def reset(self):
        
        #随机初始化底层无人机位置，范围{0，1000}
        #self.f_uav_location = np.random.uniform(0, 1000, size = (self.f_uav_num, 2))
        self.f_uav_location = np.zeros((self.f_uav_num, 2))

        #随机初始化顶层无人机位置
        self.l_uav_location = np.zeros(2)

        # 创建初始观察值state, 包括底层无人机的位置(f_uav_num, 2)、顶层无人机的位置(f_uav_num, 2)、飞行时间

        self.local_accuracy_sum = np.zeros(1)
        self.time_total = 0
        self.state = np.hstack((self.f_uav_location.reshape(self.f_uav_num*2, ), self.l_uav_location, self.time_total, self.local_accuracy_sum))

        return self.state

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        state = self.state
        
        f_uav_location = state[:self.f_uav_num*2]
        f_uav_location = f_uav_location.reshape((self.f_uav_num, 2))
        
        l_uav_location = state[self.f_uav_num*2:self.f_uav_num*2 + 2]
        
     
        time_total = state[self.f_uav_num*2 + 2:self.f_uav_num*2 + 3]
        local_accuracy_sum = state[self.f_uav_num*2 + 3:self.f_uav_num*2 + 4]
        
      
        action = action
        action[0] = (self.l_v_max-self.l_v_min)/2*action[0] + (self.l_v_max + self.l_v_min)/2
        action[1] = math.pi/2*action[1]
        action[2] = math.ceil((self.iteration_max-self.iteration_min)/2*action[2] + (self.iteration_max + self.iteration_min)/2)
        # action[3:6] = action[3:6]
        distance = self.systemmodel.Distance(f_uav_location, l_uav_location) 

        t_down_=self.systemmodel.t_down(f_uav_location, l_uav_location)
        # t_up_=self.systemmodel.t_up(f_uav_location, l_uav_location)
        t_up_=self.systemmodel.ofdma_t_up(f_uav_location, l_uav_location)
        t_comp=self.systemmodel.t_comp(action[2])
        
        #此处无人机飞行时间为当前状态下，底层无人机计算时间(与本次决策相关)和上下行传输时间(只与当前位置有关)
        fly_time = np.max(t_comp + t_down_ + t_up_) #+ t_agg
        next_time_total = time_total + fly_time
        self.time_total = next_time_total


        #根据action，环境进行下一步, 

        
        #底层无人机速度和方向是随机的
        f_uav_v = np.random.uniform(self.v_min, self.v_max, size = (self.f_uav_num, 1))
        f_uav_theta = np.random.uniform(self.theta_min, self.theta_max, size = (self.f_uav_num, 1))

        #底层无人机新坐标
        next_f_uav_location = []
        for i in range(self.f_uav_num):
            for j in range(2):
                if j == 0:
                    next_f_uav_location_x = f_uav_location[i][0] + fly_time*f_uav_v[i]*math.cos(f_uav_theta[i])
                    next_f_uav_location_x = max(min(
                            next_f_uav_location_x, self.x_range[1]), self.x_range[0])
                    next_f_uav_location.append(next_f_uav_location_x)

                else:
                    next_f_uav_location_y = f_uav_location[i][1] + fly_time*f_uav_v[i]*math.sin(f_uav_theta[i])
                    next_f_uav_location_y = max(min(
                        next_f_uav_location_y, self.y_range[1]), self.y_range[0])
                    next_f_uav_location.append(next_f_uav_location_y)
        
        next_f_uav_location = np.array(next_f_uav_location).reshape(self.f_uav_num, 2)

       
        #数据量不变
        # next_f_uav_data = f_uav_data    # + np.random.uniform(20, 50, size = (self.f_uav_num, 1))

        #顶层无人机新坐标
     
        l_uav_location[0] = l_uav_location[0] + fly_time*action[0]*math.cos(action[1])
        l_uav_location[1] = l_uav_location[1] + fly_time*action[0]*math.sin(action[1]) 

        l_uav_location[0] = max(min(
            l_uav_location[0], self.x_range[1]), self.x_range[0])
        l_uav_location[1] = max(min(
            l_uav_location[1], self.y_range[1]), self.y_range[0])
        
        self.l_uav_location = deepcopy(l_uav_location)

      
        yita = math.exp(-action[2]*(2-self.L*self.lamda)*self.lamda*self.gamma/2)
        local_accuracy = math.log(1-((1-yita)*self.gamma**2*self.ksi)/(2*self.L**2))
        next_local_accuracy_sum = local_accuracy_sum + local_accuracy
        
        #判断该观察状态下，本次episode是否结束

        done = 1 if next_local_accuracy_sum < math.log(self.epsilon) else 0
        done = np.array(done)
        

        #环境状态改变, 下一个state
        next_state = np.hstack((next_f_uav_location.reshape(self.f_uav_num*2, ), l_uav_location.reshape(2, ), self.time_total, 
                              next_local_accuracy_sum))
        
       
        self.state = next_state.astype(float)
        #奖励函数求解
    
        reward1 = 0
        reward2 = np.array(( -fly_time) * self.systemmodel.p_fly(action[0])/ 1000) 
      

        #结算奖励
        if done :
            reward2 += self.end_reward
            if self.time_total > self.time_max:
                reward2 -= self.time_penalty
        
        reward = reward1 + reward2

        l = deepcopy(self.l_uav_location)
        f = deepcopy(next_f_uav_location)
        d = deepcopy(np.max(distance))
       
        return self.state, reward, done, np.max(t_up_ + t_down_), np.max(t_comp + t_up_ + t_down_), l, f, d

