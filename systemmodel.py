import math
import numpy as np
import matplotlib.pyplot as plt

# np.seterr(divide='ignore', invalid='ignore')
# ======================resnet18
# Total params: 23,379,024
# Trainable params: 23,379,024
# Non-trainable params: 0
# Params size (MB): 93.52
# ======================resnet18

# =========================res50
# Total params: 51,114,064
# Trainable params: 51,114,064
# Non-trainable params: 0
# Params size (MB): 204.46
# =========================res50

class SystemModel(object):
    def __init__(self, datasize, f_uav_num = 5, is_beam = True, m_num =20):
        self.is_beam = is_beam
        self.f_uav_num = f_uav_num              # 底层无人机数，K
        self.f_uav_H = 140               # 无人机的飞行高度
        self.l_uav_H = 150

        #通信参数
        self.rou_0 = 1.42*10**-4  # 1m参考距离的信道增益
        # self.rou_0 = 1*10**-5
        self.alpha = 2                    #自由空间损耗
        self.p_L = 1                     #顶层无人机发射功率
        self.p_i  = 1                # 底层无人机的发射功率
        self.G_iL = 1                     #天线增益
        self.G_Li = 1
        self.B = 10 ** 6               # 广播带宽
        self.N_0 = 10**-20.4           # 噪声功率谱密度
        # self.N_0 = 10**-16
        self.N_B = 10**-9

        self.B_up = 1*10**6  #上行总带宽
        self.B_il = self.B/self.f_uav_num     #FDMA
        # self.subbandwidth = self.B_up/self.M  #OFDMA
        self.subbandwidth = 5*10**4  #OFDMA
        self.M = m_num 

        self.delta = 0.7
        self.A_x = 3
        self.A_y = 3
        self.A_z = 3
       
        self.G_iL = self.A_x*self.A_y*self.A_z                     #天线增益
        self.G_Li = np.zeros(shape = (self.f_uav_num,1))

        self.S_w = 28*1024  #下行传输数据量
        self.S_w_ = 28*1024  #上行传输数据量
        self.L  = 10000

        self.S_w = 2*1024*1024  #下行传输数据量
        self.S_w_ = 2*1024*1024  #上行传输数据量
        self.L  = 1000000   #10000               # 训练1sample需要的CPU计算周期数
        
        # self.S_w = 200*1024  #下行传输数据量
        # self.S_w_ = 200*1024  #上行传输数据量
        # self.L  = 100000 
        
         
        self.f_uav_f = 1 * (10 ** 9)      # 无人机的计算频率
        
        #计算参数
        # self.f_uav_data = np.random.randint(800,1000,size = (self.f_uav_num,1))
        self.f_uav_data = np.array(datasize).reshape(self.f_uav_num, 1)
        
        self.C_T = 100
        
        #self.k = 10 ** -28              # 无人机CPU的电容系数
        
        self.p0 = 80
        self.pi = 89
        self.U_tip = 120
        self.v0 = 4.03
        self.zeta = 0.6
        self.s = 0.05
        self.rou = 1.225
        self.A = 0.5

       #model based on location
    def Distance(self, f_uav_location, l_uav_location):
        self.f_uav_location = f_uav_location
        self.l_uav_location = l_uav_location
        D = []

        for i in range(self.f_uav_num):
            d_il = math.sqrt((self.f_uav_location[i][0] - self.l_uav_location[0]) ** 2 
                + (self.f_uav_location[i][1] - self.l_uav_location[1])**2+(self.f_uav_location[i][2] - self.l_uav_location[2]) ** 2 )
            D.append(d_il)
        D = np.array(D).reshape(self.f_uav_num, 1)
        return D

    #def Gain(self, f_uav_location, l_uav_location, alpha, d, num_episode):
        alpha = alpha
        f_uav_location = f_uav_location
        l_uav_location = l_uav_location
        num_episode = num_episode
        d = d
        G = []
        index = []
        
        for i in range(self.f_uav_num):
            phi_x = (l_uav_location[0]-f_uav_location[i][0] )/d[i]+alpha[0]
            phi_y = (l_uav_location[1]-f_uav_location[i][1])/d[i]+alpha[1]
            phi_z = (self.l_uav_H-self.f_uav_H)/d[i]+alpha[2]
#----------------------------------------------------------------------------------------------
            # g_x = (math.sin(self.A_x*math.pi/2*phi_x)/(self.A_x*math.sin(math.pi/2*phi_x)))**2
            # g_y = (math.sin(self.A_y*math.pi/2*phi_y)/(self.A_y*math.sin(math.pi/2*phi_y)))**2
            # g_z = (math.sin(self.A_z*math.pi/2*phi_z)/(self.A_z*math.sin(math.pi/2*phi_z)))**2

            # if phi_x < -2*self.delta/self.A_x  or  2*self.delta/self.A_x < phi_x :
            #     g_x = 1/(self.A_x*self.A_y*self.A_z)
            # if phi_y < -2*self.delta/self.A_y  or  2*self.delta/self.A_y < phi_y :
            #     g_y = 1/(self.A_x*self.A_y*self.A_z)
            # if phi_z < -2*self.delta/self.A_z  or  2*self.delta/self.A_z < phi_z :
            #     g_z = 1/(self.A_x*self.A_y*self.A_z)
           
            # g = self.A_x*self.A_y*self.A_z*g_x*g_y*g_z

#--------------------------------------------------------------------------------
            phi_x = phi_x*math.pi/2
            phi_y = phi_y*math.pi/2
            phi_z = phi_z*math.pi/2
            if np.abs(phi_x) < self.delta*np.pi/self.A_x:
                g_x = np.power(np.cos(self.A_x*phi_x/2),2)
            else :
                g_x = np.power(np.cos(self.delta*np.pi/2),2)/(1+(np.power(phi_x,2)-\
                    np.power(self.delta*np.pi/self.A_x,2))*np.tan(self.delta*np.pi/2)*np.power(self.A_x,2)/(2*self.delta*np.pi))
            if np.abs(phi_y) < self.delta*np.pi/self.A_y:
                g_y = np.power(np.cos(self.A_y*phi_y/2),2)
            else:
                g_y = np.power(np.cos(self.delta*np.pi/2),2)/(1+(np.power(phi_y,2)-\
                    np.power(self.delta*np.pi/self.A_y,2))*np.tan(self.delta*np.pi/2)*np.power(self.A_y,2)/(2*self.delta*np.pi))
            if np.abs(phi_z) < self.delta*np.pi/self.A_z:
                g_z = np.power(np.cos(self.A_z*phi_z/2),2)
            else:
                g_z = np.power(np.cos(self.delta*np.pi/2),2)/(1+(np.power(phi_z,2)-\
                    np.power(self.delta*np.pi/self.A_z,2))*np.tan(self.delta*np.pi/2)*np.power(self.A_z,2)/(2*self.delta*np.pi))
            
            g = self.A_x*self.A_y*self.A_z*g_x*g_y*g_z
#-----------------------------------------------------------------------------------
          
      
            # if g  > 0.01*(num_episode - 200):    
            #     index.append(i)
            
            if g  > 0.01:    
                index.append(i)               
            G.append(g)


        num_seletced = len(index)
        G = np.array(G).reshape(self.f_uav_num, 1)
        index = np.array(index).reshape(num_seletced, 1)

        return G, index, 

    def ofdma_t_up(self, index, D, G = None):
        if self.is_beam:
            assert G.any() != None
        channel_SNR_up = []
        comm_rate_up = []
        t_up = []
        D = D
        G = G
        index = index
        # episode_num = len(index)
        # SNR
        for i in range(self.f_uav_num):
            if self.is_beam:
                SNR_up = self.p_i*self.rou_0*G[i]*self.G_iL / (self.N_B*pow(D[i],self.alpha))
            else:
                SNR_up = self.p_i*self.rou_0 / (self.N_B*pow(D[i],self.alpha))
            #(self.subbandwidth*self.N_0*pow(d_il,self.alpha))
            channel_SNR_up.append(SNR_up)
        channel_SNR_up = np.array(channel_SNR_up).reshape(self.f_uav_num, 1)

        # rate
        for i in range(self.f_uav_num):
            rate_up = self.M/self.f_uav_num*self.subbandwidth * math.log2(1 +channel_SNR_up[i])   #标量
            comm_rate_up.append(rate_up)
        comm_rate_up = np.array(comm_rate_up).reshape(self.f_uav_num, 1)

        # latency
        for i in range(self.f_uav_num):
            t_up_ = self.S_w_ / (comm_rate_up[i]+0.00001)
            t_up.append(t_up_)

        t_up = np.array(t_up).reshape(self.f_uav_num, 1)
        #for selection
        time_up = []
        for i in index:
            time_up.append(t_up[i])

        return time_up
    
    def t_up(self, f_uav_location,  l_uav_location):
        
        self.f_uav_location = f_uav_location
        self.l_uav_location = l_uav_location


        channel_SNR_up = []
        comm_rate_up = []
        t_up = []
        
        

        # 通信模型,上行信道
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_il = math.sqrt((self.f_uav_location[i][0] - self.l_uav_location[0]) ** 2 \
                + (self.f_uav_location[i][1] - self.l_uav_location[1])**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_i*self.rou_0*self.G_iL*self.G_iL / (self.B_il*self.N_0*pow(d_il,self.alpha))
            channel_SNR_up.append(SNR_up)

        channel_SNR_up = np.array(channel_SNR_up).reshape(self.f_uav_num, 1)

        # 计算每个底层无人机与顶层无人机之间数据的传输速率
        for i in range(self.f_uav_num):
            rate_up = self.B_il * math.log2(1 +channel_SNR_up[i])   #标量
            comm_rate_up.append(rate_up)

        comm_rate_up = np.array(comm_rate_up).reshape(self.f_uav_num, 1)


        # 计算底层无人机和顶层无人机的通信时延
        for i in range(self.f_uav_num):
            t_up_ = self.S_w_ / comm_rate_up[i]
            t_up.append(t_up_)

        t_up = np.array(t_up).reshape(self.f_uav_num, 1)

        return t_up

    def t_down(self, index, D, G = None):
        if self.is_beam:
            assert G.any() != None
        channel_SNR_down = []
        comm_rate_down = []
        t_down = []
        D = D
        G = G
        episode_num = len(index)

        # SNR
        for i in range(self.f_uav_num):
            if self.is_beam:
                SNR_down = self.p_L*self.rou_0*G[i]*self.G_iL / (self.N_B*pow(D[i],self.alpha))
            else:
                SNR_down = self.p_L*self.rou_0 / (self.N_B*pow(D[i],self.alpha))
            channel_SNR_down.append(SNR_down)
        channel_SNR_down = np.array(channel_SNR_down).reshape(self.f_uav_num, 1)

        # rate
        for i in range(self.f_uav_num):
            rate_down = self.B * math.log2(1 +channel_SNR_down[i])  
            comm_rate_down.append(rate_down)

        comm_rate_down = np.array(comm_rate_down).reshape(self.f_uav_num, 1)


        # t_down
        for i in range(self.f_uav_num):
            t_down_ = self.S_w / (comm_rate_down[i]+0.00001)
            t_down.append(t_down_)

        t_down = np.array(t_down).reshape(self.f_uav_num, 1)
        # for selection
        time_down = []
        for i in index:
            time_down.append(t_down[i])

        return time_down

    def t_comp(self, index, I):

        t_comp = []
        self.I = I
        index = index
        episode_num = len(index)
        #计算时延
        for i in index :
            t_comp_ = self.I*self.L*self.f_uav_data[i] / self.f_uav_f

            t_comp.append(t_comp_)
        t_comp = np.array(t_comp).reshape(episode_num, 1)
        return t_comp

    def t_agg(self,F):
        t_uav_f = F
        t_agg = self.f_uav_num*self.S_w*self.C_T/t_uav_f
        return t_agg

    def p_fly_(self,v):
        P = self.p0*(1+3*v**2/(self.U_tip**2))+self.pi*self.v0/v+0.5*self.zeta*self.s*self.rou*self.A*v**3
        return P
    
    def p_fly(self,v):
        P = self.p0*(1+3*v**2/(self.U_tip**2)) +  0.5*self.zeta*self.s*self.rou*self.A*v**3 
        + self.pi*math.pow(math.pow(1 + math.pow(v,4)/(4*math.pow(self.v0,4)), 0.5) - math.pow(v,2)/(2*math.pow(self.v0, 2)), 0.5)
                                                                 
        return P

##with beamforming
class SystemModel2(SystemModel):
    def __init__(self, datasize, ula_num = 3, f_uav_num = 5, is_beam = True, m_num =20):
        super().__init__(datasize, f_uav_num = f_uav_num, is_beam=is_beam, m_num= m_num)
        self.A_x = ula_num
        self.A_y = ula_num
        self.A_z = ula_num

    def Distance(self, f_uav_location, l_uav_location):
        return super().Distance(f_uav_location, l_uav_location)

    def Phi(self,  f_uav_location, l_uav_location, d, ):
        
        self.f_uav_location = f_uav_location
        self.l_uav_location = l_uav_location
        
        d = d
        G = []
        phi = []
        
        for i in range(self.f_uav_num):
            phi_x = (l_uav_location[0]-f_uav_location[i][0])/d[i]
            phi_y = (l_uav_location[1]-f_uav_location[i][1])/d[i]
            phi_z = (l_uav_location[2]-f_uav_location[i][2])/d[i]
            phi.append(phi_x)
            phi.append(phi_y)
            phi.append(phi_z)
        phi = np.array(phi)#.reshape(self.f_uav_num, 3)
        return phi

    def Gain(self, phi,  alpha, num_episode):
        num_episode = num_episode
        G = []
        index = []
        # print(phi)

        for i in range(self.f_uav_num): 
            phi_x = phi[i][0]+alpha[0]
            phi_y = phi[i][1]+alpha[1]
            phi_z = phi[i][2]+alpha[2]
#----------------------------------------------------------------------------------------------
            #to process zerodivision error  for DDPG
            g_x = (math.sin(self.A_x*math.pi/2*phi_x)/(self.A_x*math.sin(math.pi/2*phi_x)+0.00001))**2
            g_y = (math.sin(self.A_y*math.pi/2*phi_y)/(self.A_y*math.sin(math.pi/2*phi_y)+0.00001))**2
            g_z = (math.sin(self.A_z*math.pi/2*phi_z)/(self.A_z*math.sin(math.pi/2*phi_z)+0.00001))**2

            if phi_x < -2*self.delta/self.A_x  or  2*self.delta/self.A_x < phi_x :
                g_x = 1/(self.A_x*self.A_y)
            if phi_y < -2*self.delta/self.A_y  or  2*self.delta/self.A_y < phi_y :
                g_y = 1/(self.A_x*self.A_y)
            if phi_z < -2*self.delta/self.A_z  or  2*self.delta/self.A_z < phi_z :
                g_z = 1/(self.A_x*self.A_y)
#--------------------------------------------------------------
            # phi_x = phi_x*math.pi/2
            # phi_y = phi_y*math.pi/2
            # phi_z = phi_z*math.pi/2
            # if np.abs(phi_x) < self.delta*np.pi/self.A_x:
            #     g_x = np.power(np.cos(self.A_x*phi_x/2),2)
            # else :
            #     g_x = np.power(np.cos(self.delta*np.pi/2),2)/(1+(np.power(phi_x,2)-\
            #         np.power(self.delta*np.pi/self.A_x,2))*np.tan(self.delta*np.pi/2)*np.power(self.A_x,2)/(2*self.delta*np.pi))
            # if np.abs(phi_y) < self.delta*np.pi/self.A_y:
            #     g_y = np.power(np.cos(self.A_y*phi_y/2),2)
            # else:
            #     g_y = np.power(np.cos(self.delta*np.pi/2),2)/(1+(np.power(phi_y,2)-\
            #         np.power(self.delta*np.pi/self.A_y,2))*np.tan(self.delta*np.pi/2)*np.power(self.A_y,2)/(2*self.delta*np.pi))
            # if np.abs(phi_z) < self.delta*np.pi/self.A_z:
            #     g_z = np.power(np.cos(self.A_z*phi_z/2),2)
            # else:
            #     g_z = np.power(np.cos(self.delta*np.pi/2),2)/(1+(np.power(phi_z,2)-\
            #         np.power(self.delta*np.pi/self.A_z,2))*np.tan(self.delta*np.pi/2)*np.power(self.A_z,2)/(2*self.delta*np.pi))
#----------------------------------------------------------------------------------------            
            g = self.A_x*self.A_y*self.A_z*g_x*g_y*g_z
            ##3 0.037;5,0.008;7,0.003
            if g  > 0.001:    
                index.append(i)               
            G.append(g)

        if len(index) == 0:
            print(g_x, g_y, g_z)
            print(self.f_uav_location)
            print(self.l_uav_location)
            print(alpha)
            print(phi)
            
        num_seletced = len(index)
        G = np.array(G).reshape(self.f_uav_num, 1)
        index = np.array(index).reshape(num_seletced, 1)

        return G, index

    def ofdma_t_up(self, index, D, G):
        return super().ofdma_t_up(index, D, G)
    def t_down(self, index, D, G):
        return super().t_down(index, D, G)

    def t_comp(self, index, I):
        return super().t_comp(index, I)
    def t_agg(self, F):
        return super().t_agg(F)

    def p_fly_(self,v):
        return super().p_fly_(v)
    
    def p_fly(self,v):                                                        
        return super().p_fly(v)
    



'''the below is old version for DANE FL algorithm, deprecated'''
#model based on distance(horizontal)
class SystemModel1(SystemModel):
    def __init__(self,f_uav_num = 5):
        
        super().__init__()

    
    def ofdma_t_up(self, d):
        distance = d
        channel_SNR_up = []
        comm_rate_up = []
        t_up = []
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_il =  math.sqrt(distance[i]**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_i*self.rou_0 / (self.N_B*pow(d_il,self.alpha))
            # SNR_up = self.p_i*self.rou_0*self.G_iL*self.G_Li / (self.N_B*pow(d_il,self.alpha))
            channel_SNR_up.append(SNR_up)

        # channel_SNR_up =  np.array(channel_SNR_up).reshape(self.f_uav_num, 1)

        # 计算每个底层无人机与顶层无人机之间数据的传输速率
        for i in range(self.f_uav_num):
            rate_up =  self.M/self.f_uav_num*self.subbandwidth * math.log2(1 +channel_SNR_up[i])   #标量
            comm_rate_up.append(rate_up)

        # comm_rate_up =  np.array(comm_rate_up).reshape(self.f_uav_num, 1)


        # 计算底层无人机和顶层无人机的通信时延
        for i in range(self.f_uav_num):
            t_up_ = self.S_w_ / comm_rate_up[i]
            t_up.append(t_up_)

        t_up = np.array(t_up).reshape(self.f_uav_num, 1)
        return t_up

    def t_up(self, d):
         # 通信模型,上行信道
        distance = d
        channel_SNR_up = []
        comm_rate_up = []
        t_up = []
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_il =  math.sqrt(distance[i]**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_i*self.rou_0*self.G_iL*self.G_Li / (self.B_il*self.N_0*pow(d_il,self.alpha))
            channel_SNR_up.append(SNR_up)

        channel_SNR_up =  np.array(channel_SNR_up,dtype = 'float32').reshape(self.f_uav_num, 1)

        # 计算每个底层无人机与顶层无人机之间数据的传输速率
        for i in range(self.f_uav_num):
            rate_up =  self.B_il * math.log2(1 +channel_SNR_up[i])   #标量
            comm_rate_up.append(rate_up)

        comm_rate_up =  np.array(comm_rate_up,dtype = 'float32').reshape(self.f_uav_num, 1)


        # 计算底层无人机和顶层无人机的通信时延
        for i in range(self.f_uav_num):
            t_up_ = self.S_w_ / comm_rate_up[i]
            t_up.append(t_up_)

        t_up = np.array(t_up).reshape(self.f_uav_num, 1)
        return t_up
    
    
    def t_down(self, d):
         # 通信模型,上行信道
        distance = d
        channel_SNR_down = []
        comm_rate_down = []
        t_down = []
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_Li =  math.sqrt(distance[i]**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_L*self.rou_0 / (self.N_B*pow(d_Li,self.alpha))
            # SNR_up = self.p_L*self.rou_0*self.G_iL*self.G_Li / (self.N_B*pow(d_Li,self.alpha))
            channel_SNR_down.append(SNR_up)

        channel_SNR_down =  np.array(channel_SNR_down,dtype = 'float32').reshape(self.f_uav_num, 1)

        # 传输速率
        for i in range(self.f_uav_num):
            rate_down = self.B * math.log2(1 +channel_SNR_down[i])  
            comm_rate_down.append(rate_down)

        comm_rate_down =  np.array(comm_rate_down,dtype = 'float32').reshape(self.f_uav_num, 1)


        # 通信时延
        for i in range(self.f_uav_num):
            t_down_ = self.S_w / comm_rate_down[i]
            t_down.append(t_down_)

        t_down = np.array(t_down).reshape(self.f_uav_num, 1)

        return t_down


    def t_comp(self,I):

        t_comp = []
        self.I = I
        '''计算模型'''
        #计算时延
        for i in range(self.f_uav_num):
            t_comp_ = self.I*self.L*self.f_uav_data[i] / self.f_uav_f
           
            t_comp.append(t_comp_)
           

        t_comp = np.array(t_comp).reshape(self.f_uav_num, 1)
        

        return t_comp

    def t_agg(self, F):
        return super().t_agg(F)

    def p_fly_(self, v):
        return super().p_fly_(v)
    def p_fly(self, v):
        return super().p_fly(v)


# model based on location
class SystemModel0(SystemModel):

    def __init__(self, f_uav_num = 5):
        self.f_uav_num = f_uav_num
        super().__init__(f_uav_num = self.f_uav_num)
        


    def Distance(self, f_uav_location, l_uav_location):
        return super().Distance(f_uav_location, l_uav_location)
    def ofdma_t_up(self,f_uav_location,  l_uav_location):

        channel_SNR_up = []
        comm_rate_up = []
        t_up = []
        
        self.f_uav_location = f_uav_location
        self.l_uav_location = l_uav_location
        

        # 通信模型,上行信道
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_il =  math.sqrt((self.f_uav_location[i][0] - self.l_uav_location[0]) ** 2 \
                + (self.f_uav_location[i][1] - self.l_uav_location[1])**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_i*self.rou_0*self.G_iL*self.G_Li / (self.N_B*pow(d_il,self.alpha))
            #(self.subbandwidth*self.N_0*pow(d_il,self.alpha))
            channel_SNR_up.append(SNR_up)

        channel_SNR_up =  np.array(channel_SNR_up).reshape(self.f_uav_num, 1)

        # 计算每个底层无人机与顶层无人机之间数据的传输速率
        for i in range(self.f_uav_num):
            rate_up =  self.M/self.f_uav_num*self.subbandwidth * math.log2(1 +channel_SNR_up[i])   #标量
            comm_rate_up.append(rate_up)

        comm_rate_up =  np.array(comm_rate_up).reshape(self.f_uav_num, 1)


        # 计算底层无人机和顶层无人机的通信时延
        for i in range(self.f_uav_num):
            t_up_ = self.S_w_ / comm_rate_up[i]
            t_up.append(t_up_)

        t_up = np.array(t_up).reshape(self.f_uav_num, 1)

        return t_up
    
    def t_up(self, f_uav_location,  l_uav_location):
        
        channel_SNR_up = []
        comm_rate_up = []
        t_up = []
        
        self.f_uav_location = f_uav_location
        self.l_uav_location = l_uav_location
        

        # 通信模型,上行信道
        # 计算每个底层无人机与顶层无人机之间的SNR
        for i in range(self.f_uav_num):
            d_il =  math.sqrt((self.f_uav_location[i][0] - self.l_uav_location[0]) ** 2 \
                + (self.f_uav_location[i][1] - self.l_uav_location[1])**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_up = self.p_i*self.rou_0*self.G_iL*self.G_Li / (self.B_il*self.N_0*pow(d_il,self.alpha))
            channel_SNR_up.append(SNR_up)

        channel_SNR_up =  np.array(channel_SNR_up).reshape(self.f_uav_num, 1)

        # 计算每个底层无人机与顶层无人机之间数据的传输速率
        for i in range(self.f_uav_num):
            rate_up =  self.B_il * math.log2(1 +channel_SNR_up[i])   #标量
            comm_rate_up.append(rate_up)

        comm_rate_up =  np.array(comm_rate_up).reshape(self.f_uav_num, 1)


        # 计算底层无人机和顶层无人机的通信时延
        for i in range(self.f_uav_num):
            t_up_ = self.S_w_ / comm_rate_up[i]
            t_up.append(t_up_)

        t_up = np.array(t_up).reshape(self.f_uav_num, 1)

        return t_up

    def t_down(self, f_uav_location, l_uav_location):

        channel_SNR_down = []
        comm_rate_down = []
        t_down = []

        self.f_uav_location = f_uav_location
        self.l_uav_location = l_uav_location
        # 通信模型,下行信道

        # SNR
        for i in range(self.f_uav_num):
            d_il =  math.sqrt((self.f_uav_location[i][0] - self.l_uav_location[0]) ** 2\
                + (self.f_uav_location[i][1] - self.l_uav_location[1])**2+(self.f_uav_H-self.l_uav_H )**2)
            SNR_down = self.p_L*self.rou_0*self.G_iL*self.G_Li / (self.N_B*pow(d_il,self.alpha))
            channel_SNR_down.append(SNR_down)

        channel_SNR_down =  np.array(channel_SNR_down).reshape(self.f_uav_num, 1)

        # 传输速率
        for i in range(self.f_uav_num):
            rate_down = self.B * math.log2(1 +channel_SNR_down[i])   #标量
            comm_rate_down.append(rate_down)

        comm_rate_down =  np.array(comm_rate_down).reshape(self.f_uav_num, 1)


        # 通信时延
        for i in range(self.f_uav_num):
            t_down_ = self.S_w / comm_rate_down[i]
            t_down.append(t_down_)

        t_down = np.array(t_down).reshape(self.f_uav_num, 1)

        return t_down

    def t_comp(self,I):

        sys = SystemModel1()
        return sys.t_comp(I)

    def p_fly(self, v):
        return super().p_fly(v)

