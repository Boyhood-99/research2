from pickle import NONE
from turtle import color
import torch
import numpy as np
import torch.nn as nn
from buffer import ReplayBuffer
from copy import deepcopy
from torch.distributions import Normal

###there are some rl algrithms

# DDPG algrithm easily converge to the action edge, cause some 
#results wo don't want, raise zerodivision or zero value problem
class DDPG(object):
    def __init__(self, state_dim, action_dim,device,replay_buffer_size=NONE,replacement=NONE,
        batch_size=256,lr_a=1e-4, lr_c=1e-4,net_target=NONE,tau=0.005,gamma=0.99,) :
        # super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hard_replacement_counter = 0
        
        self.device=device
        self.replacement=replacement
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size
        self.tau=tau

        # 记忆库
        # self.experiencepool = ExperiencePool(memory_capacity)
        
        # # 定义 Actor 网络
        # self.actor = Actor(state_dim, action_dim)
        # self.actor_target = Actor(state_dim, action_dim)
        # # 定义 Critic 网络
        # self.critic = Critic(state_dim,action_dim)
        # self.critic_target = Critic(state_dim,action_dim)
        self.net=Net(self.state_dim,self.action_dim).cuda()
        self.net_target=deepcopy(self.net).cuda() if net_target==NONE else net_target.cuda()

        # 定义优化器
        self.aopt = torch.optim.Adam(self.net.actor.parameters(), lr=self.lr_a)
        self.copt = torch.optim.Adam(self.net.critic.parameters(), lr=self.lr_c)

        # 选取损失函数
        self.mse_loss = nn.MSELoss()

    def choose_action(self, s):
        # s=np.reshape(s,(1,8*10))
        s = torch.FloatTensor(s).cuda('cuda:0')
        action = self.net.actor(s)     
        return action.cpu().detach().numpy()
    def test_choose_action(self,s):
        s = torch.FloatTensor(s).cuda('cuda:0')
        action = self.net_target.actor(s)     
        return action.cpu().detach().numpy()

    def soft_update(self,target_net, current_net, tau):
        
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def learn(self):
        bs,ba,br,bs_,bd = self.replay_buffer.sample(self.batch_size)
        bs     = torch.FloatTensor(bs).to(self.device)
        ba     = torch.FloatTensor(ba).to(self.device)
        br     = torch.FloatTensor(br).unsqueeze(1).to(self.device)
        bs_ = torch.FloatTensor(bs_).to(self.device)
        bd      = torch.FloatTensor(np.float32(bd)).unsqueeze(1).to(self.device)
        

        
        # 训练Actor
        a = self.net.actor(bs)
        q = self.net.critic(bs, a)
        a_loss = -torch.mean(q)
       
        self.aopt.zero_grad()
        a_loss.backward(retain_graph=True)
        self.aopt.step()
        
        # 训练critic
        #compute the target Q value using the information of next state
        a_ = self.net_target.actor(bs_)
        q_ = self.net_target.critic(bs_, a_)
        q_target = br + (1-bd)*self.gamma * q_
        q_eval = self.net.critic(bs, ba)
        
        td_error = self.mse_loss(q_target,q_eval)
        
       
        self.copt.zero_grad()
        td_error.backward(retain_graph=True)
        self.copt.step()
        

        self.soft_update(self.net_target,self.net,self.tau)
        return a_loss,td_error

class SAC(object,):
    def __init__(self,state_dim, action_dim,device,batch_size=256,replay_buffer_size=NONE,
                soft_tau=0.005,soft_q_lr = 1e-5,policy_lr = 1e-4,gamma=0.99,policy_net=NONE,
                hidden_dim=128,hidden_dim2=64,hidden_dim3=34,
                #hidden_dim=256,hidden_dim2=128,hidden_dim3=128
                ):#1e-2):
#增加节点数有时候可以使网络训练更快
        self.action_dim = action_dim
        self.state_dim  = state_dim

        self.batch_size=batch_size
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        
        self.device=device
        self.hidden_dim=hidden_dim
        self.hidden_dim2=hidden_dim2
        self.hidden_dim3=hidden_dim3
        

        # self.value_lr  =value_lr
        self.soft_q_lr = soft_q_lr
        self.policy_lr = policy_lr
        self.gamma=gamma
        self.soft_tau=soft_tau
      
        
        #网络搭建
        #value network deleted
        # self.value_net        = ValueNetwork(self.state_dim, self.hidden_dim).to(device)
        # self.target_value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(device)

        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim,self.hidden_dim2,self.hidden_dim3).to(device)\
            if policy_net==NONE else policy_net.to(device)
        

        # self.soft_q_net = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.soft_q_net = CriticTwin(self.state_dim, self.action_dim,self.hidden_dim,self.hidden_dim2, self.hidden_dim3).to(device)
        
        self.soft_q_net_target = deepcopy(self.soft_q_net).to(device)

       
        #损失函数和优化器
        # self.value_criterion  = nn.MSELoss()
        # self.soft_q_criterion = nn.MSELoss()
        self.value_criterion = torch.nn.SmoothL1Loss(reduction="mean")
        self.soft_q_criterion = torch.nn.SmoothL1Loss(reduction="mean")

        # self.value_opt  = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.soft_q_opt = torch.optim.Adam(self.soft_q_net.parameters(), lr=self.soft_q_lr)
        self.policy_opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        

        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.soft_plus = nn.Softplus()
#1e-7
        self.alpha_log = torch.tensor(
            (np.log(0),), dtype=torch.float32, requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=0.00001)#0.0001)
        self.target_entropy = -self.action_dim

    def evaluate(self, state):
        mean, log_std = self.policy_net(state)
        std = log_std.exp()
       
        noise = torch.randn_like(mean, requires_grad=True)
        a_noise = mean + std * noise
        
        # action = torch.tanh(mean+ std*z.to(self.device))
        # log_prob = log_std + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)
        log_prob = Normal(mean, std).log_prob(a_noise)
        # log_prob = log_prob + (-a_noise.pow(2) + 1.000001).log()
        log_prob += (np.log(2.0) - a_noise - self.soft_plus(-2.0 * a_noise)) * 2
        
         # log_prob = torch.mean(Normal(mean, std).log_prob(mean+ std*z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon),dim=1,keepdim=True)
       
        
        return  a_noise.tanh(), log_prob.sum(1, keepdim=True)
        
    def choose_action(self, state):
        state = state.astype(float)
        state = torch.FloatTensor(state).to(self.device)
       
        mean, log_std = self.policy_net(state)
        
        std = log_std.exp()
        
        action=torch.normal(mean, std).tanh()
        action  = action.detach().cpu().numpy()

        return action
        
    def test_choose_action(self,state):
        state = torch.FloatTensor(state).to(self.device)
        mean, log_std = self.policy_net(state) 
        mean=mean.tanh()  
        return mean.detach().cpu().numpy()

    def soft_update(self,target_net, current_net, tau):
        
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data* tau + tar.data * (1.0 - tau))

    def get_q_loss(self):
        with torch.no_grad():
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            state      = torch.FloatTensor(state).to(self.device)
            action     = torch.FloatTensor(action).to(self.device)
            reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

            mask = (1 - done) * self.gamma

            next_action, next_log_prob = self.evaluate(next_state)
            next_q=self.soft_q_net_target.get_q_min(next_state,next_action)

            alpha = self.alpha_log.exp().detach()
            q_target = reward + mask* (next_q-alpha*next_log_prob)
        q1,q2=self.soft_q_net.get_q1_q2(state,action)
        q_loss=(self.soft_q_criterion(q1,q_target)+self.soft_q_criterion(q2,q_target))/2
        return q_loss, state


    def learn(self):
        q_loss = torch.zeros(1)
        policy_loss = torch.zeros(1)
        #Q网络更新
        q_loss, state=self.get_q_loss()

        self.soft_q_opt.zero_grad()
        q_loss.backward()
        self.soft_q_opt.step()

        #软更新
        self.soft_update(self.soft_q_net_target,self.soft_q_net,self.soft_tau)
        
        #阿尔法更新
        action, log_prob = self.evaluate(state)
        alpha_loss = (self.alpha_log* (-log_prob - self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad
        alpha_loss.backward()
        self.alpha_optim.step()

        #actor更新
        
        alpha = self.alpha_log.exp().detach()
        with torch.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
        q_value_pg = self.soft_q_net.get_q_min(state, action)
        policy_loss = -(q_value_pg - log_prob * alpha).mean()
       
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        return q_loss.item(),-policy_loss.item(),self.alpha_log.exp().detach().item()

        
    #SAC算法一
    '''def learn(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        
        next_action, next_log_prob = self.evaluate(next_state)
        next_q=self.soft_q_net_target.get_q_min(next_state,next_action)
        alpha = self.alpha_log.exp().detach()
        q_target = reward + (1 - done) * self.gamma * (next_q-next_log_prob)
        
        q_loss=(self.soft_q_criterion(q1,q_target)+self.soft_q_criterion(q2,q_target))/2
        
        expected_q_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        # new_action, log_prob = self.evaluate(state)
        
   

        target_value = self.target_value_net(next_state)
        
       
        # q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - self.alpha*log_prob
        # print(expected_value,next_value)
        value_loss = self.value_criterion(expected_value, next_value.detach())
        
      
        log_prob_target = expected_new_q_value - expected_value
        #policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        # policy_loss = (log_prob-expected_new_q_value).mean().detach()
        policy_loss = (log_prob - expected_new_q_value.detach()).mean()
        
        # print(q_value_loss,value_loss,policy_loss)
        self.soft_q_opt.zero_grad()
        q_loss.backward()
        self.soft_q_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        #软更新
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)'''




###there are some nn for rl algrithm 
class SimpleRNN(nn.Module):
    def __init__(self, x_size, hidden_size, n_layers, batch_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        #self.inp = nn.Linear(1, hidden_size) 
        self.rnn = nn.RNN(x_size, hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size) # 10 in and 10 out
 
    def forward(self, inputs, hidden=None):
        hidden = self.__init__hidden()
        #print("Forward hidden {}".format(hidden.shape))
        #print("Forward inps {}".format(inputs.shape))
        output, hidden = self.rnn(inputs.float(), hidden.float())
        #print("Out1 {}".format(output.shape))
        output = self.out(output.float());
        #print("Forward outputs {}".format(output.shape))
 
        return output, hidden
 
    def __init__hidden(self):
        hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_size, dtype=torch.float64)
        return hidden

#DDPG网络
class Net(nn.Module):
    def __init__(self,state_dim,action_dim) :
        super(Net,self).__init__()
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.actor=Actor(self.state_dim,self.action_dim)
        self.critic=Critic(self.state_dim,self.action_dim)
  
'''class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor,self).__init__()
        self.action_bound = torch.tensor(action_bound)

        # 神经网络层layer
        self.layer1 = nn.Linear(state_dim, 30)
        nn.init.normal_(self.layer1.weight, 0., 0.3) # 以正态分布初始化权重，均值0，标准差0.3
        nn.init.constant_(self.layer1.bias, 0.1)
        
        #输出层
        self.output = nn.Linear(30, action_dim)
        self.output.weight.data.normal_(0.,0.3) # 初始化权重
        self.output.bias.data.fill_(0.1)

    def forward(self, s):
        #relu激活
        a = torch.relu(self.layer1(s))
        # 对action进行放缩,实际上a in [-1,1]
        a = torch.tanh(self.output(a))
        #恢复action
        scaled_a = a * self.action_bound
        return scaled_a'''
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,init_w=3e-3):
        super(Actor,self).__init__()
      
        self.hidsize=256
        self.hidsize1=128
        # self.hidsize2=8
        # 神经网络层layer
        self.layer1 = nn.Linear(state_dim, self.hidsize)
       
        self.layer2=nn.Linear(self.hidsize,self.hidsize1)
        # self.layer3=nn.Linear(self.hidsize1,self.hidsize2)
        #输出层
        self.output = nn.Linear(self.hidsize1, action_dim)
        self.output.weight.data.uniform_(-init_w, init_w)
        self.output.bias.data.uniform_(-init_w, init_w)
         # nn.init.normal_(self.layer1.weight, 0., 0.3) # 以正态分布初始化权重，均值0，标准差0.3
        # nn.init.constant_(self.layer1.bias, 0.1)
        # self.output.weight.data.normal_(0.,0.3) # 初始化权重
        # self.output.bias.data.fill_(0.1)

    def forward(self, s):
        #relu激活
        a = F.relu(self.layer1(s))
        a=F.relu(self.layer2(a))
        # a=F.relu(self.layer3(a))
        a = torch.tanh(self.output(a))
        #恢复action
       
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,init_w=3e-3):
        super(Critic,self).__init__()
        self.hidsize =256
        self.hidsize1=128
        # self.hidsize2=8
        # 状态维度层
        self.layer1 = nn.Linear(state_dim+action_dim, self.hidsize)
        self.layer2 = nn.Linear(self.hidsize, self.hidsize1)
        # self.layer3 = nn.Linear(self.hidsize1, self.hidsize2)
        #输出层
        self.output = nn.Linear(self.hidsize1, 1)
        # self.output.weight.data.normal_(0.,0.3) # 初始化权重
        # self.output.bias.data.fill_(0.1)
        self.output.weight.data.uniform_(-init_w, init_w)
        self.output.bias.data.uniform_(-init_w, init_w)

    def forward(self,s,a):
        x = torch.cat([s,a], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        q_value = self.output(x)
        return q_value

        
   #SAC     

#SAC Actor-network 
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, actions_dim, hidden_size,hidden_size2, hidden_size3, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hidden_size=hidden_size
        self.hidden_size2=hidden_size2
        # self.hidden_size3=hidden_size3
        
        
        self.linear1 = nn.Linear(state_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size2)
        # self.linear3=nn.Linear(self.hidden_size2,self.hidden_size3)
        
        #均值输出层
        self.mean_linear = nn.Linear(self.hidden_size2, actions_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        #log方差输出层
        self.log_std_linear = nn.Linear(self.hidden_size2, actions_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        #log方差裁剪
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

#SAC Q-network 
class CriticTwin(nn.Module):  
    # shared parameter
    '''def __init__(self,  state_dim, action_dim,hidden_size,hidden_size3):
        super().__init__()
        self.net_sa = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())  # concat(state, action)
        self.net_q1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size3), nn.ReLU(), nn.Linear(hidden_size3, 1))  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size3), nn.ReLU(), nn.Linear(hidden_size3, 1))  # q2 value'''

    # no shared parameter
    def __init__(self,  state_dim, action_dim,hidden_size,hidden_size2,hidden_size3,init_w=3e-3):
        super(CriticTwin,self).__init__()
        self.hidden_size=hidden_size
        self.hidden_size2=hidden_size2
        # self.hidden_size3=hidden_size3
        self.net_q1=nn.Sequential(
            nn.Linear(state_dim+action_dim,self.hidden_size),nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size2),nn.ReLU(),
            #  nn.Linear(self.hidden_size2,self.hidden_size3),nn.ReLU(), 
           )
       
        self.net_q2=nn.Sequential(
            nn.Linear(state_dim+action_dim,self.hidden_size),nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size2),nn.ReLU(),
            # nn.Linear(self.hidden_size2,self.hidden_size3),nn.ReLU(),
          )

        self.out_q1=nn.Linear(self.hidden_size2,1)
        self.out_q1.weight.data.uniform_(-init_w, init_w)
        self.out_q1.bias.data.uniform_(-init_w, init_w)

        self.out_q2=nn.Linear(self.hidden_size2,1)
        self.out_q2.weight.data.uniform_(-init_w, init_w)
        self.out_q2.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        
        return torch.add(*self.get_q1_q2(state, action)) / 2.0  # mean Q value

    def get_q_min(self, state, action):
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state, action):
        x=torch.cat((state,action),dim=1)
        return self.out_q1(self.net_q1(x)),self.out_q2(self.net_q2(x))


        # tmp = self.net_sa(torch.cat((state, action), dim=1))
        # return self.net_q1(tmp), self.net_q2(tmp)  # two Q values
   

#SAC Q网络,版本一  
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, actions_dim, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + actions_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


#SAC value网络,版本一
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
         #两层隐藏层
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        #输出层初始化
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

            
      

        
        

        

        

