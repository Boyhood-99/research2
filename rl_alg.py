from pickle import NONE
from turtle import color
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from buffer import ReplayBuffer, RolloutBuffer
from copy import deepcopy
from torch.distributions import Normal
from rl_net import *
###there are some rl algrithms

# DDPG algrithm easily converge to the action edge, cause some 
#results wo don't want, raise zerodivision or zero value problem
class DDPG(object):
    def __init__(self, state_dim, action_dim, device = None, 
                 replay_buffer_size = NONE,replacement = NONE,
                batch_size = 256, lr_a = 1e-4, lr_c = 1e-5,
                net_target = NONE, tau = 0.005, gamma = 0.99,
                hidden_dim = 128,
        ) :

        # super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hard_replacement_counter = 0
        
        self.device = device
        self.replacement = replacement
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size
        self.tau = tau

        # 记忆库
        # self.experiencepool = ExperiencePool(memory_capacity)
        
        # # 定义 Actor 网络
        # self.actor = Actor(state_dim, action_dim)
        # self.actor_target = Actor(state_dim, action_dim)
        # # 定义 Critic 网络
        # self.critic = Critic(state_dim,action_dim)
        # self.critic_target = Critic(state_dim,action_dim)
        self.net = Net(self.state_dim, self.action_dim, hidden_dim = hidden_dim).cuda()
        self.net_target = deepcopy(self.net).cuda() if net_target == NONE else net_target.cuda()

        # 定义优化器
        self.aopt = torch.optim.Adam(self.net.actor.parameters(), lr = self.lr_a)
        self.copt = torch.optim.Adam(self.net.critic.parameters(), lr = self.lr_c)

        # 选取损失函数
        self.mse_loss = nn.MSELoss()

    def choose_action(self, s):
        # s = np.reshape(s,(1,8*10))
        s = torch.FloatTensor(s).cuda('cuda:0')
        action = self.net.actor(s)     
        return action #.cpu().detach().numpy()
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
        x = torch.cat([bs,a], 1)
        q = self.net.critic(x)
        a_loss = -torch.mean(q)
       
        self.aopt.zero_grad()
        a_loss.backward(retain_graph = True)
        self.aopt.step()
        
        # 训练critic
        #compute the target Q value using the information of next state
        a_ = self.net_target.actor(bs_)
        x_ = torch.cat([bs_,a_], 1)
        q_ = self.net_target.critic(x_)
        q_target = br + (1-bd)*self.gamma * q_
        q_eval = self.net.critic(torch.cat([bs,ba], 1))
        
        td_error = self.mse_loss(q_target,q_eval)
        
       
        self.copt.zero_grad()
        td_error.backward(retain_graph = True)
        self.copt.step()
        

        self.soft_update(self.net_target, self.net,self.tau)
        return a_loss,td_error

class SAC(object,):
    def __init__(self,state_dim, action_dim,device,batch_size = 256,replay_buffer_size = NONE,
                soft_tau = 0.005,soft_q_lr = 1e-5,policy_lr = 1e-4,gamma = 0.99,actor = NONE,
                hidden_dim = 128,hidden_dim2 = 64,hidden_dim3 = 64,
                #hidden_dim = 256,hidden_dim2 = 128,hidden_dim3 = 128
                ):#1e-2):
#增加节点数有时候可以使网络训练更快
        self.action_dim = action_dim
        self.state_dim  = state_dim

        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        
        self.device = device
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        

        # self.value_lr   = value_lr
        self.soft_q_lr = soft_q_lr
        self.policy_lr = policy_lr
        self.gamma = gamma
        self.soft_tau = soft_tau
      
        
        #网络搭建
        #value network deleted
        # self.value_net        = ValueNetwork(self.state_dim, self.hidden_dim).to(device)
        # self.target_value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(device)

        self.actor = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.hidden_dim2, self.hidden_dim3).to(device)\
            if actor == NONE else actor.to(device)
        

        # self.critic = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic = CriticTwin(self.state_dim, self.action_dim, self.hidden_dim,self.hidden_dim2, self.hidden_dim3).to(device)
        
        self.critic_target = deepcopy(self.critic).to(device)

       
        #损失函数和优化器
        # self.value_criterion  = nn.MSELoss()
        # self.soft_q_criterion = nn.MSELoss()
        self.value_criterion = torch.nn.SmoothL1Loss(reduction = "mean")
        self.soft_q_criterion = torch.nn.SmoothL1Loss(reduction = "mean")

        # self.value_opt  = torch.optim.Adam(self.value_net.parameters(), lr = self.value_lr)
        self.soft_q_opt = torch.optim.Adam(self.critic.parameters(), lr = self.soft_q_lr)
        self.policy_opt = torch.optim.Adam(self.actor.parameters(), lr = self.policy_lr)

        
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.soft_plus = nn.Softplus()
        #1e-7
        self.alpha_log = torch.tensor(
            (np.log(0),), dtype = torch.float32, requires_grad = True, device = self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr = 0.00001)#0.0001)
        self.target_entropy = -self.action_dim

    def evaluate(self, state):
        mean, log_std = self.actor(state)
        std = log_std.exp()
       
        noise = torch.randn_like(mean, requires_grad = True)
        a_noise = mean + std * noise
        
        # action = torch.tanh(mean+ std*z.to(self.device))
        # log_prob = log_std + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)
        log_prob = Normal(mean, std).log_prob(a_noise)
        # log_prob = log_prob + (-a_noise.pow(2) + 1.000001).log()
        log_prob +=  (np.log(2.0) - a_noise - self.soft_plus(-2.0 * a_noise)) * 2
        
         # log_prob = torch.mean(Normal(mean, std).log_prob(mean+ std*z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon),dim = 1,keepdim = True)
       
        return  a_noise.tanh(), log_prob.sum(1, keepdim = True)
        
    def choose_action(self, state):
        state = state.astype(float)
        state = torch.FloatTensor(state).to(self.device)
        mean, log_std = self.actor(state)
        std = log_std.exp()
        action = torch.normal(mean, std).tanh()
        action  = action.detach().cpu().numpy()

        return action
        
    def test_choose_action(self,state):
        state = torch.FloatTensor(state).to(self.device)
        mean, log_std = self.actor(state) 
        mean = mean.tanh()  
        # print(log_std.exp())
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
            next_q = self.critic_target.get_q_min(next_state,next_action)

            alpha = self.alpha_log.exp().detach()
            q_target = reward + mask* (next_q-alpha*next_log_prob)
        q1,q2 = self.critic.get_q1_q2(state,action)
        q_loss = (self.soft_q_criterion(q1,q_target)+self.soft_q_criterion(q2,q_target))/2
        return q_loss, state


    def learn(self):
        q_loss = torch.zeros(1)
        policy_loss = torch.zeros(1)
        #Q网络更新
        q_loss, state = self.get_q_loss()

        self.soft_q_opt.zero_grad()
        q_loss.backward()
        self.soft_q_opt.step()

        #软更新
        self.soft_update(self.critic_target, self.critic,self.soft_tau)
        
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
        q_value_pg = self.critic.get_q_min(state, action)
        policy_loss = -(q_value_pg - log_prob * alpha).mean()
       
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        return q_loss.item(),-policy_loss.item(),self.alpha_log.exp().detach().item()


class PPO:
    def __init__(self, state_dim, action_dim, device, hidden_dim = 128, lr_a = 1e-4, lr_c = 1e-5,
                  gamma  = 0.99, K_epochs = 10, eps_clip = 0.1, has_continuous_action_space = True, 
                  action_std_init=0.6, 
                  ):

        self.has_continuous_action_space = has_continuous_action_space
        self.hidden_dim = hidden_dim
        self.device = device
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.replay_buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, self.hidden_dim, self.has_continuous_action_space, 
                                    action_std_init, self.device).to(self.device)
        
        self.policy_old = ActorCritic(state_dim, action_dim, self.hidden_dim, self.has_continuous_action_space, 
                                      action_std_init, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_a},
                        {'params': self.policy.critic.parameters(), 'lr': lr_c}
                    ])
        self.MseLoss = nn.MSELoss()

    def test_choose_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action_mean, log_std = self.policy.actor(state)
            return action_mean.tanh().detach().cpu().numpy().flatten()
        ###
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action_prob = self.policy.actor(state)
            return max(action_prob).item()
    def choose_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)
            action_clip = action.tanh()
            self.replay_buffer.states.append(state)
            self.replay_buffer.actions.append(action)
            # self.replay_buffer.actions_clip.append(action_clip)
            self.replay_buffer.logprobs.append(action_logprob)
            self.replay_buffer.state_values.append(state_val)

            return action_clip.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.replay_buffer.states.append(state)
            self.replay_buffer.actions.append(action)
            self.replay_buffer.logprobs.append(action_logprob)
            self.replay_buffer.state_values.append(state_val)
            return action.item()

    def learn(self, ):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(self.replay_buffer.rewards), reversed(self.replay_buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        
        old_states = torch.squeeze(torch.stack(self.replay_buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.replay_buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.replay_buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.replay_buffer.state_values, dim=0)).detach().to(self.device)
        # print(old_states)
        # print(old_actions)
        # print(old_logprobs)
        # print(old_state_values)
        
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # print(loss)
        
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.replay_buffer.clear()
    
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
          
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        



    #SAC算法一
    '''def learn(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        
        next_action, next_log_prob = self.evaluate(next_state)
        next_q = self.critic_target.get_q_min(next_state,next_action)
        alpha = self.alpha_log.exp().detach()
        q_target = reward + (1 - done) * self.gamma * (next_q-next_log_prob)
        
        q_loss = (self.soft_q_criterion(q1,q_target)+self.soft_q_criterion(q2,q_target))/2
        
        expected_q_value = self.critic(state, action)
        expected_value   = self.value_net(state)
        # new_action, log_prob = self.evaluate(state)
        
   

        target_value = self.target_value_net(next_state)
        
       
        # q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.critic(state, new_action)
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




            
      

        
        

        

        

