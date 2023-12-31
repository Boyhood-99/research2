import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

###DDPG network
class Net(nn.Module):
    def __init__(self,state_dim, action_dim, hidden_dim = 128) :
        super(Net,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.actor = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim*2),
                                   nn.ReLU(),
                                #    nn.Linear(self.hidden_dim*2, self.hidden_dim*2), 
                                #    nn.ReLU(),
                                   nn.Linear(self.hidden_dim*2, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.action_dim),
                                   nn.Tanh()

        )
        self.critic = nn.Sequential(nn.Linear(self.state_dim+self.action_dim, self.hidden_dim*2),
                                   nn.ReLU(),
                                #    nn.Linear(self.hidden_dim*2, self.hidden_dim*2), 
                                #    nn.ReLU(),
                                   nn.Linear(self.hidden_dim*2, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, 1),
        )
    
### SAC     
#SAC Actor-network 
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, actions_dim, hidden_size,hidden_size2, hidden_size3, init_w = 3e-3, log_std_min = -20, log_std_max = 2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        # self.hidden_size3 = hidden_size3
        
        self.linear1 = nn.Linear(state_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size2)
        # self.linear3 = nn.Linear(self.hidden_size2,self.hidden_size3)
        
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
    def __init__(self,  state_dim, action_dim,hidden_size,hidden_size2,hidden_size3,init_w = 3e-3):
        super(CriticTwin,self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        # self.hidden_size3 = hidden_size3
        self.net_q1 = nn.Sequential(
            nn.Linear(state_dim+action_dim,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size2),
            nn.ReLU(),
            #  nn.Linear(self.hidden_size2,self.hidden_size3),
            # nn.ReLU(), 
           )
       
        self.net_q2 = nn.Sequential(
            nn.Linear(state_dim+action_dim,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size2),
            nn.ReLU(),
            # nn.Linear(self.hidden_size2,self.hidden_size3),
            # nn.ReLU(),
          )

        self.out_q1 = nn.Linear(self.hidden_size2,1)
        self.out_q1.weight.data.uniform_(-init_w, init_w)
        self.out_q1.bias.data.uniform_(-init_w, init_w)

        self.out_q2 = nn.Linear(self.hidden_size2,1)
        self.out_q2.weight.data.uniform_(-init_w, init_w)
        self.out_q2.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        
        return torch.add(*self.get_q1_q2(state, action)) / 2.0  # mean Q value

    def get_q_min(self, state, action):
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state, action):
        x = torch.cat((state,action),dim = 1)
        return self.out_q1(self.net_q1(x)),self.out_q2(self.net_q2(x))


        # tmp = self.net_sa(torch.cat((state, action), dim = 1))
        # return self.net_q1(tmp), self.net_q2(tmp)  # two Q values

### PPO
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, has_continuous_action_space, action_std_init, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.has_continuous_action_space = has_continuous_action_space
        self.hidden_size = hidden_size
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
            # print(self.action_var)
        # actor
        self.actor = PolicyNetwork(state_dim, action_dim, self.hidden_size*2, self.hidden_size, self.hidden_size).to(device)
        # self.actor = nn.Sequential(
        #                     nn.Linear(state_dim, self.hidden_size*2),
        #                     # nn.Tanh(),
        #                     nn.ReLU(),
        #                     nn.Linear(self.hidden_size*2, self.hidden_size),
        #                     # nn.Tanh(),
        #                     nn.ReLU(),
        #                     nn.Linear(self.hidden_size, action_dim),
        #                     nn.Tanh(),
        #                 )
        if not has_continuous_action_space :
            self.actor[-1] = nn.Softmax(dim=-1)

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, self.hidden_size*2),
                        # nn.Tanh(),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size*2, self.hidden_size),
                        # nn.Tanh(),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size, 1)
                    )

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean, log_std = self.actor(state)
            std = log_std.exp()
            # print(std)
            # action = torch.normal(action_mean, std).tanh()
            cov_mat = torch.diag_embed(std)#.unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean, log_std = self.actor(state)
            std = log_std.exp()
            cov_mat = torch.diag_embed(std).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim) 
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy
    def act_(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag_embed(self.action_var)#.unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()
    def evaluate_(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            cov_mat = torch.diag_embed(self.action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")

    def forward(self):
        raise NotImplementedError


''''''
###SAC 版本一网络
#SAC Q网络,版本一  
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, actions_dim, hidden_size, init_w = 3e-3):
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
    def __init__(self, state_dim, hidden_dim, init_w = 3e-3):
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


