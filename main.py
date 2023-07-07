import torch
import json
from uav import *
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from visualization import rlvisual
from configuration import ConfigDraw, ConfigTrain
from agent import AgentSAC
import sys
# sys.path.append('./RL')
sys.path.append('/root/paper2/RL')
writer = SummaryWriter('./tensorboard/log/')

##原来的40 收敛， 100episodes足够
###time is episodes*global_epochs*local_epochs*4s  such as 50*100*2*4 =40000s = 12h

def main(conf):
   

    
    rl = AgentSAC(conf)
    reward_ls = []
    for i in range(conf['config_train'].MAX_EPISODES):
        ep_reward = rl.episode(i, conf['global_epochs'])
        
        rl.update(update_times = 200)
        reward_ls.append(ep_reward)
    # episode_reward_sac,q,p,alpha,t_comm_sac,t_total_sac,l_uav_location_sac,f_uav_location_sac,d_sac,total_time_sac=sac_train()
    # episode_reward_sac_0,_,_,_,_,_,_,_,_,_=sac_train(env=Environment0())##state space based on location
    # episode_reward_sac_1,_,_,_,_,_,_,_,_,_=sac_train(env=Environment1())##state space based on distance
    
    
    
    rlvisual(conf['f_uav_num'], conf['config_draw'], reward_ls)
    return 


if __name__ == '__main__':
	# date = datetime.datetime.now().strftime('%H:%M:%S')
	# print(date)
    print(torch.cuda.is_available())
    print(torch.__version__)
    # torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # torch.set_float32_matmul_precision('high')
    with open('./utils/conf.json' , 'r') as f:
        conf = json.load(f)
    config_train = ConfigTrain
    config_draw = ConfigDraw
    conf['config_train'] = config_train
    conf["config_draw"] = config_draw 
   
    main(conf=conf)
        
    
	