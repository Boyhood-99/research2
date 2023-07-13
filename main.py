import torch
import json
import pandas as pd
import datetime
from uav import *
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from visualization import rlvisual, flvisual
from configuration import ConfigDraw, ConfigTrain
from agent import AgentSAC
import matplotlib.pyplot as plt
import sys
# sys.path.append('./RL')
sys.path.append('/root/paper2/RL')
writer = SummaryWriter('./tensorboard/log/')

##原来的40 收敛， 100episodes足够
###time is episodes*global_epochs*local_epochs*4s  such as 50*100*2*4 =40000s = 12h

def main(conf):
    date = datetime.datetime.now().strftime('%m-%d')
    
    rl = AgentSAC(conf)
    return_ls = []
    ene_consum_ls = []
    for i in range(conf['config_train'].MAX_EPISODES):
        if i == conf['config_train'].MAX_EPISODES - 1:
            ep_reward, energy_consum, actions, acc_loss_test = rl.episode(i, conf['global_epochs'], test = True)
            return_ls.append(ep_reward)
            ene_consum_ls.append(energy_consum)
        else :
            ep_reward, energy_consum = rl.episode(i, conf['global_epochs'])
            
            rl.update(update_times = 200)
            return_ls.append(ep_reward)
            ene_consum_ls.append(energy_consum)
            
    print(actions)
    df = pd.DataFrame(acc_loss_test)
    df.to_csv(f'acc_loss_test.csv')
    
    flvisual(df, date)

    
    fig1, ax1 = plt.subplots()
    

    # plt.plot(episode_reward_fix,color='g', linewidth=1, linestyle='-',label='fixed')
    # plt.plot(episode_reward_sac,color='b', linewidth=1, linestyle='-',label='with beamforming')
    # plt.plot(episode_reward_ddpg,color='r', linewidth=1, linestyle='-',label='DDPG')

    # plt.plot(episode_reward_sac_0, color='r', linewidth=1, linestyle='-',label='without beamforming(location)')
    # plt.plot(episode_reward_sac_1, color='g', linewidth=1, linestyle='-',label='without beamforming(distance)')
    ax1.plot(return_ls, color='r', linewidth=1, linestyle='-',label='Return with beamforming')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Return')
    
    plt.legend()
    plt.savefig('./SAC/ddpg+sac_return.jpg')
    plt.savefig('./SAC/ddpg+sac_return.eps')
    plt.savefig('./SAC/ddpg+sac_return.pdf')
    #####

    fig1, ax1 = plt.subplots()
    
    ax1.plot(ene_consum_ls, color='b', linewidth=1, linestyle='-',label='Energy consumption with beamforming')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Energy consumption')

    
    plt.legend()
    
    plt.savefig('./SAC/ddpg+sac_energy.jpg')
    plt.savefig('./SAC/ddpg+sac_energy.eps')
    plt.savefig('./SAC/ddpg+sac_energy.pdf')
    
    return 


if __name__ == '__main__':
	# date = datetime.datetime.now().strftime('%H:%M:%S')
	# print(date)
    print(torch.cuda.is_available())
    print(torch.__version__)
    # torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # torch.set_float32_matmul_precision('high')
    with open('./conf.json' , 'r') as f:
        conf = json.load(f)
    config_train = ConfigTrain
    config_draw = ConfigDraw
    conf['config_train'] = config_train
    conf["config_draw"] = config_draw 
   
    main(conf=conf)
        
    
	