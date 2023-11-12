import torch
import json
import pandas as pd
import datetime
from uav import *
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from visualization import  flvisual
from configuration import ConfigDraw, ConfigTrain
from agent import AgentSAC, AgentDDPG, AgentPPO
import matplotlib.pyplot as plt
import sys
import os


# sys.path.append('./RL')
sys.path.append('/root/paper2/RL')
writer = SummaryWriter('./tensorboard/log/')


###time is episodes*global_epochs*local_epochs*4s  such as 50*100*2*4 =40000s = 12h

##twenty global epochs, 2s, -0.3 energy
def train(conf, rl, dir):
    # np.random.seed(2023)
    # torch.manual_seed(2023)
    if not os.path.exists(dir):
        os.makedirs(dir)

    date1 = datetime.datetime.now().strftime('%m-%d')
    
    rl = rl
    assert isinstance(rl, AgentSAC)
    return_ls = []
    ene_consum_ls = []
    for i in range(conf['config_train'].MAX_EPISODES):
        ##for test
        if i == conf['config_train'].MAX_EPISODES :
            ep_reward, energy_consum, actions, acc_loss_test, tra_ls = rl.episode_test(i, conf['global_epochs'], )
            return_ls.append(ep_reward)
            ene_consum_ls.append(energy_consum/1000)
        else :
            ep_reward, energy_consum = rl.episode(i, conf['global_epochs'])
            if len(return_ls) >= 1 and ep_reward > max(return_ls):
                rl.save_model()
            if i%5 == 0:
                rl.update(update_times = 200)
            if i % 25 == 0:
                rl.agent.decay_action_std(rl.action_std_decay_rate, rl.min_action_std)


            return_ls.append(ep_reward)
            ene_consum_ls.append(energy_consum/1000)
            print('\n')


    ######保存文件
    df_actions = pd.DataFrame(actions)
    df_actions.to_csv(os.path.join(dir, f'actions.csv'))

    df_tra = pd.DataFrame(tra_ls)
    df_tra.to_csv(os.path.join(dir, f'tra.csv'))
    ###
    ls_return_ene = [[i ,j] for i, j in zip(return_ls, ene_consum_ls)]  
    df_return_ene = pd.DataFrame(ls_return_ene, columns=['return', 'energy'])
    df_return_ene.to_csv(os.path.join(dir, f'return_ene.csv'))

    df_fl = pd.DataFrame(acc_loss_test)
    df_fl.to_csv(os.path.join(dir, f'acc_loss.csv'))


    return return_ls, ene_consum_ls, df_fl   


    
def rlvisual(df_fl_SAC = None, df_fl_DDPG = None, 
            return_ls_SAC = None, return_ls_DDPG = None, return_ls_PPO = None,
            ene_consum_ls_SAC = None, ene_consum_ls_DDPG = None, ene_consum_ls_PPO = None
            ):
    #####     可视化FL
    df_fl_SAC  = df_fl_SAC   if df_fl_SAC  is not None else pd.read_csv('./output/main_output/SAC/acc_loss.csv')
    df_fl_DDPG = df_fl_DDPG  if df_fl_DDPG is not None else pd.read_csv('./output/main_output/DDPG/acc_loss.csv')

    date = datetime.datetime.now().strftime('%m-%d')
    df = pd.concat([df_fl_SAC[['global_accuracy', 'global_loss']], df_fl_DDPG[['global_accuracy', 'global_loss']]], axis=1,)
    df.to_csv(f'fl.csv')
    df.columns =  ['SAC_acc', 'SAC_loss',  'DDPG_acc', 'DDPG_loss']
    # ['SAC', 'SAC_acc', 'SAC_loss', 'DDPG', 'DDPG_acc', 'DDPG_loss']
    flvisual(df, date)

    ######   return和energy 可视化
    return_ene_SAC  =  pd.read_csv('./output/main_output/SAC/return_ene.csv')
    return_ene_DDPG = pd.read_csv('./output/main_output/DDPG/return_ene.csv')
    return_ene_PPO = pd.read_csv('./output/main_output/PPO/return_ene.csv')

    return_ls_SAC  = return_ls_SAC   if return_ls_SAC   is not None else return_ene_SAC['return']
    return_ls_DDPG = return_ls_DDPG   if return_ls_DDPG  is not None else return_ene_DDPG['return']
    return_ls_PPO = return_ls_PPO   if return_ls_PPO  is not None else return_ene_PPO['return']

    ene_consum_ls_SAC   = ene_consum_ls_SAC   if ene_consum_ls_SAC   is not None else return_ene_SAC['energy']
    ene_consum_ls_DDPG  = ene_consum_ls_DDPG  if ene_consum_ls_DDPG  is not None else return_ene_DDPG['energy']
    ene_consum_ls_PPO   = ene_consum_ls_PPO  if ene_consum_ls_PPO  is not None else return_ene_PPO['energy']

    #####
    fig1, ax1 = plt.subplots()

    # plt.plot(episode_reward_sac_0, color='r', linewidth=1, linestyle='-',label='without beamforming(location)')
    # plt.plot(episode_reward_sac_1, color='g', linewidth=1, linestyle='-',label='without beamforming(distance)')
    ax1.plot(return_ls_SAC,  linewidth = 1, linestyle='-',label='return with proposed')
    ax1.plot(return_ls_DDPG, linewidth = 1, linestyle='-',label='return with DDPG')
    ax1.plot(return_ls_PPO, linewidth = 1, linestyle='-',label='return with PPO')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Return')
    
    ax1.legend()
    fig1.savefig('./output/main_output/RL/ddpg+sac_return.jpg')
    fig1.savefig('./output/main_output/RL/ddpg+sac_return.eps')
    fig1.savefig('./output/main_output/RL/ddpg+sac_return.pdf')

    ########
    fig2, ax2 = plt.subplots()
    
    ax2.plot(ene_consum_ls_SAC,  linewidth=1, linestyle='-',label='energy consumption with proposed')
    ax2.plot(ene_consum_ls_DDPG, linewidth=1, linestyle='-',label='energy consumption with DDPG')
    ax2.plot(ene_consum_ls_PPO, linewidth=1, linestyle='-',label='energy consumption with PPO')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Energy consumption (kJ)')

    ax2.legend()
    fig2.savefig('./output/main_output/RL/ddpg+sac_energy.jpg')
    fig2.savefig('./output/main_output/RL/ddpg+sac_energy.eps')
    fig2.savefig('./output/main_output/RL/ddpg+sac_energy.pdf')
    
    return 


if __name__ == '__main__':
	# date = datetime.datetime.now().strftime('%H:%M:%S')
	
    print(torch.cuda.is_available())
    print(torch.__version__)

    # torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # torch.set_float32_matmul_precision('high')
    with open('./conf.json' , 'r') as f:
        conf = json.load(f)
    config_train = ConfigTrain()
    config_draw = ConfigDraw()
    conf['config_train'] = config_train
    conf["config_draw"] = config_draw 


    ##you can train from the start or get data from saved csv file
    return_ls_PPO, ene_consum_ls_PPO, df_fl_PPO = train(conf, AgentPPO(conf, dir='./output/main_output/PPO'), dir='./output/main_output/PPO/')
    # return_ls_SAC, ene_consum_ls_SAC, df_fl_SAC = train(conf, AgentSAC(conf, dir='./output/main_output/SAC'), dir='./output/main_output/SAC/')
    # return_ls_DDPG, ene_consum_ls_DDPG, df_fl_DDPG = train(conf, AgentDDPG(conf, dir='DDPG/'), dir='DDPG/')

    if True:
        rlvisual(
                # df_fl_SAC =df_fl_SAC, 
                # ene_consum_ls_SAC = ene_consum_ls_SAC, 
                # return_ls_SAC = return_ls_SAC, 
                # df_fl_DDPG = df_fl_DDPG, 
                # return_ls_DDPG = return_ls_DDPG, 
                # ene_consum_ls_DDPG = ene_consum_ls_DDPG, 
                )   

        
    
	