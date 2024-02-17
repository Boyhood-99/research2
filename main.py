import torch
import json
import pandas as pd
import numpy as np
import datetime
from uav import *
from torch.utils.tensorboard import SummaryWriter
from visualization import  flvisual, rlvisual, tra_visual
from configuration import ConfigDraw, ConfigTrain
from agent import AgentSAC, AgentDDPG, AgentPPO, DDPG_
import os

writer = SummaryWriter('./tensorboard/log/')

def train(conf, fl, test, rl, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    date1 = datetime.datetime.now().strftime('%m-%d')
    beam = conf['config_train'].IS_BEAM
    v = conf['config_train'].VEL
    rl = rl
    ula_num = conf['config_train'].ULA_NUM
    is_beam = conf['config_train'].IS_BEAM
    assert isinstance(rl, AgentSAC)
    return_ls = []
    ene_consum_ls = []
    for epi_num in range(conf['config_train'].MAX_EPISODES):
        ##for test
        if test and epi_num == conf['config_train'].MAX_EPISODES - 1:
            ep_reward, energy_consum, actions, acc_loss_test, tra_ls = rl.episode_test(epi_num, conf['global_epochs'], )
            return_ls.append(ep_reward)
            ene_consum_ls.append(energy_consum/1000)
        else :
            ep_reward, energy_consum = rl.episode(epi_num, conf['global_epochs'])
            if len(return_ls) >= 1 and ep_reward > max(return_ls):
                rl.save_model()
            if epi_num% 2 == 0:
                # print(len(rl.agent.replay_buffer.rewards))
                # print(rl.agent.replay_buffer.rewards)
                # print(rl.agent.replay_buffer.is_terminals)
                rl.update(update_times = 20)
            # if epi_num % 10 == 0:
            #     rl.std_decay()

            return_ls.append(ep_reward)
            ene_consum_ls.append(energy_consum/1000)
            # print('\n')

    ######保存文件
    return_ls_mean = np.mean(return_ls[-len(return_ls)//5:])
    return_ls.append(return_ls_mean)
    ene_consum_ls_mean = np.mean(ene_consum_ls[-len(ene_consum_ls)//5:])
    ene_consum_ls.append(ene_consum_ls_mean)

    ls_return_ene = [[i ,j] for i, j in zip(return_ls, ene_consum_ls)]  
    df_return_ene = pd.DataFrame(ls_return_ene, columns=['return', 'energy'])
    df_return_ene.to_csv(os.path.join(dir, f'return_ene{is_beam}{ula_num}.csv'))
    ### test
    df_fl = None
    if test:
        if fl:
            df_fl = pd.DataFrame(acc_loss_test)
            df_fl.to_csv(os.path.join(dir, f'acc_loss{is_beam}{ula_num}.csv'))
        df_actions = pd.DataFrame(actions)
        df_actions.to_csv(os.path.join(dir, f'actions{is_beam}{ula_num}.csv'))

        df_tra = pd.DataFrame(tra_ls)
        df_tra.to_csv(os.path.join(dir, f'tra{is_beam}{ula_num}.csv'))

    return return_ls, ene_consum_ls, df_fl   

if __name__ == '__main__':
	# date = datetime.datetime.now().strftime('%H:%M:%S')
    # np.random.seed(0)
    # torch.manual_seed(0)
    test = True
    fl = True
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

    train(conf, fl = fl, test = test, rl = AgentDDPG(conf,  dir='./output/main_output/DDPG'), dir='./output/main_output/DDPG/')
    train(conf, fl = fl, test = test, rl = AgentPPO(conf, dir='./output/main_output/PPO'), dir='./output/main_output/PPO/')
    train(conf, fl = fl, test = test, rl = AgentSAC(conf, dir='./output/main_output/SAC'), dir='./output/main_output/SAC/')

    # train(conf, fl = fl, test = test, rl = DDPG_(conf,  dir='./output/main_output/Proposed'), dir='./output/main_output/Proposed')


    if True:
        rlvisual(patent = False, is_beam=config_train.IS_BEAM, ula_num=config_train.ULA_NUM)   
        # tra_visual(dir='./output/main_output/PPO/')
        # tra_visual(dir='./output/main_output/SAC/')
        # tra_visual(dir='./output/main_output/DDPG/')
        # tra_visual(dir='./output/main_output/Proposed')
        pass


        
    
	