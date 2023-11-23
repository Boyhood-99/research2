import torch
import json
import pandas as pd
import datetime
from uav import *
from torch.utils.tensorboard import SummaryWriter
from visualization import  flvisual, rlvisual
from configuration import ConfigDraw, ConfigTrain
from agent import AgentSAC, AgentDDPG, AgentPPO
import os


writer = SummaryWriter('./tensorboard/log/')

def train(conf, rl, dir):
    # np.random.seed(2023)
    # torch.manual_seed(2023)
    if not os.path.exists(dir):
        os.makedirs(dir)

    date1 = datetime.datetime.now().strftime('%m-%d')
    test = False
    
    rl = rl
    assert isinstance(rl, AgentSAC)
    return_ls = []
    ene_consum_ls = []
    for i in range(conf['config_train'].MAX_EPISODES):
        ##for test
        if test and i == conf['config_train'].MAX_EPISODES - 1:
            ep_reward, energy_consum, actions, acc_loss_test, tra_ls = rl.episode_test(i, conf['global_epochs'], )
            return_ls.append(ep_reward)
            ene_consum_ls.append(energy_consum/1000)
        else :
            ep_reward, energy_consum = rl.episode(i, conf['global_epochs'])
            if len(return_ls) >= 1 and ep_reward > max(return_ls):
                rl.save_model()
            if i%5 == 0:
                # print(len(rl.agent.replay_buffer.rewards))
                rl.update(update_times = 200)
            if i % 25 == 0:
                rl.std_decay()

            return_ls.append(ep_reward)
            ene_consum_ls.append(energy_consum/1000)
            print('\n')

    ######保存文件
    df_fl = None
    if test:
        df_actions = pd.DataFrame(actions)
        df_actions.to_csv(os.path.join(dir, f'actions.csv'))

        df_tra = pd.DataFrame(tra_ls)
        df_tra.to_csv(os.path.join(dir, f'tra.csv'))

        df_fl = pd.DataFrame(acc_loss_test)
        df_fl.to_csv(os.path.join(dir, f'acc_loss.csv'))
    ###
    ls_return_ene = [[i ,j] for i, j in zip(return_ls, ene_consum_ls)]  
    df_return_ene = pd.DataFrame(ls_return_ene, columns=['return', 'energy'])
    df_return_ene.to_csv(os.path.join(dir, f'return_ene.csv'))

    return return_ls, ene_consum_ls, df_fl   

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

    
    # train(conf, AgentPPO(conf, dir='./output/main_output/PPO'), dir='./output/main_output/PPO/')
    train(conf, AgentSAC(conf, dir='./output/main_output/SAC'), dir='./output/main_output/SAC/')
    train(conf, AgentDDPG(conf, dir='./output/main_output/DDPG/'), dir='./output/main_output/DDPG/')

    if True:
        rlvisual(patent = False)   

        
    
	