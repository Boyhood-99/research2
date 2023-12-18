import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import datetime
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from configuration import *



font = {'family' : 'Times New Roman',
        'size'   : 12}
plt.rc('font', **font)


def data_dis_visual(df_list, patent = True, flag = None, diretory = f'./output/data_output/'):
    # df = pd.read_csv('./log07-05.csv')
    len_ = len(df_list[0][0])
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for i, (df, dir_alpha) in enumerate(df_list):
        glo_acc = list(df['global_accuracy'])
        glo_loss = list(df['global_loss'])

        axis1, = ax1.plot(glo_acc,  color= color_dic[f'{i+1}'], marker=marker_dict[f'{i+1}'],  label = f'$η$ = {dir_alpha}')
        axis2, = ax2.plot(glo_loss, color= color_dic[f'{i+1}'], marker=marker_dict[f'{i+1}'],  label = f'$η$ = {dir_alpha}')

    ax1.set_xlabel('Global epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(list(range(len_)))
    ax1.legend()

    ax2.set_xlabel('Global epoch')
    ax2.set_ylabel('Loss')
    ax2.set_xticks(list(range(len_)))
    ax2.legend()

    if patent:
        ax1.set_xlabel('全局轮次', fontproperties = 'SimHei')
        ax1.set_ylabel('准确度', fontproperties = 'SimHei')
        ax2.set_xlabel('全局轮次', fontproperties = 'SimHei')
        ax2.set_ylabel('损失', fontproperties = 'SimHei')

    if not os.path.exists(diretory):
        os.makedirs(diretory)  
    fig1.savefig(os.path.join(diretory, 'acc_.png'))
    fig1.savefig(os.path.join(diretory, 'acc_.pdf'))
    fig1.savefig(os.path.join(diretory, 'acc_.eps'))

    fig2.savefig(os.path.join(diretory, 'loss_.png'))
    fig2.savefig(os.path.join(diretory, 'loss_.pdf'))
    fig2.savefig(os.path.join(diretory, 'loss_.eps'))


def flvisual(df, date = None, patent = False, diretory = f'./output/main_output/FL'):
    if not os.path.exists(diretory):
        os.makedirs(diretory)
    # df = pd.read_csv('./log07-05.csv')
    SAC_glo_acc = list(df['SAC_acc'])
    SAC_glo_loss = list(df['SAC_loss'])
    DDPG_glo_acc = list(df['DDPG_acc'])
    DDPG_glo_loss = list(df['DDPG_loss'])

    fig1, ax1 = plt.subplots()
    if patent:
        axis1, = ax1.plot(SAC_glo_acc,  color= color_dic['1'], marker=marker_dict['1'], label = '所提算法', )
        axis2, = ax1.plot(DDPG_glo_acc, color= color_dic['2'], marker=marker_dict['2'], label = 'DDPG', )
        ax1.set_xlabel('全局轮次', fontproperties = 'SimHei')
        ax1.set_ylabel('准确度', fontproperties = 'SimHei')
        # ax1.legend()
        
        ax2 = ax1.twinx()
        axis3, = ax2.plot(SAC_glo_loss,  color= color_dic['3'], marker=marker_dict['3'], label = '所提算法')
        axis4, = ax2.plot(DDPG_glo_loss, color= color_dic['4'], marker=marker_dict['4'], label = 'DDPG')
        ax2.set_ylabel('损失', fontproperties = 'SimHei')
        # ax2.legend()
        plt.legend([axis1, axis2, axis3, axis4], ['模型准确度(所提算法)', '模型准确度(DDPG)',
                     '模型损失(所提算法)', '模型损失(DDPG)'], loc = 'center right', prop = 'SimHei')
    else:
        axis1, = ax1.plot(SAC_glo_acc,  color= color_dic['1'], marker=marker_dict['1'], label = 'test accuracy with proposed')
        axis2, = ax1.plot(DDPG_glo_acc, color= color_dic['2'], marker=marker_dict['2'], label = 'test accuracy with DDPG')
        axis1, = ax1.plot(SAC_glo_acc,  color= color_dic['1'], marker=marker_dict['1'], label = 'test accuracy with proposed')
        axis2, = ax1.plot(DDPG_glo_acc, color= color_dic['2'], marker=marker_dict['2'], label = 'test accuracy with DDPG')
        ax1.set_xlabel('Global rounds')
        ax1.set_ylabel('Accuracy')
        # ax1.legend()
        
        ax2 = ax1.twinx()
        axis3, = ax2.plot(SAC_glo_loss,  color= color_dic['3'], marker=marker_dict['3'], label = 'test loss with proposed')
        axis4, = ax2.plot(DDPG_glo_loss, color= color_dic['4'], marker=marker_dict['4'], label = 'test loss with DDPG')
        ax2.set_ylabel('Loss')
        # ax2.legend()
        plt.legend([axis1, axis2, axis3, axis4], ['test accuracy with proposed', 'test accuracy with DDPG',
                                                'test loss with proposed', 'test loss with DDPG'], loc = 'center right')
    
    
    fig1.savefig(os.path.join(diretory, f'{date}.png'))
    fig1.savefig(os.path.join(diretory, f'{date}.pdf'))
    fig1.savefig(os.path.join(diretory, f'{date}.eps'))


def rlvisual(is_smooth = False, fl = False, patent = True, is_beam = True, ula_num = 3,):
    
    #####     可视化FL
    if fl:
        df_fl_SAC  = pd.read_csv('./patent/SAC/acc_loss.csv')   if patent else pd.read_csv('./output/main_output/SAC/acc_loss.csv')
        df_fl_DDPG = pd.read_csv('./patent/DDPG/acc_loss.csv')  if patent else pd.read_csv('./output/main_output/DDPG/acc_loss.csv')

        date = datetime.datetime.now().strftime('%m-%d')
        df = pd.concat([df_fl_SAC[['global_accuracy', 'global_loss']], df_fl_DDPG[['global_accuracy', 'global_loss']]], axis=1,)
        df.to_csv(f'fl.csv')
        df.columns =  ['SAC_acc', 'SAC_loss',  'DDPG_acc', 'DDPG_loss']
        # ['SAC', 'SAC_acc', 'SAC_loss', 'DDPG', 'DDPG_acc', 'DDPG_loss']
        flvisual(df, date, patent = True)


    ######   return和energy 可视化

    return_ene_SAC  =  pd.read_csv('./patent/SAC/return_ene.csv') if patent else pd.read_csv(f'./output/main_output/SAC/return_ene{is_beam}{ula_num}.csv')
    return_ene_DDPG = pd.read_csv('./patent/DDPG/return_ene.csv') if patent else pd.read_csv(f'./output/main_output/DDPG/return_ene{is_beam}{ula_num}.csv')
    return_ene_PPO = pd.read_csv('./patent/PPO/return_ene.csv') if patent else pd.read_csv(f'./output/main_output/PPO/return_ene{is_beam}{ula_num}.csv')
    return_ene_Pro = pd.read_csv('./patent/PPO/return_ene.csv') if patent else pd.read_csv(f'./output/main_output/Proposed/return_ene{is_beam}{ula_num}.csv')

    return_ls_SAC  = return_ene_SAC['return']
    return_ls_DDPG = return_ene_DDPG['return']
    return_ls_PPO  = return_ene_PPO['return']
    return_ls_Pro  = return_ene_Pro['return']

    ene_consum_ls_SAC   =  return_ene_SAC['energy']
    ene_consum_ls_DDPG  =  return_ene_DDPG['energy']
    ene_consum_ls_PPO   =  return_ene_PPO['energy']
    ene_consum_ls_Pro   =  return_ene_Pro['energy']

    #####
    fig1, ax1 = plt.subplots()
    
    if patent:
        ax1.plot(return_ls_SAC,  linewidth = 1, linestyle='-',label='所提算法', )
        ax1.plot(return_ls_DDPG, linewidth = 1, linestyle='-',label='DDPG')
        ax1.plot(return_ls_PPO, linewidth = 1, linestyle='-',label='PPO')
        ax1.set_xlabel('回合', fontproperties='SimHei',)
        ax1.set_ylabel('回报', fontproperties='SimHei',)
        ax1.legend(loc = 'best', prop = {'family':'SimHei','size':14})

    else:
        if is_smooth:
            window_size = 100
            smoothed_DDPG = smooth([return_ls_DDPG, return_ls_DDPG, return_ls_DDPG], 19)
            
            ax1.plot(smoothed_DDPG, color = 'red' ,  linewidth = 1, linestyle='-',label='return with SAC')
            # ax1.plot(smoothed_SAC, color = 'blue', linewidth = 1, linestyle='-',label='return with DDPG')
            # ax1.plot(smoothed_PPO, color = 'green', linewidth = 1, linestyle='-',label='return with PPO')
            # ax1.plot(smoothed_Pro, color = 'black', linewidth = 1, linestyle='-',label='return with Proposed')

            ax1.plot(return_ls_DDPG, color = 'lightblue', linewidth = 1, linestyle='-',label='return with DDPG')
            # ax1.plot(return_ls_SAC, color = 'mistyrose', linewidth = 1, linestyle='-',label='return with SAC')
            # ax1.plot(return_ls_PPO, color = 'lightgreen', linewidth = 1, linestyle='-',label='return with PPO')
            # ax1.plot(return_ls_Pro, color = 'gray', linewidth = 1, linestyle='-',label='return with Proposed')
        else:
            # ax1.plot(return_ls_DDPG, color = 'blue', linewidth = 1, linestyle='-',label='return with DDPG')
            ax1.plot(return_ls_PPO, color = 'green', linewidth = 1, linestyle='-',label='return with PPO')
            ax1.plot(return_ls_Pro, color = 'red', linewidth = 1, linestyle='-',label='return with DDPG')
            ax1.plot(return_ls_SAC, color = 'lime' ,  linewidth = 1, linestyle='-',label='return with Proposed')
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Return')
        ax1.legend()
    
    fig1.savefig('./output/main_output/RL/return.jpg')
    fig1.savefig('./output/main_output/RL/return.eps')
    fig1.savefig('./output/main_output/RL/return.pdf')

    ########
    fig2, ax2 = plt.subplots()
    if patent:
        ax2.plot(ene_consum_ls_SAC,  linewidth=1, linestyle='-',label='所提算法')
        ax2.plot(ene_consum_ls_DDPG, linewidth=1, linestyle='-',label='DDPG')
        ax2.plot(ene_consum_ls_PPO,  linewidth=1, linestyle='-',label='PPO')
        ax2.set_xlabel('回合', fontproperties='SimHei',)
        ax2.set_ylabel('能耗（千焦）', fontproperties='SimHei',)
        ax2.legend(loc = 'best', prop = {'family':'SimHei','size':14})
    else:
        
        # ax2.plot(ene_consum_ls_DDPG, linewidth=1, linestyle='-',label='energy consumption with DDPG')
        ax2.plot(ene_consum_ls_PPO, color = 'green', linewidth=1, linestyle='-',label='energy consumption with PPO')
        ax2.plot(ene_consum_ls_Pro, color = 'red',  linewidth=1, linestyle='-',label='energy consumption with DDPG')
        ax2.plot(ene_consum_ls_SAC, color = 'lime',  linewidth=1, linestyle='-',label='energy consumption with Proposed')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Energy consumption (kJ)')
        ax2.legend()
        ax2.set_title()
    fig2.savefig('./output/main_output/RL/energy.jpg')
    fig2.savefig('./output/main_output/RL/energy.eps')
    fig2.savefig('./output/main_output/RL/energy.pdf')
    
    return 

def tra_visual(dir = f'./output/main_output/DDPG/'):

    df = pd.read_csv(os.path.join(dir, 'tra.csv'))
    h_uav = df['h_uav']
    h_uav_ls = []
    i = 0
    while i < len(h_uav):
        h_uav_ls.append(eval(h_uav[i]))
        i += 1
    h_uav_ls = list(map(list, zip(*h_uav_ls)))
    ##l-uav
    l_uavs = []
    for i in range(5):
        l_uav = df[f'l_uav{i}']
        l_uav_ls = []
        j = 0
        while j < len(l_uav):           
            l_uav_ls.append(eval(l_uav[j]))
            j += 1
        l_uav_ls = list(map(list, zip(*l_uav_ls)))
        
        l_uavs.append(l_uav_ls)
    ### plot h_uav
    fig = plt.figure(figsize=(5,5))
    fig = plt.figure(dpi = 200)
    ax = fig.add_subplot(projection='3d')
    
    ax.plot(h_uav_ls[0], h_uav_ls[1], h_uav_ls[2], marker=marker_dict['1'], markersize=2, label = 'The trajectory of H-UAV')
    for i in range(5):
        ax.plot(l_uavs[i][0], l_uavs[i][1], l_uavs[i][2], )

    ax.legend()
    ax.set_title('The trajectory of H-UAV and L-UAVs')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
        

    fig.savefig(os.path.join(dir, 'tra.jpg'))
    fig.savefig(os.path.join(dir, 'tra.pdf'))
    fig.savefig(os.path.join(dir, 'tra.eps'))


def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data
