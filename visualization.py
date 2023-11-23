import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import datetime
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

color_dict = {
    1: 'red',       # 红色
    2: 'green',     # 绿色
    3: 'blue',      # 蓝色
    4: 'orange',    # 橙色
    5: 'purple',    # 紫色
    6: 'cyan',      # 青色
    7: 'magenta',   # 洋红色
    8: 'yellow',    # 黄色
    9: 'black',     # 黑色
    10: 'gray',     # 灰色
    11: 'brown',    # 棕色
    12: 'pink',     # 粉色
    13: 'teal',     # 青绿色
    14: 'indigo',   # 靛蓝色
    15: 'lime',     # 酸橙色
    16: 'olive',    # 橄榄色
    17: 'gold',     # 金色
    18: 'silver',   # 银色
    19: 'navy',     # 海军蓝
    20: 'maroon',   # 褐红色
    # ... 可以继续添加更多的颜色
}

color1 = "#038355" # 孔雀绿
color2 = "#ffc34e" # 向日黄
color3 = "#66ce63" # 湖蓝
color4 = "#ec661f" #橘红
color_dic = {'1':color1, '2':color2, '3':color3, '4':color4}

marker_dict = {
    '1': 'o',      # 圆圈
    '2': 's',      # 正方形
    '3': '*',      # 星星
    '4': 'v',      # 下三角形
    '5': '>',      # 右三角形
    '6': '<',      # 左三角形
    '7': 'x',      # 叉叉
    '8': '^',      # 上三角形
    '9': '+',      # 加号
    '10': 'D',     # 菱形
    '11': 'p',     # 五边形
    '12': '|',     # 竖线
    '13': '_',     # 横线
    # ... 可以继续添加更多的标记
}


font = {'family' : 'Times New Roman',
        'size'   : 12}
plt.rc('font', **font)


def data_dis_visual(df_list, flag = None, diretory = f'./output/data_output/'):
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
    if not os.path.exists(diretory):
        os.makedirs(diretory)  
    fig1.savefig(os.path.join(diretory, 'acc_.png'))
    fig1.savefig(os.path.join(diretory, 'acc_.pdf'))
    fig1.savefig(os.path.join(diretory, 'acc_.eps'))

    fig2.savefig(os.path.join(diretory, 'loss_.png'))
    fig2.savefig(os.path.join(diretory, 'loss_.pdf'))
    fig2.savefig(os.path.join(diretory, 'loss_.eps'))


def flvisual(df, date, patent = False, diretory = f'./output/main_output/FL/'):
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


def rlvisual(patent = True, ):
    
    #####     可视化FL
    df_fl_SAC  = pd.read_csv('./patent/SAC/acc_loss.csv')   if patent else pd.read_csv('./output/main_output/SAC/acc_loss.csv')
    df_fl_DDPG = pd.read_csv('./patent/DDPG/acc_loss.csv')  if patent else pd.read_csv('./output/main_output/DDPG/acc_loss.csv')

    date = datetime.datetime.now().strftime('%m-%d')
    df = pd.concat([df_fl_SAC[['global_accuracy', 'global_loss']], df_fl_DDPG[['global_accuracy', 'global_loss']]], axis=1,)
    df.to_csv(f'fl.csv')
    df.columns =  ['SAC_acc', 'SAC_loss',  'DDPG_acc', 'DDPG_loss']
    # ['SAC', 'SAC_acc', 'SAC_loss', 'DDPG', 'DDPG_acc', 'DDPG_loss']
    flvisual(df, date, patent = True)

    ######   return和energy 可视化
    return_ene_SAC  =  pd.read_csv('./patent/SAC/return_ene.csv') if patent else pd.read_csv('./output/main_output/SAC/return_ene.csv')
    return_ene_DDPG = pd.read_csv('./patent/DDPG/return_ene.csv') if patent else pd.read_csv('./output/main_output/DDPG/return_ene.csv')
    return_ene_PPO = pd.read_csv('./patent/PPO/return_ene.csv') if patent else pd.read_csv('./output/main_output/PPO/return_ene.csv')

    return_ls_SAC  = return_ene_SAC['return']
    return_ls_DDPG = return_ene_DDPG['return']
    return_ls_PPO  = return_ene_PPO['return']

    ene_consum_ls_SAC   =  return_ene_SAC['energy']
    ene_consum_ls_DDPG  =  return_ene_DDPG['energy']
    ene_consum_ls_PPO   =  return_ene_PPO['energy']

    #####
    fig1, ax1 = plt.subplots()
    
    # plt.plot(episode_reward_sac_0, color='r', linewidth=1, linestyle='-',label='without beamforming(location)')
    # plt.plot(episode_reward_sac_1, color='g', linewidth=1, linestyle='-',label='without beamforming(distance)')
    if patent:
        ax1.plot(return_ls_SAC,  linewidth = 1, linestyle='-',label='所提算法', )
        ax1.plot(return_ls_DDPG, linewidth = 1, linestyle='-',label='DDPG')
        ax1.plot(return_ls_PPO, linewidth = 1, linestyle='-',label='PPO')
        ax1.set_xlabel('回合', fontproperties='SimHei',)
        ax1.set_ylabel('回报', fontproperties='SimHei',)
        ax1.legend(loc = 'best', prop = {'family':'SimHei','size':14})

    else:
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
    if patent:
        ax2.plot(ene_consum_ls_SAC,  linewidth=1, linestyle='-',label='所提算法')
        ax2.plot(ene_consum_ls_DDPG, linewidth=1, linestyle='-',label='DDPG')
        ax2.plot(ene_consum_ls_PPO, linewidth=1, linestyle='-',label='PPO')
        ax2.set_xlabel('回合', fontproperties='SimHei',)
        ax2.set_ylabel('能耗（千焦）', fontproperties='SimHei',)
        ax2.legend(loc = 'best', prop = {'family':'SimHei','size':14})
    else:
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


