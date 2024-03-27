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

def num_uav_fl(uav_num_ls = [5, 10, 20], patent = False, is_beam = True, ula_num = 3, m_num = 20, diretory = f'./output/main_output/SAC/'):

    fig1, ax1 = plt.subplots(dpi = 200)
    for i, uav_num in enumerate(uav_num_ls):
        if uav_num != 5:
            df_  = pd.read_csv(f'./output/main_output/SAC/acc_loss{is_beam}{ula_num}_{m_num}_{uav_num}.csv')
        else:
            df_  = pd.read_csv(f'./output/main_output/SAC/acc_loss{is_beam}{ula_num}_{m_num}.csv')

        axis1, = ax1.plot(df_['global_accuracy'], marker=marker_dict[f'{i+1}'], 
                            # color= color_dic[f'{i+1}'], 
                            label = f'K={uav_num}',
                            )
        len_ = len(df_['global_accuracy'])
    ax1.set_xticks(list(range(len_)))
    
    if patent:
        ax1.set_xlabel('全局轮次', fontproperties = 'SimHei')
        ax1.set_ylabel('准确率（%）', fontproperties = 'SimHei')
        plt.legend(prop = 'SimHei')
    else:
        ax1.set_xlabel('Communication rounds')
        ax1.set_ylabel('Accuracy (%)')
        plt.legend()
    
    date = datetime.datetime.now().strftime('%m-%d')
    fig1.savefig(os.path.join(diretory, f'{date}.png'), bbox_inches='tight', )
    fig1.savefig(os.path.join(diretory, f'{date}.pdf'), bbox_inches='tight', )
    fig1.savefig(os.path.join(diretory, f'{date}.eps'), bbox_inches='tight', )


def m_bar(m_num_ls = [10,20,40], patent = False,  bar = True):
    if bar:
        initial_point = 10#250
        beam_none = []
        beam_3 = []
        beam_5 = []
        beam_7 = []
        for m_num in m_num_ls:
            return_ene_none  =  pd.read_csv(f'./output/main_output/SAC/return_eneFalse3_{m_num}.csv')
            return_ene_3 =  pd.read_csv(f'./output/main_output/SAC/return_eneTrue3_{m_num}.csv')
            return_ene_5 =   pd.read_csv(f'./output/main_output/SAC/return_eneTrue5_{m_num}.csv')
            return_ene_7 =   pd.read_csv(f'./output/main_output/SAC/return_eneTrue7_{m_num}.csv')

            return_ls_none  = return_ene_none['return']
            return_ls_3 = return_ene_3['return']
            return_ls_5  = return_ene_5['return']
            return_ls_7  = return_ene_7['return']
            # return_ls_Pro  = return_ene_Pro['return']

            ene_consum_ls_none   =  return_ene_none['energy']#[:200]
            ene_consum_ls_3  =  return_ene_3['energy']#[:200]
            ene_consum_ls_5   =  return_ene_5['energy']#[:200]
            ene_consum_ls_7   =  return_ene_7['energy']#[:200]
            # ene_consum_ls_Pro   =  return_ene_Pro['energy']
            
            ene_none = np.array(sorted(ene_consum_ls_none)[:initial_point]).mean()
            ene_3 = np.array(sorted(ene_consum_ls_3)[:initial_point]).mean()
            ene_5 = np.array(sorted(ene_consum_ls_5)[:initial_point]).mean()
            ene_7 = np.array(sorted(ene_consum_ls_7)[:initial_point]).mean()
            if m_num == 20:
                ene_3 = np.array(sorted(ene_consum_ls_3)[:2*initial_point]).mean()
                ene_5 = np.array(sorted(ene_consum_ls_5)[:2*initial_point]).mean()
                ene_7 = np.array(sorted(ene_consum_ls_7)[:2*initial_point]).mean()
            beam_none.append(ene_none)
            beam_3.append(ene_3)
            beam_5.append(ene_5)
            beam_7.append(ene_7)
       
        
        m_num = ['10', '20', '40']
        xticks = np.arange(len(m_num))

        fig1, ax = plt.subplots(dpi = 200)

        # xticks1 = xticks + 0.2
        # xticks2 = xticks + 0.4
        
        xticks1 = xticks + 0.2
        xticks2 = xticks + 0.4
        xticks3 = xticks + 0.6
     
        width = 0.2

        if patent:
            
            ax.bar(xticks, beam_3,width = width, label = '${A_x} = {A_y} = {A_z} = 3$')
            ax.bar(xticks1, beam_5,width = width, label = '${A_x} = {A_y} = {A_z} = 5$')
            ax.bar(xticks2, beam_7,width = width, label = '${A_x} = {A_y} = {A_z} = 7$')
            ax.bar(xticks3,  beam_none, width = width, label = '全向天线(${G_a}==1$)')
            

            ax.set_xlabel('子信道数目M',  
                #    fontproperties = font,
                     fontproperties = "SimHei",
                     )
            ax.set_ylabel('能耗(kJ)', 
                        # fontproperties = font,
                    fontproperties = "SimHei",
                    )

            ax.legend(loc = 'best', prop = {'family':'SimHei'})
            ax.set_xticks(xticks + 0.3)
            ax.set_xticklabels(m_num)
        else:
            
            ax.bar(xticks, beam_3,    width = width, label = '${A_x} = {A_y} = {A_z} = 3$')
            ax.bar(xticks1, beam_5,    width = width, label = '${A_x} = {A_y} = {A_z} = 5$')
            ax.bar(xticks2, beam_7,    width = width, label = '${A_x} = {A_y} = {A_z} = 7$')
            ax.bar(xticks3,  beam_none, width = width, label = 'without 3DULA(${G_a} \equiv 1$)')

            ax.set_xlabel('The numbers of subchannels M')
            ax.set_ylabel('Energy Consumption (kJ)')

            ax.legend(loc = 'best', prop = {'size':9})

            ax.set_xticks(xticks + 0.3)
            ax.set_xticklabels(m_num)

#-----------------------------------------------------------
        # fig1.savefig('./output/main_output/SAC/ene_comp.jpg', bbox_inches='tight', transparent= True)
        fig1.savefig('./output/main_output/SAC/ene_comp.eps', bbox_inches='tight', transparent= True)
        fig1.savefig('./output/main_output/SAC/ene_comp.pdf', bbox_inches='tight', transparent= True)
        fig1.savefig('./output/main_output/SAC/ene_comp.png', bbox_inches='tight', )

def bar(ula_num_ls = [3,5,7], patent = False, is_beam = True,  m_num = 20, bar = True):
    if bar:
        initial_point = 20#250
        ppo_ls = []
        sac_ls = []
        ddpg_ls = []
        for ula_num in ula_num_ls:
            return_ene_SAC  =  pd.read_csv(f'./output/main_output/SAC/return_ene{is_beam}{ula_num}_{m_num}.csv')
            return_ene_DDPG =  pd.read_csv(f'./output/main_output/DDPG/return_ene{is_beam}{ula_num}_{m_num}.csv')
            return_ene_PPO =   pd.read_csv(f'./output/main_output/PPO/return_ene{is_beam}{ula_num}_{m_num}.csv')

            return_ls_SAC  = return_ene_SAC['return']
            return_ls_DDPG = return_ene_DDPG['return']
            return_ls_PPO  = return_ene_PPO['return']
            # return_ls_Pro  = return_ene_Pro['return']

            ene_consum_ls_SAC   =  return_ene_SAC['energy']#[:200]
            ene_consum_ls_DDPG  =  return_ene_DDPG['energy']#[:200]
            ene_consum_ls_PPO   =  return_ene_PPO['energy']#[:200]
            # ene_consum_ls_Pro   =  return_ene_Pro['energy']
            
            sac_ = np.array(sorted(ene_consum_ls_SAC)[:initial_point]).mean()
            ddpg_ = np.array(sorted(ene_consum_ls_DDPG)[:initial_point]).mean()
            ppo_ = np.array(sorted(ene_consum_ls_PPO)[:initial_point]).mean()
            sac_ls.append(sac_)
            ddpg_ls.append(ddpg_)
            ppo_ls.append(ppo_)
            
            print(sac_)
            print(ddpg_)
            print(ppo_)
       
        
        ula_num = ['3', '5', '7']
        xticks = np.arange(len(ula_num))

        fig1, ax = plt.subplots(dpi = 200)

        # xticks1 = xticks + 0.2
        # xticks2 = xticks + 0.4
        
        xticks1 = xticks + 0.3
        xticks2 = xticks + 0.6
     
        width = 0.3

        if patent:
            ax.bar(xticks,  sac_ls,width = width, label = '所提算法')
            ax.bar(xticks1, ddpg_ls,width = width, label = 'DDPG')
            ax.bar(xticks2,  ppo_ls,width = width, label = 'PPO')

            ax.set_xlabel('天线元件数目$\mathrm{A_x}$/$\mathrm{A_y}$/$\mathrm{A_z}$',  
                #    fontproperties = font,
                     fontproperties = "SimHei",
                     )
            ax.set_ylabel('能耗(kJ)', 
                        # fontproperties = font,
                    fontproperties = "SimHei",
                    )

            ax.legend(loc = 'best', prop = {'family':'SimHei'})
            ax.set_xticks(xticks + 0.3)
            ax.set_xticklabels(ula_num)
        else:
            ax.bar(xticks, sac_ls,width = width, label = 'Proposed')
            ax.bar(xticks1, ddpg_ls,width = width, label = 'DDPG')
            ax.bar(xticks2,  ppo_ls,width = width, label = 'PPO')

            ax.set_xlabel('The numbers of antenna elements $A_x$/$A_y$/$A_z$')
            ax.set_ylabel('Energy Consumption (kJ)')

            ax.legend(loc = 'best', )

            ax.set_xticks(xticks + 0.3)
            ax.set_xticklabels(ula_num)

#-----------------------------------------------------------
        fig1.savefig('./output/main_output/RL/ene_comp.jpg', bbox_inches='tight', transparent= True)
        fig1.savefig('./output/main_output/RL/ene_comp.eps', bbox_inches='tight', transparent= True)
        fig1.savefig('./output/main_output/RL/ene_comp.pdf', bbox_inches='tight', transparent= True)
        fig1.savefig('./output/main_output/RL/ene_comp.png', bbox_inches='tight', transparent= True)


def fl_alg_visual(fl_name_ls = None, dir_alpha = 0.3, patent = False, diretory = f'./output/FL_main_output'):
    fig1, ax1 = plt.subplots(dpi = 200)
    for i, fl_name in enumerate(fl_name_ls):
        df_  = pd.read_csv(f'./output/FL_main_output/{fl_name}{dir_alpha}.csv')
        axis1, = ax1.plot(df_['global_accuracy'],  color= color_dic[f'{i+1}'], marker=marker_dict[f'{i+1}'], 
                            label = fl_name,
                            )
        len_ = len(df_['global_accuracy'])
    ax1.set_xticks(list(range(len_)))
    
    if patent:
        ax1.set_xlabel('全局轮次', fontproperties = 'SimHei')
        ax1.set_ylabel('准确率（%）', fontproperties = 'SimHei')
        plt.legend(prop = 'SimHei')
    else:
        ax1.set_xlabel('Communication rounds')
        ax1.set_ylabel('Accuracy (%)')
        plt.legend()
    
    date = datetime.datetime.now().strftime('%m-%d')
    fig1.savefig(os.path.join(diretory, f'{date}.png'), bbox_inches='tight', )
    fig1.savefig(os.path.join(diretory, f'{date}.pdf'), bbox_inches='tight', )
    fig1.savefig(os.path.join(diretory, f'{date}.eps'), bbox_inches='tight', )

    
def data_dis_visual(df_list, patent = False, flag = None, diretory = f'./output/data_output/'):
    # df = pd.read_csv('./log07-05.csv')
    len_ = len(df_list[0][0])
    fig1, ax1 = plt.subplots(dpi=200)
    fig2, ax2 = plt.subplots(dpi=200)
    for i, (df, dir_alpha) in enumerate(df_list):
        glo_acc = list(df['global_accuracy'])
        glo_loss = list(df['global_loss'])
        if dir_alpha: 
            axis1, = ax1.plot(glo_acc,  color= color_dic[f'{i+1}'], marker=marker_dict[f'{i+1}'],  label = f'$η$ = {dir_alpha}')
            axis2, = ax2.plot(glo_loss, color= color_dic[f'{i+1}'], marker=marker_dict[f'{i+1}'],  label = f'$η$ = {dir_alpha}')
        else:
            axis1, = ax1.plot(glo_acc,  color= color_dic[f'{i+1}'], marker=marker_dict[f'{i+1}'],  label = f'IID')
            axis2, = ax2.plot(glo_loss, color= color_dic[f'{i+1}'], marker=marker_dict[f'{i+1}'],  label = f'IID')


    ax1.set_xticks(list(range(len_)))
    ax2.set_xticks(list(range(len_)))

    if patent:
        ax1.set_xlabel('全局轮次', fontdict={'family':'SimHei', })
        ax1.set_ylabel('准确率（%）', fontdict={'family':'SimHei', })
        ax2.set_xlabel('全局轮次', fontdict={'family':'SimHei', })
        ax2.set_ylabel('损失', fontdict={'family':'SimHei', })

        ax1.legend(prop={'family':'SimHei', })
        ax2.legend(prop={'family':'SimHei', })
    else:
        ax1.set_xlabel('Communication rounds')
        ax1.set_ylabel('Accuracy')
        ax2.set_xlabel('Communication rounds')
        ax2.set_ylabel('Loss')
        
        ax1.legend()
        ax2.legend()

    if not os.path.exists(diretory):
        os.makedirs(diretory)  
    date = datetime.datetime.now().strftime('%m-%d')
    fig1.savefig(os.path.join(diretory, f'{date}acc.png'), bbox_inches='tight', )
    fig1.savefig(os.path.join(diretory, f'{date}acc.pdf'), bbox_inches='tight', )
    fig1.savefig(os.path.join(diretory, f'{date}acc.eps'), bbox_inches='tight', )

    fig2.savefig(os.path.join(diretory, f'{date}loss.png'), bbox_inches='tight', )
    fig2.savefig(os.path.join(diretory, f'{date}loss.pdf'), bbox_inches='tight', )
    fig2.savefig(os.path.join(diretory, f'{date}loss.eps'), bbox_inches='tight', )


def flvisual(df, date = None, patent = False, diretory = f'./output/main_output/FL'):
    if not os.path.exists(diretory):
        os.makedirs(diretory)
    # df = pd.read_csv('./log07-05.csv')
    SAC_glo_acc = list(df['SAC_acc'])
    SAC_glo_loss = list(df['SAC_loss'])
    DDPG_glo_acc = list(df['DDPG_acc'])
    DDPG_glo_loss = list(df['DDPG_loss'])
    PPO_glo_acc = list(df['PPO_acc'])
    PPO_glo_loss = list(df['PPO_loss'])

    fig1, ax1 = plt.subplots(dpi=200)
    ax1.set_xticks(list(range(len(SAC_glo_acc))))
    if patent:
        axis1, = ax1.plot(SAC_glo_acc,  color= color_dic['1'], marker=marker_dict['1'], label = '所提算法', )
        axis2, = ax1.plot(DDPG_glo_acc, color= color_dic['2'], marker=marker_dict['2'], label = 'DDPG', )
        axis5, = ax1.plot(PPO_glo_acc, color= color_dic['5'], marker=marker_dict['5'], label = 'PPO', )
        ax1.set_xlabel('全局轮次', fontproperties = 'SimHei')
        ax1.set_ylabel('准确率（%）', fontproperties = 'SimHei')
        # ax1.legend()
        
        ax2 = ax1.twinx()
        axis3, = ax2.plot(SAC_glo_loss,  color= color_dic['3'], marker=marker_dict['3'], label = '所提算法')
        axis4, = ax2.plot(DDPG_glo_loss, color= color_dic['4'], marker=marker_dict['4'], label = 'DDPG')
        axis6, = ax2.plot(PPO_glo_loss, color= color_dic['6'], marker=marker_dict['6'], label = 'PPO')
        ax2.set_ylabel('损失', fontproperties = 'SimHei')
        # ax2.legend()
        plt.legend([axis1, axis2, axis5, axis3, axis4,  axis6], ['模型准确率(所提算法)', '模型准确率(DDPG)', '模型准确率(PPO)',
                     '模型损失(所提算法)', '模型损失(DDPG)', '模型损失(PPO)'], loc = 'center right', prop = 'SimHei')
    else:
        axis1, = ax1.plot(SAC_glo_acc,  color= color_dic['1'], marker=marker_dict['1'], label = 'model accuracy with Proposed')
        axis2, = ax1.plot(DDPG_glo_acc, color= color_dic['2'], marker=marker_dict['2'], label = 'model accuracy with DDPG')
        axis5, = ax1.plot(PPO_glo_acc, color= color_dic['5'], marker=marker_dict['5'], label = 'model accuracy with PPO', )
        ax1.set_xlabel('Communication rounds')
        ax1.set_ylabel('Accuracy')
        


        ax2 = ax1.twinx()
        axis3, = ax2.plot(SAC_glo_loss,  color= color_dic['3'], marker=marker_dict['3'], label = 'model loss with Proposed')
        axis4, = ax2.plot(DDPG_glo_loss, color= color_dic['4'], marker=marker_dict['4'], label = 'model loss with DDPG')
        axis6, = ax2.plot(PPO_glo_loss, color= color_dic['6'], marker=marker_dict['6'],  label = 'model loss with PPO')
        ax2.set_ylabel('Loss')
        
        plt.legend([axis1, axis2, axis5, axis3, axis4,  axis6], ['model accuracy with Proposed', 'model accuracy with DDPG',
        'model accuracy with PPO', 'model loss with Proposed', 'model loss with DDPG', 'model loss with PPO'], loc = 'center right')
    
    
    fig1.savefig(os.path.join(diretory, f'{date}.png'), bbox_inches='tight', )
    fig1.savefig(os.path.join(diretory, f'{date}.pdf'), bbox_inches='tight', )
    fig1.savefig(os.path.join(diretory, f'{date}.eps'), bbox_inches='tight', )


def rlvisual(is_smooth = False, fl = True, patent = False, is_beam = True, ula_num = 3, m_num = 20):
    
    #####   FL可视化
    if fl:
        df_fl_SAC  = pd.read_csv(f'./output/main_output/SAC/acc_loss{is_beam}{ula_num}_{m_num}.csv')
        df_fl_DDPG = pd.read_csv(f'./output/main_output/DDPG/acc_loss{is_beam}{ula_num}_{m_num}.csv')
        df_fl_PPO = pd.read_csv(f'./output/main_output/PPO/acc_loss{is_beam}{ula_num}_{m_num}.csv')

        date = datetime.datetime.now().strftime('%m-%d')
        df = pd.concat([df_fl_SAC[['global_accuracy', 'global_loss']], df_fl_DDPG[['global_accuracy', 'global_loss']], \
                        df_fl_PPO[['global_accuracy', 'global_loss']]], axis=1,)
        df.to_csv(f'fl.csv')
        df.columns =  ['SAC_acc', 'SAC_loss',  'DDPG_acc', 'DDPG_loss', 'PPO_acc', 'PPO_loss',]
        # ['SAC', 'SAC_acc', 'SAC_loss', 'DDPG', 'DDPG_acc', 'DDPG_loss']
        flvisual(df, date, patent = patent)


    ######   return和energy 可视化

    return_ene_SAC  =  pd.read_csv(f'./output/main_output/SAC/return_ene{is_beam}{ula_num}_{m_num}.csv')
    return_ene_DDPG =  pd.read_csv(f'./output/main_output/DDPG/return_ene{is_beam}{ula_num}_{m_num}.csv')
    return_ene_PPO =   pd.read_csv(f'./output/main_output/PPO/return_ene{is_beam}{ula_num}_{m_num}.csv')
    # return_ene_Pro =   pd.read_csv(f'./output/main_output/Proposed/return_ene{is_beam}{ula_num}_{m_num}.csv')

    return_ls_SAC  = return_ene_SAC['return']
    return_ls_DDPG = return_ene_DDPG['return']
    return_ls_PPO  = return_ene_PPO['return']
    # return_ls_Pro  = return_ene_Pro['return']

    ene_consum_ls_SAC   =  return_ene_SAC['energy']
    ene_consum_ls_DDPG  =  return_ene_DDPG['energy']
    ene_consum_ls_PPO   =  return_ene_PPO['energy']
    # ene_consum_ls_Pro   =  return_ene_Pro['energy']

    #####    return
    fig1, ax1 = plt.subplots(dpi=200)
    
    if patent:
        ax1.plot(return_ls_PPO, color = 'green', linewidth = 1, linestyle='-', label='PPO')
        ax1.plot(return_ls_DDPG, color = 'red', linewidth = 1, linestyle='-', label='DDPG')
        ax1.plot(return_ls_SAC, color = 'lime' ,  linewidth = 1, linestyle='-', label='所提算法')
        # ax1.annotate('DDPG', xy=(130, return_ls_Pro[130]), xytext=(200, -20), arrowprops=dict(arrowstyle = '->', ),   font = 'SimHei')
        # ax1.annotate('所提算法', xy=(120, return_ls_SAC[120]), xytext=(0, -10), arrowprops=dict(arrowstyle = '->', ), font = 'SimHei')
        ax1.set_xlabel('回合', fontproperties='SimHei',)
        ax1.set_ylabel('回报', fontproperties='SimHei',)
        ax1.legend(loc = 'best', prop = {'family':'SimHei', })

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
            ax1.plot(return_ls_DDPG, color = 'blue', linewidth = 1, linestyle='-',label='DDPG')
            ax1.plot(return_ls_PPO, color = 'green', linewidth = 1, linestyle='-',label='PPO')
            # ax1.plot(return_ls_Pro, color = 'red', linewidth = 1, linestyle='-',label='return with DDPG')
            ax1.plot(return_ls_SAC, color = 'lime' ,  linewidth = 1, linestyle='-',label='Proposed')
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Return')
        ax1.legend()
    
    fig1.savefig('./output/main_output/RL/return.jpg', bbox_inches='tight', )
    fig1.savefig('./output/main_output/RL/return.eps', bbox_inches='tight', )
    fig1.savefig('./output/main_output/RL/return.pdf', bbox_inches='tight', )
    fig1.savefig('./output/main_output/RL/return.png', bbox_inches='tight', )

    ########  energy
    fig2, ax2 = plt.subplots(dpi=200)
    if patent:
        ax2.plot(ene_consum_ls_PPO, color = 'green', linewidth=1, linestyle='-',label='PPO')
        ax2.plot(ene_consum_ls_DDPG, color = 'red',  linewidth=1, linestyle='-',label='DDPG')
        ax2.plot(ene_consum_ls_SAC, color = 'lime',  linewidth=1, linestyle='-',label='所提算法')
        ax2.set_xlabel('回合', fontproperties='SimHei',)
        ax2.set_ylabel('能耗（千焦）', fontproperties='SimHei',)
        ax2.legend(loc = 'best', prop = {'family':'SimHei','size':14})
    else:
        # ax2.plot(ene_consum_ls_Pro, color = 'red',  linewidth=1, linestyle='-',label='energy consumption with DDPG')
        # ax2.plot(ene_consum_ls_DDPG, color = 'red', linewidth=1, linestyle='-',label='energy consumption with DDPG')
        ax2.plot(ene_consum_ls_PPO, color = 'green', linewidth=1, linestyle='-',label='energy consumption with PPO')
        # ax2.plot(ene_consum_l
        #  color = 'lime',  linewidth=1, linestyle='-',label='energy consumption with Proposed')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Energy consumption (kJ)')
        ax2.legend()
        # ax2.set_title()
    fig2.savefig('./output/main_output/RL/energy.jpg')
    fig2.savefig('./output/main_output/RL/energy.eps')
    fig2.savefig('./output/main_output/RL/energy.pdf')
    fig2.savefig('./output/main_output/RL/energy.png')

    
    return 


def tra_visual(patent = False, is_beam = True, ula_num = 3, m_num = 20, dir = f'./output/main_output/DDPG/'):

    df = pd.read_csv(os.path.join(dir, f'tra{is_beam}{ula_num}_{m_num}.csv'))
    # df = pd.read_csv(os.path.join(dir, f'tra_0.001_5.csv'))
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
    # fig = plt.figure(figsize=(5,5))
    fig = plt.figure(dpi = 200, 
                    # figsize=(4, 5)
                     )
    ax = fig.add_subplot(projection='3d')
    
    
    if patent:
        ax.plot(h_uav_ls[0], h_uav_ls[1], h_uav_ls[2], marker=marker_dict['1'], markersize=2, label = '顶层无人机轨迹')
        ax.legend(loc = 'best', prop={'family':'SimHei'})
    else:
        ax.plot(h_uav_ls[0], h_uav_ls[1], h_uav_ls[2], marker=marker_dict['1'], markersize=2, label = 'The trajectory of H-UAV')
        ax.legend()
    for i in range(5):
        ax.plot(l_uavs[i][0], l_uavs[i][1], l_uavs[i][2], )

    # ax.set_title('顶层无人机和底层无人机的轨迹')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
        
    # fig.tight_layout()
    # plt.subplots_adjust(bottom=0.1,left=0.01)

    fig.savefig(os.path.join(dir, 'tra.jpg'), 
                # bbox_inches='tight'
                )
    fig.savefig(os.path.join(dir, 'tra.pdf'), 
                bbox_inches='tight', 
                pad_inches=0.2
                )
    fig.savefig(os.path.join(dir, 'tra.eps'), 
                # bbox_inches='tight'
                )
    fig.savefig(os.path.join(dir, 'tra.png'), 
                # bbox_inches='tight'
                )
    # print(fig.get_figwidth())


def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data
