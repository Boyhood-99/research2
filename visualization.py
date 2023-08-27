import matplotlib.pyplot as plt
import os
import numpy as np
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


def data_dis_visual(df_list, flag = None, diretory = f'./data_dis/'):
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


def flvisual(df, date, diretory = f'./FL/'):
    if not os.path.exists(diretory):
        os.makedirs(diretory)
    # df = pd.read_csv('./log07-05.csv')
    SAC_glo_acc = list(df['SAC_acc'])
    SAC_glo_loss = list(df['SAC_loss'])
    DDPG_glo_acc = list(df['DDPG_acc'])
    DDPG_glo_loss = list(df['DDPG_loss'])

    fig1, ax1 = plt.subplots()
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
                                               'test loss with proposed', 'test loss with DDPG'])
    
    
    fig1.savefig(os.path.join(diretory, f'{date}.png'))
    fig1.savefig(os.path.join(diretory, f'{date}.pdf'))
    fig1.savefig(os.path.join(diretory, f'{date}.eps'))


def rlvisual_prev(f_uav_num, config_draw, reward_ls):
    '''this function is deprecated'''
    
    
    # episode_reward_fix,t_comm,t_total,l_uav_location_f,f_uav_location_f,d_fix=fixaction()
    # episode_reward_sto,t_comm,t_total,l_uav_location,f_uav_location,d_sto=stoaction()
    # episode_reward_sac_fe,q_fe,_,_,_,_,_,_,_=sac_train_federated()
    
    # episode_reward_sac_tra15,_,_,_,_,_,_,_,_=sac_train_trajectory()#15
    # episode_reward_sac_tra10,_,_,_,_,_,_,_,_=sac_train_trajectory(I=-11/29)#10


    # episode_reward_sac,q,p,alpha,t_comm_sac,t_total_sac,l_uav_location_sac,f_uav_location_sac,d_sac,total_time_sac=sac_train()
    # episode_reward_sac_0,_,_,_,_,_,_,_,_,_=sac_train(env=Environment0())##state space based on location
    # episode_reward_sac_1,_,_,_,_,_,_,_,_,_=sac_train(env=Environment1())##state space based on distance
    # episode_reward_sac_2,_,_,_,_,_,_,_,_,_=sac_train(env=Environment2())##state space based on 3D location and phi
   
    # episode_reward_ddpg,a_loss,td_error,t_comm_ddpg,t_total_ddpg,l_uav_location_ddpg,f_uav_location_ddpg,d_ddpg=ddpg_train(env=Environment2())

   

    #学习率
    # episode_reward_sac_1e_4,_,_,_,_,_,_,_,_=sac_train(policy_lr = 1e-4,path='./SAC/policy_sac_model')
    # episode_reward_sac_3e_6,_,_,_,_,_,_,_,_=sac_train(policy_lr = 3e-6,path='./SAC/policy_sac_model')
    # episode_reward_sac_1e_5,_,_,_,_,_,_,_,_=sac_train(policy_lr = 1e-5,path='./SAC/policy_sac_model')


    #无人机数量
    # episode_reward_sac_10,q,p,alpha,t_comm_sac,t_total_sac,l_uav_location_sac,f_uav_location_sac,d_sac=sac_train(env=Environment(f_uav_num=10),f_uav_num=10)
    # episode_reward_sac_15,q,p,alpha,t_comm_sac,t_total_sac,l_uav_location_sac,f_uav_location_sac,d_sac=sac_train(env=Environment(f_uav_num=15),f_uav_num=15)
    # episode_reward_sac_20,q,p,alpha,t_comm_sac,t_total_sac,l_uav_location_sac,f_uav_location_sac,d_sac=sac_train(env=Environment(f_uav_num=20),f_uav_num=20)


    #模型精度
    # episode_reward_sac_001,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.01,time_max=50,end_reward=15))
    # episode_reward_sac_0005,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.005,time_max=60,end_reward=17.5))
    # episode_reward_sac_0001,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.001,time_max=60,end_reward=22))

# accuracy,number
    # episode_reward_sac_5_01,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.1))
    # episode_reward_sac_5_001,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.01))
    # episode_reward_sac_5_0005,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(epsilon=0.005))#基准设置

    # episode_reward_sac_10_01,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=10,epsilon=0.1))
    # episode_reward_sac_10_001,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=10,epsilon=0.01))
    # episode_reward_sac_10_0005,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=10,epsilon=0.005,x_range=[-20,3000],y_range=[-200,200]))
    # episode_reward_sac_10_0001,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=10,x_range=[-20,3000],y_range=[-200,200]))

    # episode_reward_sac_20_01,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=20,epsilon=0.1,x_range=[-20,1000],y_range=[-200,200]))
    # episode_reward_sac_20_001,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=20,epsilon=0.01,x_range=[-20,2000],y_range=[-200,200]),capacity=10000)
    # episode_reward_sac_20_0005,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=20,epsilon=0.005,x_range=[-20,3000],y_range=[-200,200]),capacity=12000)
    # episode_reward_sac_20_0001,_,_,_,_,_,_,_,_,_=sac_train(env=Environment(f_uav_num=20,x_range=[-20,4000],y_range=[-200,200]),capacity=15000)

    if config_draw.FIX_TRA_FLAG:
        l_uav_location_f=np.array(l_uav_location_f)     #.reshape(-1,2) #第0维
        

        # f_uav_location_sac=f_uav_location_sac[initial_location:initial_location+trajectory_length]
        f_uav_location_f=np.array(f_uav_location_f)  #.reshape(-1,f_uav_num,2) 

    #SAC trajectory

        plt.figure(20)
        ax = plt.axes(projection='3d')

        l_uav_location_f_x=l_uav_location_f[:,0]
        l_uav_location_f_y=l_uav_location_f[:,1]

        ax.plot3D(l_uav_location_f_x, l_uav_location_f_y, 150, label='The trajectory of T-UAV')
        ax.legend()

        for i in range(f_uav_num):
            f_uav_location_f_x=f_uav_location_f[:,i,0]
            f_uav_location_f_y=f_uav_location_f[:,i,1]
            ax.plot3D(f_uav_location_f_x, f_uav_location_f_y, 140 )
        ax.set_title('3D line plot')
        plt.savefig('./SAC/path_f.jpg')
        plt.savefig('./SAC/path_f.eps')
        plt.savefig('./SAC/path_f.pdf')
    
    if config_draw.BAR_FLAG:
        initial_point=350
        #柱状图
        epsilon01=[np.array(episode_reward_sac_5_01[initial_point:]).mean(),np.array(episode_reward_sac_10_01[initial_point:]).mean(),]
                   #np.array(episode_reward_sac_20_01[initial_point:]).mean()]
        epsilon001=[np.array(episode_reward_sac_5_001[initial_point:]).mean(),np.array(episode_reward_sac_10_001[initial_point:]).mean(),]
                    # np.array(episode_reward_sac_20_001[initial_point:]).mean()]
        epsilon0005=[np.array(episode_reward_sac_5_0005[initial_point:]).mean(),np.array(episode_reward_sac_10_0005[initial_point:]).mean(),]
                    #  np.array(episode_reward_sac_20_0005[initial_point:]).mean()]
        epsilon0001=[np.array(episode_reward_sac[initial_point:]).mean(),np.array(episode_reward_sac_10_0001[initial_point:]).mean(),]
                    #  np.array(episode_reward_sac_20_0001[initial_point:]).mean()]

        # num_f_uav_5=[epsilon01[0],epsilon001[0],epsilon0005[0],epsilon0001[0]]
        # num_f_uav_10=[epsilon01[1],epsilon001[1],epsilon0005[1],epsilon0001[1]]
        
        num_f_uav=['K=5','K=10']
        xticks=np.arange(len(num_f_uav))

        fig,ax=plt.subplots(dpi=200)#画布大小，分辨率；
        
        width=0.2
        ax.bar(xticks,epsilon01,width=width,label='ε=0.1')
        ax.bar(xticks+0.2,epsilon001,width=width,label='ε=0.01')
        ax.bar(xticks+0.4,epsilon0005,width=width,label='ε=0.005')
        ax.bar(xticks+0.6,epsilon0001,width=width,label='ε=0.001')

        ax.set_xlabel("Number of L-UAVs")
        ax.set_ylabel("Total energy(kJ)")
        
        # plt.rcParams.update({'font.size': 15})
        ax.legend()
        ax.set_xticks(xticks+0.3)
        ax.set_xticklabels(num_f_uav)

        # x = [5,10,20]
        # x1=np.array([i for i in range(0,15,5)])
        # #将每四个柱状图之间空一格
        # x2=x1+1
        # x3=x1+2
        # x4=x1+3
        # x5=x1+4
        
        # y2 = epsilon01
        # y3 = epsilon001
        # y4 = epsilon0005
        # y5 = epsilon0001
        
        # plt.bar(x1,y2,width=width,label='SAC(ε=0.1)')
        
        # plt.bar(x2,y3,width=width,label='SAC(ε=0.01)')
        
        # plt.bar(x3,y4,width=width,label='SAC(ε=0.005)')
        # plt.bar(x4,y5,width=width,label='SAC(ε=0.001)')
        # plt.bar(x5,0,width=width) #空格一个
        
        # plt.xlabel('Number of B-UAVs')
        # plt.ylabel('Total Energy(kJ)')
        # plt.legend()
        # plt.xticks(x1+1.5,x,rotation = 45)#+1.5是让下标在四个柱子中间
      
        
        '''#每一个柱上添加相应值
        for a,b,c,d,e,f,g,h in zip(x1,x2,x3,x4,y2,y3,y4,y5):
            plt.text(a,e+100,int(e),fontsize=4,ha='center')
            plt.text(b,f+100,int(f),fontsize=4,ha='center')
            plt.text(c,g+100,int(g),fontsize=4,ha='center')
            plt.text(d,h+100,int(h),fontsize=4,ha='center')'''
        
        

        plt.savefig('./SAC/bar.eps')
        plt.savefig('./SAC/bar.pdf')
        plt.savefig('./SAC/bar.jpg')


    if config_draw.ACCURACY_FLAG:
        plt.figure(14,dpi=200)

        plt.plot(episode_reward_sac,color='r', linewidth=1, linestyle='-',label='SAC(ε=0.1)')
        plt.plot(episode_reward_sac_001,color='y', linewidth=1, linestyle='-',label='SAC(ε=0.01)')
        # plt.plot(episode_reward_sac_15,color='b', linewidth=1, linestyle='-',label='SAC(B-UAVs=15)')
        plt.plot(episode_reward_sac_0001,color='g', linewidth=1, linestyle='-',label='SAC(ε=0.001)')

        plt.xlabel('Episodes')
        plt.ylabel('Total energy(kJ)')
        plt.legend()

        plt.savefig('./SAC/accuracy.jpg')


    if config_draw.F_UAV_NUM_COM:
        plt.figure(13,dpi=200)

        plt.plot(episode_reward_sac,color='r', linewidth=1, linestyle='-',label='SAC(B-UAVs=5)')
        plt.plot(episode_reward_sac_10,color='y', linewidth=1, linestyle='-',label='SAC(B-UAVs=10)')
        # plt.plot(episode_reward_sac_15,color='b', linewidth=1, linestyle='-',label='SAC(B-UAVs=15)')
        plt.plot(episode_reward_sac_20,color='g', linewidth=1, linestyle='-',label='SAC(B-UAVs=20)')

        plt.xlabel('Episodes')
        plt.ylabel('Total Energy(kJ)')
        plt.legend()


        plt.savefig('./SAC/num.jpg')

    
    if config_draw.LEARNING_RATE:
        plt.figure(12,dpi=200)
        plt.plot(episode_reward_sac,color='b', linewidth=1, linestyle='-',label='SAC,lr_a=3e-5')
        # plt.plot(episode_reward_sac_1e_4,color='r', linewidth=1, linestyle='-',label='SAC,lr_a=1e-4')
        plt.plot(episode_reward_sac_1e_5,color='g', linewidth=1, linestyle='-',label='SAC,lr_a=1e-5')
        plt.plot(episode_reward_sac_3e_6,color='y', linewidth=1, linestyle='-',label='SAC,lr_a=3e-6')

        plt.xlabel('Episodes')
        plt.ylabel('Total Energy(kJ)')
        plt.legend()

        plt.savefig('./SAC/lr_a.jpg')

    if config_draw.ALGRITHM_FLAG:
        plt.figure(1,dpi=200)
       

        # plt.plot(episode_reward_fix,color='g', linewidth=1, linestyle='-',label='fixed')
        # plt.plot(episode_reward_sac,color='b', linewidth=1, linestyle='-',label='with beamforming')
        # plt.plot(episode_reward_ddpg,color='r', linewidth=1, linestyle='-',label='DDPG')

        # plt.plot(episode_reward_sac_0, color='r', linewidth=1, linestyle='-',label='without beamforming(location)')
        # plt.plot(episode_reward_sac_1, color='g', linewidth=1, linestyle='-',label='without beamforming(distance)')
        plt.plot(reward_ls, color='g', linewidth=1, linestyle='-',label='with beamforming(location and phi)')

        plt.xlabel('Episodes')
        plt.ylabel('Total reward')
        plt.legend()
        #----------------------------------------
        # ax = plt.gca()
        # # ax_zoom = plt.axes([0.6, 0.55, 0.3, 0.3])  # 调整放大图的位置和大
        # ax_zoom = inset_axes(ax, width="20%", height="20%", loc='lower left',
        #             bbox_to_anchor=(0.5, 0.4, 1, 1),
        #             bbox_transform=ax.transAxes,
                   
        #            )

       
        # ax_zoom.plot(episode_reward_sac_2)

        
        # ax_zoom.set_xlim(10, 40)
        # ax_zoom.set_ylim(episode_reward_sac_2[8], episode_reward_sac_2[42])
        # mark_inset(ax, ax_zoom, loc1=3, loc2=1, fc="none", ec="0.5", )
        #-------------------------------------------------
        plt.savefig('./SAC/ddpg+sac.jpg')
        plt.savefig('./SAC/ddpg+sac.eps')
        plt.savefig('./SAC/ddpg+sac.pdf')
    if config_draw.POLICY_FLAG:
        plt.figure(2,dpi=200)
        plt.plot(episode_reward_sto,color='y', linewidth=1, linestyle='-',label='stocastic')
        plt.plot(episode_reward_sac,color='b', linewidth=1, linestyle='-',label='SAC')
        plt.plot(episode_reward_sac_tra10,color='r', linewidth=1, linestyle='-',label='SAC_tra(I=10)')
        plt.plot(episode_reward_sac_tra15,color='g', linewidth=1, linestyle='-',label='SAC_tra(I=15)')

        plt.xlabel('Episodes')
        plt.ylabel('Total reward(kJ)')
        plt.legend()
        plt.savefig('./SAC/policy.jpg')
        plt.savefig('./SAC/policy.eps')
        plt.savefig('./SAC/policy.pdf')

    #DDPG and SAC Q_loss
    if config_draw.Q_LOSS_FLAG:

        plt.figure(3)
        plt.plot(q,color='b',linewidth=1, linestyle='-',label='loss_SAC')
        plt.plot(td_error,color='r', linewidth=1, linestyle='-',label='loss_DDPG')
        
        plt.xlabel('Steps')
        plt.ylabel('the Q loss of SAC and DDPG')
        plt.legend()
        plt.savefig('./SAC/loss.jpg')

   
    if config_draw.Q_VALUE_FLAG:
        #sac
        # plt.figure(2)
        # plt.plot(entropy)
        
        plt.figure(6)
        plt.plot(p)
        plt.figure(7)
        plt.plot(alpha)

        #DDPG
        plt.figure(8)
        plt.plot(a_loss)
    
    if config_draw.DDPG_TRA_FLAG:

         #trajectory
        initial_location=0
        trajectory_length=100

        # l_uav_location_ddpg=l_uav_location_ddpg[initial_location:initial_location+trajectory_length]
        l_uav_location_ddpg=np.array(l_uav_location_ddpg)
        

        # f_uav_location_ddpg=f_uav_location_ddpg[initial_location:initial_location+trajectory_length]
        f_uav_location_ddpg=np.array(f_uav_location_ddpg)

    #DDPG trajectory
        plt.figure(9)
        ax1 = plt.axes(projection='3d')

        l_uav_location_ddpg_x=l_uav_location_ddpg[:,0]
        l_uav_location_ddpg_y=l_uav_location_ddpg[:,1]

        ax1.plot3D(l_uav_location_ddpg_x, l_uav_location_ddpg_y, 150, )

        for i in range(f_uav_num):
            f_uav_location_ddpg_x=f_uav_location_ddpg[:,i,0]
            f_uav_location_ddpg_y=f_uav_location_ddpg[:,i,1]
            ax1.plot3D(f_uav_location_ddpg_x, f_uav_location_ddpg_y, 140 )
        ax1.set_title('3D line plot')
        plt.savefig('./SAC/path_ddpg.eps')

    if config_draw.SAC_TRA_FLAG:

      
    
        #for SAC
        # l_uav_location_sac=l_uav_location_sac[initial_location:initial_location+trajectory_length]
        l_uav_location_sac=np.array(l_uav_location_sac)     #.reshape(-1,2) #第0维
        

        # f_uav_location_sac=f_uav_location_sac[initial_location:initial_location+trajectory_length]
        f_uav_location_sac=np.array(f_uav_location_sac)  #.reshape(-1,f_uav_num,2) 

    #SAC trajectory

        plt.figure(10)
        ax = plt.axes(projection='3d')

        l_uav_location_sac_x=l_uav_location_sac[:,0]
        l_uav_location_sac_y=l_uav_location_sac[:,1]

        ax.plot3D(l_uav_location_sac_x, l_uav_location_sac_y, 150, label='The trajectory of T-UAV')
        ax.legend()

        for i in range(f_uav_num):
            f_uav_location_sac_x=f_uav_location_sac[:,i,0]
            f_uav_location_sac_y=f_uav_location_sac[:,i,1]
            ax.plot3D(f_uav_location_sac_x, f_uav_location_sac_y, 140 )
        ax.set_title('3D line plot')
        plt.savefig('./SAC/path_sac.jpg')
        plt.savefig('./SAC/path_sac.eps')
        plt.savefig('./SAC/path_sac.pdf')


    if config_draw.DISTANCE_FLAG:
        plt.figure(11,dpi=200)

        # plt.plot(d_sto,color='y', linewidth=1, linestyle='-',label='stocastic')
        # plt.plot(d_fix,color='g', linewidth=1, linestyle='-',label='fixed')
        plt.plot(d_sac,color='b', linewidth=1, linestyle='-',label='SAC(ε=0.001)')
        # plt.plot(d_ddpg,color='r', linewidth=1, linestyle='-',label='DDPG')

        plt.xlabel('Episodes')
        plt.ylabel('Maximum distance(m)')
        plt.legend()

        plt.savefig('./SAC/distance.jpg')
        plt.savefig('./SAC/distance.eps')
        plt.savefig('./SAC/distance.pdf')
    
    if config_draw.TIME_FLAG :
        # plt.figure(4)
        # plt.plot(t_comm_ddpg,color='r',linewidth=1, linestyle='-',label='ddpg_t_comm')
        # plt.plot(t_total_ddpg,color='b',linewidth=1, linestyle='-',label='ddpg_t_total')
        # plt.legend()
        # plt.savefig('./SAC/ddpg.jpg')

        # plt.figure(5)
        # plt.plot(t_comm_sac,color='r',linewidth=1, linestyle='-',label='sac_t_comm')
        # plt.plot(t_total_sac,color='b',linewidth=1, linestyle='-',label='sac_t_total')
        # plt.legend()
        # plt.savefig('./SAC/sac.jpg')

        plt.figure(20)
        plt.plot(total_time_sac,color='r',linewidth=1, linestyle='-',label='total_time_sac')
        # plt.plot(t_total_sac,color='b',linewidth=1, linestyle='-',label='sac_t_total')
        plt.xlabel('Episodes')
        plt.ylabel('FL total time(s)')
        plt.legend()
    

        plt.savefig('./SAC/sac.jpg')
        plt.savefig('./SAC/sac.pdf')
        plt.savefig('./SAC/sac.eps')



    plt.show()