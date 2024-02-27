from dataclasses import dataclass
import torch
@dataclass
class ConfigDraw:
    Q_LOSS_FLAG = 0
    Q_VALUE_FLAG = 0

    DDPG_TRA_FLAG = 0
    SAC_TRA_FLAG = 0
    FIX_TRA_FLAG = 0

    ALGRITHM_FLAG = 1
    POLICY_FLAG = 0
    LEARNING_RATE = 0

    TIME_FLAG = 0
    DISTANCE_FLAG = 0

    TRA_COMPARISION = 0

    F_UAV_NUM_COM = 0
    ACCURACY_FLAG = 0

    BAR_FLAG = 0

@dataclass
class ConfigTrain:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_EPISODES = 400#500#1000
    BATCHSIZE = 256##256 ##256
    
    UAV_NUM = 5
    CONDIDATE = list(range(UAV_NUM))
    IS_NORM_MAN = False

    ###should larger than batchsize 
    WARM_UP = 1000#2000 
    BUFFER_SIZE = 7000#8000 #15000
    ###  beamforming
    IS_BEAM = True
    ULA_NUM = 3
    ### tra planning
    IS_CON_VEL = False
    IS_CON_DIR = False
    VEL = -1
    ## net para
    hidden_dim = 128 #32,64,
    lr_a = 3e-4#3e-4
    lr_c = 1e-3#1e-3
    #DDPG
    NOISE = 0.5


'''[20,30],[5,40],100000,200k'''
'''没有波束，速度10，方向0，0.83'''
'''没有波束，速度5，方向0，0.821'''
'''没有波束，速度自适应，方向0，0.870'''
'''没有波束，速度自适应，方向自适应，0.953'''
'''隐藏层64，没有波束，速度自适应，方向自适应，1.115'''
'''隐藏层256，没有波束，速度自适应，方向自适应，0.928'''


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
color5 = 'lime'
color6 = 'teal'
color_dic = {'1':color1, '2':color2, '3':color3, '4':color4, '5':color5, '6':color6}

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



    # model_name = "resnet50"
	# model_name : "resnet18"
	# model_name : "cnn"
	# type : cifar

	# "lr" : 0.001,
	
	# "momentum" : 0.0001,
	
	# "lambda" : 0.1,
	# "compile" : false,

	
	# "num_f_uav" : 5,
	
	# "global_epochs" : 10,
	
	# "local_epochs" : 2,
	# "#local_epochs" : 5,
	
	# "candidates" : [0,1,2,3,4],
	
	# "#batch_size" : 5,
	# "batch_size" : 32