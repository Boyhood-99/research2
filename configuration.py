from dataclasses import dataclass
import torch
@dataclass
class ConfigDraw:
    Q_LOSS_FLAG=0
    Q_VALUE_FLAG=0

    DDPG_TRA_FLAG=0
    SAC_TRA_FLAG=0
    FIX_TRA_FLAG=0

    ALGRITHM_FLAG=1
    POLICY_FLAG=0
    LEARNING_RATE=0

    TIME_FLAG=0
    DISTANCE_FLAG=0

    TRA_COMPARISION=0

    F_UAV_NUM_COM=0
    ACCURACY_FLAG=0

    BAR_FLAG=0

@dataclass
class ConfigTrain:
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_EPISODES=50
    BATCHSIZE=256
    

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
	