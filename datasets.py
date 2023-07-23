from torchvision import datasets, transforms
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import partition_report, save_dict
#in-place 操作可能会覆盖计算梯度所需的值。

#每个 in-place 操作实际上都需要重写计算图的实现。out-of-place只是分配新对象并保留对旧计算图的引用，
# 而 in-place 操作则需要将所有输入的创建更改为代表此操作的函数。

#输出高度 = （输入高度 + 2 * 填充 - 卷积核高度）/ 步幅 + 1
#输出宽度 = （输入宽度 + 2 * 填充 - 卷积核宽度）/ 步幅 + 1
#默认的步幅（stride=1）和填充（padding=0）
#池化层
#输出特征图高度 = （输入特征图高度 - 池化窗口高度）/ 步幅 + 1
#输出特征图宽度 = （输入特征图宽度 - 池化窗口宽度）/ 步幅 + 1
class Dataset(object):
    def __init__(self, conf) -> None:
        self.conf = conf
        self.train_dataset, self.eval_dataset = self.get_dataset(self.conf['data_dir'], self.conf['type'])
        self.dataset_indice_list = self.get_indice()

    def get_dataset(self, dir, name):

        if name=='mnist':
            transform_train = transforms.Compose([ transforms.ToTensor(),  
                                transforms.Normalize((0.5,), (0.5,)) 
                                ]) 
            train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transform_train())
            eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
            
        elif name=='cifar10':
            transform_train = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            # transform_train = transforms.Compose([transforms.Resize((224, 224)),
            #                 transforms.RandomHorizontalFlip(p=0.5),
            #                 transforms.ToTensor(),
            #                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
            
            transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            train_dataset = datasets.CIFAR10(dir, train=True, download=True,transform=transform_train)
            eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
        
        return train_dataset, eval_dataset
    
    def get_indice(self,):
        num_clients = self.conf['f_uav_num']
        ######---------------------------------#客户端平分数据集
        
        #cifar10训练集每个data_batch,10000,其中十个类别是随机独立同分布,
        #DATA.sampler.SubsetRandomSampler用于从给定列表按照列表元素对应样本索引在数据集中抽取样本并
        # 进行打乱，比如抽取样本索引为[25，86，34，75],返回给loader可能会变为[34,86,25,75]
        #因此不需要shuffle进行打乱，因为已经打乱了

        dataset_indice_list = []
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / num_clients)
        for i in range(num_clients):
            dataset_indice = all_range[i * data_len: (i + 1) * data_len]
            dataset_indice_list.append(dataset_indice)

        ####利用fedlab生成数据分布------------------------------
        num_classes = 10
        seed = 2023
        #####Hetero Dirichlet
        # cifar10part = CIFAR10Partitioner(self.train_dataset.targets,
        #                                     num_clients,
        #                                     balance=None,
        #                                     partition="dirichlet",
        #                                     dir_alpha=0.3,
        #                                     seed=seed)
        ####基于shards的划分
        # num_shards = 200
        # cifar10part = CIFAR10Partitioner(self.train_dataset.targets,
        #                      num_clients,
        #                      balance=None,
        #                      partition="shards",
        #                      num_shards=num_shards,
        #                      seed=seed)       
        
        ###均衡IID
        # cifar10partt = CIFAR10Partitioner(self.train_dataset.targets,
        #                           num_clients,
        #                           balance=True,
        #                           partition="iid",
        #                           seed=seed)
        ###非均衡IID划分
        # cifar10part = CIFAR10Partitioner(self.train_dataset.targets,
        #                             num_clients,
        #                             balance=False,
        #                             partition="iid",
        #                             unbalance_sgm=0.3,
        #                             seed=seed)
        
        ####均衡dirichlet划分
        # cifar10part = CIFAR10Partitioner(self.train_dataset.targets,
        #                             num_clients,
        #                             balance=True,
        #                             partition="dirichlet",
        #                             dir_alpha=0.3,
        #                             seed=seed)
        
        ###非均衡dirichlet划分
        # cifar10part = CIFAR10Partitioner(self.train_dataset.targets,
        #                             num_clients,
        #                             balance=False,
        #                             partition="dirichlet",
        #                             unbalance_sgm=0.3,
        #                             dir_alpha=0.3,
        #                             seed=seed)

        #####
        # csv_file = f"./partition-reports/cifar10_hetero_dir_0.3_{num_clients}clients.csv"
        # partition_report(self.train_dataset.targets, cifar10part.client_dict,
        #                 class_num=num_classes,
        #                 verbose=False, file=csv_file)

        #######---------------------------
        # dataset_indice_list = []
        # for i in range(num_clients):
        #     dataset_indice = cifar10part.client_dict[i]
        #     # if i ==0:
        #     #     # print("client_dict[0]:",dataset_indice)
        #     #     pass
            
        #     dataset_indice_list.append(dataset_indice)

        ######---------------------------
        return dataset_indice_list



# labels = np.argmax(train_labels, axis=1)
# # 对数据标签进行排序

# order = np.argsort(labels)
# print("标签下标排序")
# print(train_labels[order[0:10]])
# self.train_data = train_images[order]
# self.train_label = train_labels[order]



	





from torchvision import transforms
from torchvision.datasets import CIFAR10 as cifar10
from torch.utils.data import random_split




def CIFAR10(params):
    mean_val = [0.4914, 0.4822, 0.4465]
    std_val = [0.2470, 0.2435, 0.2616]
    save_path = './data'

    random_transform1 = transforms.RandomHorizontalFlip(p=0.5)
    random_transform2 = transforms.Compose([transforms.Pad(padding=4),
                                            transforms.RandomCrop((32, 32))])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val,
                             std=std_val),
        transforms.RandomChoice([random_transform1, random_transform2]),

    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val,
                             std=std_val),
    ])

    train_dataset = cifar10(save_path,
                            train=True,
                            download=True,
                            transform=train_transform)

    test_dataset = cifar10(save_path,
                           train=False,
                           download=True,
                           transform=test_transform)

    train_size = int(params['split']['train'] * len(train_dataset))
    valid_size = int(params['split']['valid'] * len(train_dataset))
    train_other_size = len(train_dataset) - train_size - valid_size

    train_datasets = random_split(train_dataset, [train_size,
                                                  valid_size,
                                                  train_other_size])

    test_size = int(params['split']['test'] * len(test_dataset))
    test_other_size = len(test_dataset) - test_size

    test_datasets = random_split(test_dataset, [test_size,
                                                test_other_size])

    train_dataset, valid_dataset, _ = train_datasets
    test_dataset, _ = test_datasets

    return train_dataset, valid_dataset, test_dataset