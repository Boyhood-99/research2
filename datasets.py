from torchvision import datasets, transforms
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import partition_report #save_dict
import numpy as np
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
    def __init__(self, conf, dir_alpha = 0.3) -> None:
        self.conf = conf
        self.train_dataset, self.eval_dataset = self.get_dataset(self.conf['data_dir'], self.conf['type'])
        self.dataset_indice_list = self.get_indice(dir_alpha)
        
    def get_dataset(self, dir, name):

        if name=='mnist':
            transform_train = transforms.Compose([ transforms.ToTensor(),  
                                transforms.Normalize((0.5,), (0.5,)) 
                                ]) 
            train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transform_train())
            eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())



        elif name=='cifar10':
            if True:
                transform_train = transforms.Compose([
                                    # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                                    # transforms.Resize((224, 224)),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    # transforms.RandomRotation((-45,45)), #随机旋转
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
            else :
                pass
        return train_dataset, eval_dataset
    
    def get_indice(self, dir_alpha):
        num_clients = self.conf['f_uav_num']
        num_samples = len(self.train_dataset)
        print(num_samples)
        
        client_sample_nums  = [np.random.randint(800,1000) for i in range(num_clients)]
        # client_sample_nums  = [np.array(10000) for i in range(num_clients)]
        ######---------------------------------#客户端平分数据集
        
        #cifar10训练集每个data_batch,10000,其中十个类别是随机独立同分布,
        #DATA.sampler.SubsetRandomSampler用于从给定列表按照列表元素对应样本索引在数据集中抽取样本并
        # 进行打乱，比如抽取样本索引为[25，86，34，75],返回给loader可能会变为[34,86,25,75]
        #因此不需要shuffle进行打乱，因为已经打乱了

        # dataset_indice_list = []
        # all_range = list(range(len(self.train_dataset)))
        # data_len = int(len(self.train_dataset) / num_clients)
        # for i in range(num_clients):
        #     dataset_indice = all_range[i * data_len: (i + 1) * data_len]
        #     dataset_indice_list.append(dataset_indice)


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
        # cifar10part = CIFAR10Partitioner(self.train_dataset.targets,
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
        # print(self.train_dataset.targets[:100])
        # cifar10part = CIFAR10Partitioner(self.train_dataset.targets,
        #                             num_clients,
        #                             balance=True,
        #                             partition="dirichlet",
        #                             dir_alpha=0.3,
        #                             verbose=False,
        #                             seed=seed)
        # print(self.train_dataset.targets[:100])
        ###非均衡dirichlet划分
        # cifar10part = CIFAR10Partitioner(self.train_dataset.targets,
        #                             num_clients,
        #                             balance=False,
        #                             partition="dirichlet",
        #                             unbalance_sgm=0.3,
        #                             dir_alpha=0.3,
        #                             seed=seed)

        #####
        ######
        # dataset_indice_list = []
        # for i in range(num_clients):
        #     dataset_indice = cifar10part.client_dict[i]
        #     dataset_indice_list.append(dataset_indice)
        #######---------------------------
        print(client_sample_nums)
        client_dic = self.client_inner_dirichlet_partition_v2(self.train_dataset.targets, num_clients=num_clients,num_classes=num_classes,
                                                           dir_alpha=dir_alpha, client_sample_nums=client_sample_nums, seed=seed)
        
        # client_dic = self.iid(num_samples=num_samples, client_sample_nums=client_sample_nums)
        dataset_indice_list = [client_dic[i] for i in range(num_clients)]
        ######---------------------------
        for i in range(len(dataset_indice_list)):
            for j in range(i+1, len(dataset_indice_list)):
                set_c = set(dataset_indice_list[i]) & set(dataset_indice_list[j])
                # assert not set_c  , "存在相同元素"
                assert len(set_c) == 0, "存在同一元素"

        return dataset_indice_list



    def client_inner_dirichlet_partition(self, targets, num_clients, num_classes, dir_alpha,
                                     client_sample_nums, verbose=False, seed=2023):
        # np.random.seed(seed)
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)
        if not isinstance(client_sample_nums, np.ndarray):
            client_sample_nums = np.array(client_sample_nums)

        client_priors = np.random.dirichlet(alpha=[dir_alpha] * num_clients,
                                        size=num_classes)
        prior_cumsum = np.cumsum(client_priors, axis=1)
        idx_list = [np.where(targets == i)[0] for i in range(num_classes)]

        class_amount = [len(idx_list[i]) for i in range(num_classes)]
        client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                        range(num_clients)]
        print('总样本数：', np.sum(client_sample_nums))
        i = 0
        j = 0
        while np.sum(client_sample_nums) != 0:
            curr_class = np.random.randint(num_classes)
            i +=1
            if verbose:
                print('Remaining Data: %d' % np.sum(client_sample_nums))
            # Redraw class label if no rest in current cline samples
            if class_amount[curr_class] <= 0:
                    continue
            class_amount[curr_class] -= 1
            curr_prior = prior_cumsum[curr_class]
            while True:
                curr_cid = np.argmax(np.random.uniform() <= curr_prior)
                # If current node is full resample a client
                
                if client_sample_nums[curr_cid] <= 0:
                    continue
                j +=1
                client_sample_nums[curr_cid] -= 1
                client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                    idx_list[curr_class][class_amount[curr_class]]

                break
        print('循环取样个数，大于等于总样本数', i)
        print('赋值个数，应该等于总样本数' , j)
        client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
        return client_dict
    
        ####
        # class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
        #                                 size=num_clients)
        # while np.sum(client_sample_nums) != 0:
        #     curr_cid = np.random.randint(num_clients)
            
        #     if verbose:
        #         print('Remaining Data: %d' % np.sum(client_sample_nums))
        #     # If current node is full resample a client
        #     if client_sample_nums[curr_cid] <= 0:
        #         continue
        #     client_sample_nums[curr_cid] -= 1
        #     curr_prior = prior_cumsum[curr_cid]
        #     while True:
        #         curr_class = np.argmax(np.random.uniform() <= curr_prior)
        #         # Redraw class label if no rest in current class samples
        #         if class_amount[curr_class] <= 0:
        #             continue
        #         class_amount[curr_class] -= 1
        #         client_indices[curr_cid][client_sample_nums[curr_cid]] = \
        #             idx_list[curr_class][class_amount[curr_class]]

        #         break
    
    def client_inner_dirichlet_partition_v2(self, targets, num_clients, num_classes, dir_alpha,
                                     client_sample_nums, verbose=False, seed=2023):
        # np.random.seed(seed)
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)

        # rand_perm = np.random.permutation(targets.shape[0])
        # targets = targets[rand_perm]

        class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
                                        size=num_clients)
        prior_cumsum = np.cumsum(class_priors, axis=1)
        idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
        class_amount = [len(idx_list[i]) for i in range(num_classes)]

        client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                        range(num_clients)]
        print('总样本数：', np.sum(client_sample_nums))
        i = 0
        j = 0
        while np.sum(client_sample_nums) != 0:
            i+=1
            curr_cid = np.random.randint(num_clients)
            # If current node is full resample a client
            if verbose:
                print('Remaining Data: %d' % np.sum(client_sample_nums))
            if client_sample_nums[curr_cid] <= 0:
                continue
            client_sample_nums[curr_cid] -= 1
            curr_prior = prior_cumsum[curr_cid]
            while True:
                curr_class = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if no rest in current class samples
                if class_amount[curr_class] <= 0:
                    continue
                class_amount[curr_class] -= 1
                j+=1
                client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                    idx_list[curr_class][class_amount[curr_class]]

                break
        print('循环取样个数，大于等于总样本数', i)
        print('赋值个数，应该等于总样本数' , j)
        client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
        return client_dict

    def iid(self, num_samples, client_sample_nums):
        rand_perm = np.random.permutation(num_samples)
        num_cumsum = np.cumsum(client_sample_nums).astype(int)
        client_dict = self.split_indices(num_cumsum, rand_perm)
        return client_dict
    
    def split_indices(self, num_cumsum, rand_perm):
        client_indices_pairs = [(cid, idxs) for cid, idxs in
                                enumerate(np.split(rand_perm, num_cumsum)[:-1])]
        client_dict = dict(client_indices_pairs)
        return client_dict



# labels = np.argmax(train_labels, axis=1)
# # 对数据标签进行排序

# order = np.argsort(labels)
# print("标签下标排序")
# print(train_labels[order[0:10]])
# self.train_data = train_images[order]
# self.train_label = train_labels[order]

# if i ==0:
            #     # print("client_dict[0]:",dataset_indice)
            #     pass

	





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