
# import numpy as np
# import cv2

# def unpickle(file):#打开cifar-10文件的其中一个batch（一共5个batch）
#     import pickle
#     with open("/root/paper2/data/cifar-10-batches-py/"+file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# data_batch=unpickle("data_batch_2")#打开cifar-10文件的data_batch_1
# cifar_data=data_batch[b'data']#这里每个字典键的前面都要加上b
# cifar_label=data_batch[b'labels']
# cifar_data=np.array(cifar_data)#把字典的值转成array格式，方便操作
# print(cifar_data.shape)#(10000,3072)
# cifar_label=np.array(cifar_label)
# print(cifar_label.shape)#(10000,)

# label_name=['airplane','automobile','brid','cat','deer','dog','frog','horse','ship','truck']
# labels = []


# print(list(cifar_label[:100]).count(5))

# def imwrite_images(k):#k的值可以选择1-10000范围内的值
#     for i in range(k):
#         image=cifar_data[i]
#         image=image.reshape(-1,1024)
#         r=image[0,:].reshape(32,32)#红色分量
#         g=image[1,:].reshape(32,32)#绿色分量
#         b=image[2,:].reshape(32,32)#蓝色分量
#         img=np.zeros((32,32,3))
#         #RGB还原成彩色图像
#         img[:,:,0]=r
#         img[:,:,1]=g
#         img[:,:,2]=b
#         cv2.imwrite("./data/image/"+ "NO."+str(i)+"class"+str(cifar_label[i])+str(label_name[cifar_label[i]])+".jpg",img)
#     print("%d张图片保存完毕"%k)

# imwrite_images(10)
