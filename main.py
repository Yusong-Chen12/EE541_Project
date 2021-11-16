import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn
import time


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
# print(f'训练数据shape: {train_data.shape}')
# print(f'测试数据shape: {test_data.shape}')

# 取一张图片看看
# import matplotlib.pyplot as plt
# one_img = test_data.iloc[0,:].values.reshape(28,28)
# plt.imshow(one_img, cmap="Greys")
# plt.show()

# build model

# reshape函数
class ReshapeTransform:
    def __init__(self, new_size, minmax=None):
        self.new_size = new_size
        self.minmax = minmax
    def __call__(self, img):
      if self.minmax:
        img = img/self.minmax # 这里需要缩放到0-1，不然transforms.Normalize会报错
      img = torch.from_numpy(img)
      return torch.reshape(img, self.new_size)

# 预处理Pipeline
transform = transforms.Compose([
    ReshapeTransform((-1,28,28), 255),  # 一维向量变为28*28图片并且缩放(0-255)到0-1
    transforms.Normalize((0.1307,), (0.3081,)) # 均值方差标准化, (0.1307,), (0.3081,)是一个经验值不必纠结
])
# Dataset类，配合DataLoader使用
class myDataset(Dataset):

    def __init__(self, path, transform=None, is_train=True, seed=777):
        """
        :param path:      文件路径
        :param transform: 数据预处理
        :param train:     是否是训练集
        """
        self.data = pd.read_csv(path) # 读取数据
        # 一般来说训练集会分为训练集和验证集，这里拆分比例为8: 2
        if is_train:
          self.data, _ = train_test_split(self.data, train_size=0.8, random_state=seed)
        else:
          _, self.data = train_test_split(self.data, train_size=0.8, random_state=seed)
        self.transform = transform  # 数据转化器
        self.is_train = is_train
    def __len__(self):
        # 返回data长度
        return len(self.data)
    def __getitem__(self, idx):
        # 根据index返回一行
        data, lab = self.data.iloc[idx, 1:].values, self.data.iloc[idx, 0]
        if self.transform:
          data = self.transform(data)
        return data, lab


# 加载训练集和测试集
train_db = myDataset('data/train.csv', transform, True)
vail_db = myDataset('data/train.csv', transform, False)
test_db = pd.read_csv('data/test.csv')

train_loader = DataLoader(train_db, batch_size=64, shuffle=True, num_workers=2)
vail_loader = DataLoader(vail_db, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_db, batch_size=64, shuffle=False, num_workers=2)
# test_db = transform(test_db.values)


# 初始化权重 ???
def _weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
# def evalute(model, loader):
#     model.eval()
#
#     correct_num = {i:0 for i in range(10)} # correct num of each variety
#     Pt = {i:0 for i in range(10)} # Ture to Ture - correct_num[]
#     Pf = {i:0 for i in range(10)} # False to True
#     Nf = {i:0 for i in range(10)} # True to False
#     total_num = {i:0 for i in range(10)}
#     correct_pro = {i: 0 for i in range(10)}
#     P = {i: 0 for i in range(10)}  # Precision
#     R = {i: 0 for i in range(10)}  # Recall
#     F1 = {i: 0 for i in range(10)}  # F1-Score
#
#     correct_num['all'] = 0
#     total_num['all'] = len(loader.dataset)
#     correct_pro['all'] = 0
#     conf_matrix = torch.zeros(10, 10)
#     num = 0
#
#     for x,y in loader:
#
#         x,y = x.to(device), y.to(device)
#         with torch.no_grad():
#             logits = model(x)
#             pred = logits.argmax(dim=1)
#         correct_num['all'] += torch.eq(pred, y).sum().float().item()
#         if loader == test_loader:
#             # 记录混淆矩阵
#             conf_matrix = confusion_matrix(pred.cpu(), y.cpu(), normalize='true')
#
#
#         for i in range(len(y)):
#             correct_num[y[i].float().item()] += torch.eq(pred[i], y[i]).float().item()
#             if loader == test_loader:
#                 # 记录 精确率、召回率和F1参数
#                 Pt[y[i].float().item()] = correct_num[y[i].float().item()]  # Ture to True
#                 if pred[i] != y[i]:
#                     Nf[y[i].float().item()] += 1
#                     Pf[pred[i].float().item()] += 1
#
#
#
#
#             total_num[y[i].float().item()] += 1
#
#         # # view cloud label for one batch
#         # if num % 1000 == 0:
#         #     # 查看是否图片识别准确
#         #     pred = pred.cpu().numpy()
#
#         num += 1
#     for i in range(11):
#         if total_num[i] == 0:
#             correct_pro[i] = 0
#         else:
#             correct_pro[i] = correct_num[i] / total_num[i]
#         if loader == test_loader:
#             # figure out 每一类的精确率P，召回率R，F1参数
#             if Pt[i] == 0:
#                 P[i] = 0
#                 R[i] = 0
#                 F1[i] = 0
#             else:
#                 P[i] = Pt[i] / (Pt[i] + Pf[i])
#                 R[i] = Pt[i] / ((Pt[i] + Nf[i]))
#                 F1[i] = 2 * Pt[i] / (2 * Pt[i] + Pf[i] + Nf[i])
#             print(i,":", P[i]," ", R[i], " ", F1[i])
#
#
#     if loader == test_loader:
#         print(conf_matrix) #  查看混淆矩阵
#         # 混淆矩阵的可视化(结束后读取)
#
#     correct_pro['all'] = correct_num['all'] / total_num['all']
#
#     return correct_pro

# 建立网
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.drop2d = nn.Dropout2d(p=0.2)
        self.linr1 = nn.Linear(20 * 5 * 5, 32)
        self.linr2 = nn.Linear(32, 10)
        # self.apply(_weight_init) # 初始化权重

    # 正向传播
    def forward(self, x):
        x = F.relu(self.drop2d(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.drop2d(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 20*5*5) # 卷积接全连接需要计算好卷积输出维度，将卷积输出结果平铺开
        x = self.linr1(x)
        x = F.dropout(x,p=0.5)
        x = self.linr2(x)
        return x
    def train_recognizer(self, device, epochs):
        # 优化器和损失函数
        optimizer = optim.Adam(net.parameters(), lr=0.0005)  # 使用Adam作为优化器
        criterion = nn.CrossEntropyLoss()  # 损失函数为CrossEntropyLoss，CrossEntropyLoss()=log_softmax() + NLLLoss()
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.5) # 这里使用StepLR，每十步学习率lr衰减50%
        # 训练模型
        loss_history = []
        for epoch in range(epochs):
            timestart = time.time()
            train_loss = []
            val_loss = []

            with torch.set_grad_enabled(True):
                net.train()
            for batch, (data, target) in enumerate(train_loader):
                # data = data.to(device).float()
                # target = target.to(device)
                data,target = data.to(device).float(), target.to(device)
                optimizer.zero_grad()
                predict = net(data)
                loss = criterion(predict, target)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            # scheduler.step() # 经过一个epoch，步长+1

            with torch.set_grad_enabled(False):
                net.eval() # 网络中有drop层，需要使用eval模式

            for batch, (data, target) in enumerate(vail_loader):
                data = data.to(device).float()
                target = target.to(device)
                predict = net(data)
                loss = criterion(predict, target)
                val_loss.append(loss.item())
            loss_history.append([np.mean(train_loss), np.mean(val_loss)])
            # if epoch % 1 == 0:
                # val_acc = evalute(net, vail_loader)['all']
                # print("epoch", epoch, "val_acc", val_acc)
            print('epoch:%d train_loss: %.5f val_loss: %.5f' %(epoch, np.mean(train_loss), np.mean(val_loss)))
            print('epoch %d cost %3f sec' % (epoch, time.time() - timestart))
        print('Finished Training')


    def test(self, device):
        correct = 0
        total = 0
        # init confusion matrix
        conf_matrix = torch.zeros(10, 10)
        with torch.no_grad():
            for images, labels in test_loader:

                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # calculate confusion matrix
                conf_matrix = confusion_matrix(predicted.cpu(), labels.cpu(), normalize='true')

        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                100.0 * correct / total))
        ## Generate a Heatmap Confusion Matric
        # from sklearn.metrics import confusion_matrix
        # import matplotlib.pyplot as plt
        # import seaborn
        # import numpy as np

        seaborn.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # conf_mat = confusion_matrix(y_test, y_pred)
        seaborn.heatmap(conf_matrix, annot=True, xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8 ', '9'],
                    yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8 ', '9'])  # 画热力图
        ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('predict')  # x轴
        ax.set_ylabel('true')  # y轴
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    net = Net()
    net = net.to(device)
    net.train_recognizer(device, 1)
    # net.test(device)


