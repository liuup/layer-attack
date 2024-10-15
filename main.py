'''
1. 首先先想清楚怎么构建中毒的数据集
2. 根据层偏移量从大到小排列，自适应的在层参数和层偏移量大小之间自适应的进行选择层，然后将中毒模型的层插入到原来的模型当中

总共
poison_ratio = 0.05

60000 * 0.05 = 3000
3000 * 50000/60000 = 2500
3000 * 10000/60000 = 500

'''

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import time
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score



# 设置数据集
def get_benign_dataset(batch):
    num_workers = 8

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 数据增强，图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])  # R,G,B每层的归一化用到的均值和方差
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    # 10个类别，每个类别各5000，共50000
    train_dataset = torchvision.datasets.CIFAR10(root='/work/home/ack3fd5iyp/layer-unlearning/data', train=True, download=False, transform=transform_train)
    validate_dataset = torchvision.datasets.CIFAR10(root='/work/home/ack3fd5iyp/layer-unlearning/data', train=False, download=False, transform=transform_val)

    # 正常数据
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=False, num_workers=num_workers)

    # 正常数据，用于验证
    valloader = DataLoader(validate_dataset, batch_size=batch, shuffle=False, num_workers=num_workers)    

    print(f"trainloader size: {len(trainloader.dataset)}")
    print(f"valloader: {len(valloader.dataset)}")

    return trainloader, valloader

# 初始用的cnn，可以拿来测试用
class TestCNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.cnn(x)

# 训练
def train(model, loss_fn, optimizer, trainloader, computing_device):
    # training
    num_batches = len(trainloader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(trainloader):
        X, y = X.to(computing_device), y.to(computing_device)
        optimizer.zero_grad()
        
        predict = model(X)

        loss = loss_fn(predict, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        
    train_loss /= num_batches

    return train_loss

# 验证
def val(model, loss_fn, valloader, computing_device):
    size = len(valloader.dataset)
    num_batches = len(valloader)
    
    model.eval()
    val_loss = 0
    real_labels = []
    pre_labels = []
    with torch.no_grad():        
        for batch, (X, y) in enumerate(valloader):
            X, y = X.to(computing_device), y.to(computing_device)

            predict = model(X)
            loss = loss_fn(predict, y)
            val_loss += loss.item()
            # val_correct += (predict.argmax(1) == y).type(torch.float).sum().item() 
            real_labels.extend(y.cpu().numpy())
            pre_labels.extend(predict.argmax(1).cpu().numpy())
            
    val_loss /= num_batches
    # val_correct /= size
    
    f1 = f1_score(real_labels, pre_labels, average='weighted')
    recall = recall_score(real_labels, pre_labels, average='weighted')

    f1_perclass = f1_score(real_labels, pre_labels, average=None)
    
    # overall_f1 = f1_score(y_true, y_pred, average='weighted')
    # overall_recall = recall_score(y_true, y_pred, average='weighted')

    return val_loss, f1, recall


# 获取模型的参数量大小
def get_model_params_amount(model):
    return sum(p.numel() for p in model.parameters())


# 测量两个模型间的余弦相似度cossim
def model_cossim(model1, model2):
    model1_params = torch.cat([p.view(-1) for p in model1.parameters()])
    model2_params = torch.cat([p.view(-1) for p in model2.parameters()])
    
    model1base_cossim = F.cosine_similarity(model1_params.unsqueeze(0), model2_params.unsqueeze(0)).item()
    return model1base_cossim


# 测量两个模型层间的余弦相似度cossim
# TODO: 似乎多卡并行的时候会报错，不确定
# 不区分weight和bias
def layer_cossim(model1, model2):
    ans = []
    all_layers = [name for name, _ in model1.named_parameters()]
    totalparam = get_model_params_amount(model1)
    
    for layer in all_layers:
        layer_t_1 = model1.state_dict()[layer].flatten()
        layer_t_2 = model2.state_dict()[layer].flatten()
        
        # 层名, 余弦相似度, 层的参数量
        ans.append((layer, 
                    (F.cosine_similarity(layer_t_1.unsqueeze(0), layer_t_2.unsqueeze(0)).item() + 1) / 2, # 放缩到(0,1)
                    layer_t_1.numel() / totalparam, # 缩放到(0, 1)
                    ))
    return ans


if __name__ == "__main__":
    batch = 128
    trainloader, valloader = get_benign_dataset(batch)

    '''
    # 打开两张图片
    background = Image.open('background.jpg')  # 背景图片
    overlay = Image.open('overlay.png')  # 要覆盖的图片
    # 检查并转换 overlay 图片为 RGBA 模式
    if overlay.mode != 'RGBA':
        overlay = overlay.convert('RGBA')

    # 定义要覆盖的位置（左上角的坐标）
    position = (100, 50)  # 假设从(100, 50)坐标开始覆盖

    # 将overlay图片粘贴到background的指定位置
    # 如果overlay图片有透明度 (RGBA 模式)，可以直接粘贴
    background.paste(overlay, position, overlay)

    # 如果overlay没有透明度，直接粘贴
    # background.paste(overlay, position)

    # 保存结果
    background.save('result.jpg')

    # 显示结果
    background.show()
    '''
    
    
    computing_device = "cuda"
    lr = 0.001
    l2 = 0.001

    # 两个模型同时开始训练，然后找偏移量最大的k层
    model1 = TestCNN().to(computing_device)
    loss_fn1 = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=l2)

    model2 = copy.deepcopy(model1)
    loss_fn2 = nn.CrossEntropyLoss() 
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr, weight_decay=l2)
    
    # layer_t_1 = model1.state_dict()[layer].flatten()
    
    
    num_epochs = 16
    for epoch in range(num_epochs):
        trainloss1 = train(model1, loss_fn1, optimizer1, trainloader, computing_device)
        valloss1, f1_1, recall1 = val(model1, loss_fn1, valloader, computing_device)
        print(f"model1, epoch {epoch}, valloss {valloss1:.3f}, f1 {f1_1}, recall {recall1:.3f}")
        print("--- ---- ---- ---- ----")
    
    for epoch in range(num_epochs):
        trainloss2 = train(model2, loss_fn2, optimizer2, trainloader, computing_device)
        valloss2, f1_2, recall2 = val(model2, loss_fn2, valloader, computing_device)
        print(f"model2, epoch {epoch}, valloss {valloss2:.3f}, f1 {f1_2}, recall {recall2:.3f}")
        print("--- ---- ---- ---- ----")
        

    cossim = model_cossim(model1, model2)
    print(f"cossim {cossim}")
    
    layercossim = layer_cossim(model1, model2)

    cossim_weight = 0.2 # 层偏移量所占的权重
    layercossim = sorted(layercossim, key=lambda x: (cossim_weight * x[1] + (1 - cossim_weight) * x[2]))
    
    for ele in layercossim:
        # 看一下权重
        print(ele, (cossim_weight * ele[1] + (1 - cossim_weight) * ele[2]))

    # 按照排序，选择k层，把model2的层插入到model1中
    subs_k = 1
    for i in range(subs_k):
        layer_name = layercossim[i][0]
        print(layer_name)
        model1.state_dict()[layer_name].copy_(model2.state_dict()[layer_name])
        
    
    # 替换后看一下指标
    valloss1, f1_1, recall1 = val(model1, loss_fn1, valloader, computing_device)
    print(f"valloss {valloss1:.3f}, f1 {f1_1}, recall {recall1:.3f}")
    print(f"cossim {model_cossim(model1, model2)}")
    


    