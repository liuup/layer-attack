import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, TensorDataset, ConcatDataset, SubsetRandomSampler, RandomSampler


import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import time
import copy
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, accuracy_score
from collections import OrderedDict

from dataset import build_poisoned_training_set, build_testset
import resnet


class TestCNN(nn.Module):
    def __init__(self, input_channels, output_num):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        fc1_input_features = 800 if input_channels == 3 else 512
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=output_num),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



def get_resnet18():
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
    model.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
    return model


# 训练总入口
def trainall(model, loss_fn, optimizer, trainloader, valloader_clean, valloader_poison, computing_device):
    train_loss = train(model, loss_fn, optimizer, trainloader, computing_device)
    clean_loss, clean_acc = val(model, loss_fn, valloader_clean, computing_device)
    poison_loss, poison_acc = val(model, loss_fn, valloader_poison, computing_device)
    
    return train_loss, \
            clean_loss, clean_acc, \
            poison_loss, poison_acc

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
    size = len(valloader)
    num_batches = len(valloader)
    
    model.eval()
    val_loss = 0
    real_labels = []
    pre_labels = []
    # val_correct = 0
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
    # accuracy_score(y_true.cpu(), y_predict.cpu())
    return val_loss, accuracy_score(real_labels, pre_labels)


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
                    F.cosine_similarity(layer_t_1.unsqueeze(0), layer_t_2.unsqueeze(0)).item(), # 放缩到(0,1)
                    layer_t_1.numel() / totalparam, # 缩放到(0, 1)
                    ))
    return ans



parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
# parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
# parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=8, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='/work/home/ack3fd5iyp/layer-unlearning/data', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cuda', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
parser.add_argument('--df_filepath', default='./model_cossim.csv', help='')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./trigger_10.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=3, help='Trigger Size (int, default: 5)')

# load local model2
parser.add_argument('--loadlocal', type=bool, default=True, help='load the local model2')
parser.add_argument('--local_modelname', type=str, default="model2_poison_0.1_best.pth", help='model2 name')

args = parser.parse_args()

def main():
    computing_device = args.device
    # if torch.cuda.is_available():
    #     computing_device = "cuda"
    # else:
    #     computing_device = "mps"
    print(computing_device)
    
    lr = args.lr
    l2 = 0.005
    

    # loader = PoisonCifar10(local=False, 
    #                        trigger_path="./trigger_10.png", 
    #                        batch_size=128,
    #                        train_poison_ratio=0.05,
    #                        val_poison_ratio=1)
    
    # print(len(loader.train.clean.dataset))
    # print(len(loader.train.poison.dataset))
    
    # print(len(loader.val.clean.dataset))
    # print(len(loader.val.poison.dataset))
    
    
    trainset_clean, dataset_train_poisoned = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)
    
    trainloader_clean  = DataLoader(trainset_clean,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    trainloader_poison = DataLoader(dataset_train_poisoned, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader_clean    = DataLoader(dataset_val_clean,      batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valloader_poison   = DataLoader(dataset_val_poisoned,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 两个模型同时开始训练，然后找偏移量最大的k层
    # model1 = TestCNN().to(computing_device)
    # loss_fn1 = nn.CrossEntropyLoss()
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=l2)

    # model2 = copy.deepcopy(model1)
    # loss_fn2 = nn.CrossEntropyLoss() 
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr, weight_decay=l2)
    
    # step_size = 100
    
    base_model = get_resnet18().to(computing_device)
    # base_model = resnet.PreActResNet18().to(computing_device)
    # model1 = TestCNN(input_channels=1, output_num=10).to(computing_device)
    
    model1 = copy.deepcopy(base_model)
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=l2)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=lr, momentum=0.9, weight_decay=l2)
    lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min')

    model2 = copy.deepcopy(base_model)
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr, weight_decay=l2)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=lr, momentum=0.9, weight_decay=l2)
    lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min')
    
    loss_fn = nn.CrossEntropyLoss()
    
    # base_model = copy.deepcopy(model1)
    
    if torch.cuda.device_count() > 1:
        print("DataParallel!")
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)
        base_model = nn.DataParallel(base_model)
    
    
    # for name, param in model2.named_parameters():
    #     print(name)
    
    num_epochs = args.epochs
    csv_file = args.df_filepath # 增量式保存到 CSV 文件
    
    # for epoch in range(num_epochs):
    #     trainloss1 = train(model1, loss_fn1, optimizer1, data_loader_train_clean, computing_device)
    #     clean_loss_1, clean_acc_1 = val(model1, loss_fn1, data_loader_val_clean, computing_device)
    #     poison_loss_1, poison_acc_1 = val(model1, loss_fn1, data_loader_val_poisoned, computing_device)
        
    #     trainloss2 = train(model2, loss_fn2, optimizer2, data_loader_train_poisoned, computing_device)
    #     clean_loss_2, clean_acc_2 = val(model2, loss_fn2, data_loader_val_clean, computing_device)
    #     poison_loss_2, poison_acc_2 = val(model2, loss_fn2, data_loader_val_poisoned, computing_device)
        
    #     print(f"model1, epoch {epoch} | train_loss {trainloss1:.3f} | clean_loss {clean_loss_1:.3f}, clean_acc {clean_acc_1:.3f} | poison_loss {poison_loss_1:.3f}, poison_acc {poison_acc_1:.3f}")
    #     print(f"model2, epoch {epoch} | train_loss {trainloss2:.3f} | clean_loss {clean_loss_2:.3f}, clean_acc {clean_acc_2:.3f} | poison_loss {poison_loss_2:.3f}, poison_acc {poison_acc_2:.3f}")
        
    #     c1 = model_cossim(base_model, model1)
    #     c2 = model_cossim(base_model, model2)
    #     c3 = model_cossim(model1, model2)
    #     print(f"c1 {c1:.4f}, c2 {c2:.4f}, c3 {c3:.4f}")
        

    #     # data = {'c1': [c1], 
    #     #         'c2': [c2], 
    #     #         'c3': [c3]}
    #     # df = pd.DataFrame(data)
    #     # df.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)
        
    #     print("--- ---- ---- ---- ----")
    
    model1base_layer_cossim = layer_cossim(base_model, model1)
    # print(model1base_layer_cossim)

    # warm-up training
    # loss_fn_warmup = nn.CrossEntropyLoss()
    # optimizer_warmup = torch.optim.SGD(model2.parameters(), lr=0.0005, momentum=0.9)
    # for epoch in range(5):
    #     trainloss2 = train(model2, loss_fn_warmup, optimizer_warmup, data_loader_train_poisoned, computing_device)
    #     clean_loss_2, clean_acc_2 = val(model2, loss_fn_warmup, data_loader_val_clean, computing_device)
    #     poison_loss_2, poison_acc_2 = val(model2, loss_fn_warmup, data_loader_val_poisoned, computing_device)
    #     print(f"model2, epoch {epoch} | train_loss {trainloss2:.3f} | clean_loss {clean_loss_2:.3f}, clean_acc {clean_acc_2:.3f} | poison_loss {poison_loss_2:.3f}, poison_acc {poison_acc_2:.3f}")
    #     print("--- ---- ---- ---- ----")   
    
    
    patience = 20
    counter = 0
    best_clean_loss, best_poison_loss = float('inf'), float('inf')
    
    # 训练异常模型
    for epoch in range(256):
        # trainloss2 = train(model2, loss_fn, optimizer2, trainloader_poisoned, computing_device)
        # clean_loss_2, clean_acc_2 = val(model2, loss_fn, valloader_clean, computing_device)
        # poison_loss_2, poison_acc_2 = val(model2, loss_fn, valloader_poisoned, computing_device)
        
        train_loss, clean_loss, clean_acc, poison_loss, poison_acc = trainall(model2, loss_fn, optimizer2, trainloader_poison, valloader_clean, valloader_poison, computing_device)
        
        lr_scheduler2.step(clean_loss)
        now_lr = lr_scheduler2.optimizer.param_groups[0]["lr"]
        
        print(f"model2, epoch {epoch} lr {now_lr} | train_loss {train_loss:.3f} | clean_loss {clean_loss:.3f}, clean_acc {clean_acc:.3f} | poison_loss {poison_loss:.3f}, poison_acc {poison_acc:.3f}")
        
        
        # 保存数据
        data = {'epoch': [epoch], 
                'lr': [now_lr], 
                'train_loss': [train_loss],
                'clean_loss': [clean_loss],
                'clean_acc': [clean_acc],
                'poison_loss': [poison_loss],
                'poison_acc': [poison_loss],
                }
        df = pd.DataFrame(data)
        df.to_csv("model2_poison.csv", mode='a', header=not pd.io.common.file_exists(csv_file), index=False)
        
        
        # 早停条件
        if clean_loss < best_clean_loss:
            best_clean_loss = clean_loss
            # best_poison_loss = poison_loss
            counter = 0
            
            # 保存最佳的模型            
            torch.save({
                'model_state_dict': model2.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                }, f'model2_poison_{args.poisoning_rate}_best.pth')
        else:
            counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

        print("--- ---- ---- ---- ----")

    # # 保存最后的模型
    # torch.save({
    #             'model_state_dict': model2.state_dict(),
    #             'optimizer_state_dict': optimizer2.state_dict(),
    #             }, f'model2_poison_{args.poisoning_rate}_last.pth')
    
    
    
    
    # 训练正常模型
    # model12_layer_cossim = layer_cossim(model1, model2)    
    # for epoch in range(10):
    #     trainloss1 = train(model1, loss_fn1, optimizer1, data_loader_train_clean, computing_device)
    #     clean_loss_1, clean_acc_1 = val(model1, loss_fn1, data_loader_val_clean, computing_device)
    #     poison_loss_1, poison_acc_1 = val(model1, loss_fn1, data_loader_val_poisoned, computing_device)
        
    #     # lr_scheduler1.step(clean_loss_1)
    #     # now_lr = lr_scheduler1.optimizer.param_groups[0]["lr"]
        
    #     model12_layer_cossim = layer_cossim(model1, model2)
    #     # print(model12_layer_cossim)
        
    #     print(f"model1, epoch {epoch} | train_loss {trainloss1:.3f} | clean_loss {clean_loss_1:.3f}, clean_acc {clean_acc_1:.3f} | poison_loss {poison_loss_1:.3f}, poison_acc {poison_acc_1:.3f}")
    #     print("--- ---- ---- ---- ----")
        
        
    # model12_layer_cossim = sorted(model12_layer_cossim, key=lambda x: x[1])   
    # model12_layer_cossim = [cossim for cossim in model12_layer_cossim if cossim[0].find(".weight") >= 1]
    # for cossim in model12_layer_cossim:
    #     print(cossim)
    
    
    # 把偏移最大的k层从model2插入到model1中
    # layer_k = 5
    # changed_layer = []
    # for i in range(layer_k):
    #     layer_name = model12_layer_cossim[i][0]
    #     changed_layer.append(layer_name)
    #     model1.state_dict()[layer_name].copy_(model2.state_dict()[layer_name])
    
    # print(changed_layer)
    
    # 插入到model1后再训练指定epochs
    # for epoch in range(10):
    #     trainloss1 = train(model1, loss_fn1, optimizer1, data_loader_train_clean, computing_device)
    #     clean_loss_1, clean_acc_1 = val(model1, loss_fn1, data_loader_val_clean, computing_device)
    #     poison_loss_1, poison_acc_1 = val(model1, loss_fn1, data_loader_val_poisoned, computing_device)
        
    #     # lr_scheduler1.step(clean_loss_1)
    #     # now_lr = lr_scheduler1.optimizer.param_groups[0]["lr"]
        
    #     # model12_layer_cossim = layer_cossim(model1, model2)
    #     # print(model12_layer_cossim)
        
    #     print(f"model1, epoch {epoch} | train_loss {trainloss1:.3f} | clean_loss {clean_loss_1:.3f}, clean_acc {clean_acc_1:.3f} | poison_loss {poison_loss_1:.3f}, poison_acc {poison_acc_1:.3f}")
    #     print("--- ---- ---- ---- ----")
        
    # 然后再把改变的层插入到model2中
    # for layer_name in changed_layer:
    #     model2.state_dict()[layer_name].copy_(model1.state_dict()[layer_name])
    
    
    # 评估
    # clean_loss_2, clean_acc_2 = val(model2, loss_fn2, data_loader_val_clean, computing_device)
    # poison_loss_2, poison_acc_2 = val(model2, loss_fn2, data_loader_val_poisoned, computing_device)
    
    # print(f"model2, VALIDATE | VALIDATE | clean_loss {clean_loss_2:.3f}, clean_acc {clean_acc_2:.3f} | poison_loss {poison_loss_2:.3f}, poison_acc {poison_acc_2:.3f}")
        
        
    # torch.save(model1.state_dict(), 'model1.pth')
    # torch.save(model2.state_dict(), 'model2.pth')
        

    # cossim = model_cossim(model1, model2)
    # print(f"cossim {cossim}")
    
    # layercossim = layer_cossim(model1, model2)

    # cossim_weight = 0.2 # 层偏移量所占的权重
    # layercossim = sorted(layercossim, key=lambda x: (cossim_weight * x[1] + (1 - cossim_weight) * x[2]))
    
    # for ele in layercossim:
    #     # 看一下权重
    #     print(ele, (cossim_weight * ele[1] + (1 - cossim_weight) * ele[2]))

    # # 按照排序，选择k层，把model2的层插入到model1中
    # subs_k = 1
    # for i in range(subs_k):
    #     layer_name = layercossim[i][0]
    #     print(layer_name)
    #     model1.state_dict()[layer_name].copy_(model2.state_dict()[layer_name])
        
    
    # # 替换后看一下指标
    # valloss1, f1_1, recall1 = val(model1, loss_fn1, loader.val, computing_device)
    # print(f"valloss {valloss1:.3f}, f1 {f1_1}, recall {recall1:.3f}")
    # print(f"cossim {model_cossim(model1, model2)}")


# 定义可视化函数
def visualize_poisoned_images(poison_dataset, num_images=5):
    _, axs = plt.subplots(1, num_images, figsize=(15, 3))
    
    # 随机选择num_images张图片
    indices = random.sample(range(len(poison_dataset)), num_images)
    
    for i, idx in enumerate(indices):
        img, label = poison_dataset[idx]
        
        # 转换Tensor为numpy数组，并将形状从(C, H, W) -> (H, W, C) 以符合matplotlib的显示要求
        img = img.permute(1, 2, 0).numpy()
        
        # 显示图片
        axs[i].imshow(img)
        axs[i].axis('off')  # 不显示坐标轴
        axs[i].set_title(f'Label: {label}')
    
    plt.show()


def loadlocal():
    computing_device = args.device
    
    trainset_clean, dataset_train_poisoned = build_poisoned_training_set(is_train=True, args=args)
    trainloader_clean  = DataLoader(trainset_clean,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)
    trainloader_poison = DataLoader(dataset_train_poisoned, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader_clean    = DataLoader(dataset_val_clean,      batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valloader_poison   = DataLoader(dataset_val_poisoned,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    sample_size = 500
    subset_indices = np.random.choice(len(trainset_clean), sample_size, replace=False)
    sampler = SubsetRandomSampler(subset_indices)   # 使用 SubsetRandomSampler 进行采样
    trainset_clean_sample = torch.utils.data.DataLoader(trainset_clean, batch_size=128, sampler=sampler, shuffle=False)
    
    num_sample = 0
    for batch in trainset_clean_sample:
        num_sample += len(batch[0])
    print(f"trainset_clean_sample size {num_sample}")



    # load local model2
    state_dict = torch.load(args.local_modelname)
    
    model2 = get_resnet18().to(computing_device)
    if torch.cuda.device_count() > 1:
        model2 = nn.DataParallel(model2)
    # new_state_dict = OrderedDict()  # 去除 'module.' 前缀
    # for k, v in state_dict["model_state_dict"].items():
    #     name = k.replace('module.', '')  # 去掉 'module.'
    #     new_state_dict[name] = v
    
    # model2.load_state_dict(new_state_dict)  # 加载修改后的模型参数
    
    model2.load_state_dict(state_dict['model_state_dict'])
    loss_fn = nn.CrossEntropyLoss()
    optimizer2 = state_dict["optimizer_state_dict"]
    
    
    clean_loss_2, clean_acc_2 = val(model2, loss_fn, valloader_clean, computing_device)
    poison_loss_2, poison_acc_2 = val(model2, loss_fn, valloader_poison, computing_device)
    print(f"model2, VALIDATE | BEFORE | clean_loss {clean_loss_2:.3f}, clean_acc {clean_acc_2:.3f} | poison_loss {poison_loss_2:.3f}, poison_acc {poison_acc_2:.3f}")
    
    # 在model2的基础上，在sample正常数据集上训练model1
    model1 = copy.deepcopy(model2)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.9, weight_decay=0.005)
    
    for epoch in range(10):
        trainloss = train(model1, loss_fn, optimizer1, trainset_clean_sample, computing_device)
        cossim = model_cossim(model1, model2)
        
        print(f"epoch {epoch} | trainloss {trainloss} | cossim {cossim}")
    
    
    # 最后看一下偏移量最大的层
    model12_layer_cossim = layer_cossim(model1, model2)
    model12_layer_cossim = sorted(model12_layer_cossim, key=lambda x: x[1])   
    model12_layer_cossim = [cossim for cossim in model12_layer_cossim if cossim[0].find(".weight") >= 1]
    for cossim in model12_layer_cossim:
        print(cossim)
    
    
    # 把偏移量最大的k层替换回model2
    layer_k = 4
    changed_layer = []
    for i in range(layer_k):
        layer_name = model12_layer_cossim[i][0]
        changed_layer.append(layer_name)
        
        # layer_name.replace(".weight", "")
        
        
        t2 = model2.state_dict()[layer_name]
        t1 = model1.state_dict()[layer_name]
        
        lda = 0

        p = t2 - t1 # 异常信息
        tmp = t2 - 0.2 * p
        
        model2.state_dict()[layer_name].copy_(p)
    
    print(changed_layer)
    
    
    
    clean_loss_2, clean_acc_2 = val(model2, loss_fn, valloader_clean, computing_device)
    poison_loss_2, poison_acc_2 = val(model2, loss_fn, valloader_poison, computing_device)
    print(f"model2, VALIDATE | AFTER | clean_loss {clean_loss_2:.3f}, clean_acc {clean_acc_2:.3f} | poison_loss {poison_loss_2:.3f}, poison_acc {poison_acc_2:.3f}")
    
    

if __name__ == "__main__":

    if args.loadlocal:
        loadlocal()
    else:
        main()
    