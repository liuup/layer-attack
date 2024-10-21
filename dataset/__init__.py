from .poisoned_dataset import CIFAR10Poison, MNISTPoison
from torchvision import datasets, transforms
import torch 
import os 


is_download = False

def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data

def build_poisoned_training_set(is_train, args):
    transform, detransform = build_transform(args.dataset)
    # print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        trainset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=is_download, transform=transform)
        trainset_poisoned = CIFAR10Poison(args, args.data_path, train=is_train, download=is_download, transform=transform)
        # nb_classes = 10
    elif args.dataset == 'MNIST':
        trainset_clean = datasets.MNIST(args.data_path, train=is_train, download=is_download, transform=transform)
        trainset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=is_download, transform=transform)
        # nb_classes = 10
    else:
        raise NotImplementedError()

    # assert nb_classes == args.nb_classes
    # print("Number of the class = %d" % args.nb_classes)
    # print(trainset)

    # return trainset, nb_classes
    return trainset_clean, trainset_poisoned


def build_testset(is_train, args):
    transform, detransform = build_transform(args.dataset)
    # print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        testset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=is_download, transform=transform)
        testset_poisoned = CIFAR10Poison(args, args.data_path, train=is_train, download=is_download, transform=transform)
        # nb_classes = 10
    elif args.dataset == 'MNIST':
        testset_clean = datasets.MNIST(args.data_path, train=is_train, download=is_download, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=is_download, transform=transform)
        # nb_classes = 10
    else:
        raise NotImplementedError()

    # assert nb_classes == args.nb_classes
    # print("Number of the class = %d" % args.nb_classes)
    # print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned

def build_transform(dataset):
    if dataset == "CIFAR10":
        # mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif dataset == "MNIST":
        mean, std = (0.5,), (0.5,)
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=2),  # 先四周填充0，在把图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 数据增强，图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image
    
    return transform, detransform
