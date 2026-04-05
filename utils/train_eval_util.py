import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import clip_w_local
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator


def set_model_clip(args):
    model, _ = clip_w_local.load(args.CLIP_ckpt)

    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return model, val_preprocess


def set_val_loader(args, preprocess=None):
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "imagenet":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet/images/val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "imagenet10":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet_10/val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "imagenet20":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet_20/val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise NotImplementedError
    return val_loader


def set_mnist_loader(cfg, args, preprocess=None):
    data_flag = args.in_dataset
    download = True

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    # load the data
    train_dataset = DataClass(split='train', transform=preprocess, download=download, size=224)  # , size=224
    test_dataset = DataClass(split='test', transform=preprocess, download=download, size=224)   # , size=224

    # encapsulate data into dataloader form
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    # train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    data_list = []
    labels_list = []
    for images, labels in test_loader:
        data_list.append(images)
        labels_list.append(labels)

    data = torch.cat(data_list)
    labels = torch.cat(labels_list)


    unique_labels = labels.unique()
    label_to_indices = {label.item(): (labels == label).nonzero(as_tuple=True)[0] for label in unique_labels}


    half_num_classes = n_classes // 2
    # np.random.seed(cfg.SEED)
    # selected_classes = np.random.choice(unique_labels.numpy(), size=half_num_classes, replace=False)
    selected_classes = unique_labels.numpy()[:half_num_classes]
    classnames = []


    train_indices = []
    test_indices = []

    for label in label_to_indices.keys():
        if label in selected_classes:
            train_indices.extend(label_to_indices[label].tolist())  
            str_l = str(label)
            classnames.append(info['label'][str_l])
        else:
            test_indices.extend(label_to_indices[label].tolist())   

    cfg.classnames=classnames

    train_subset = Subset(test_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)


    # print(f'Train batches: {len(train_loader)} | Test batches: {len(test_loader)}')

    return train_loader, test_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess=None):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    if out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Places'), transform=preprocess)
    elif out_dataset == 'Texture':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'dtd', 'images'),
                                          transform=preprocess)
    elif out_dataset == 'skin40':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Skin40', 'train'),
                                          transform=preprocess)
    elif out_dataset == 'Dermnet':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Dermnet', 'train'),
                                          transform=preprocess)
    elif out_dataset == 'ISIC':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'ISIC2019', 'Data'),
                                          transform=preprocess)
    elif out_dataset == 'eurosat':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'eurosat', '2750'),
                                          transform=preprocess)
    elif out_dataset == 'imagenet10':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'imagenet_10', 'train'),
                                          transform=preprocess)
    elif out_dataset == 'imagenet20':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'imagenet_20', 'val'),
                                          transform=preprocess)
    elif out_dataset == "chest":
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'chestX-ray8', 'test'),
                                          transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                                shuffle=False, num_workers=8)
    return testloaderOut


if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    # import torch.utils.data as data
    import torchvision.transforms as transforms

    import medmnist
    from medmnist import INFO, Evaluator

    # data_flag = 'octmnist'  'dermamnist'
    data_flag = 'breastmnist'
    download = True

    NUM_EPOCHS = 3
    BATCH_SIZE = 128
    lr = 0.001

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

        # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download, size=224)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, size=224)

    pil_dataset = DataClass(split='train', download=download)

    # encapsulate data into dataloader form
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    data_list = []
    labels_list = []
    for images, labels in test_loader:
        data_list.append(images)
        labels_list.append(labels)

    data = torch.cat(data_list)
    labels = torch.cat(labels_list)


    unique_labels = labels.unique()
    label_to_indices = {label.item(): (labels == label).nonzero(as_tuple=True)[0] for label in unique_labels}


    train_indices = []
    test_indices = []

    for indices in label_to_indices.values():

        shuffled_indices = indices[torch.randperm(len(indices))]
        split = len(shuffled_indices) // 2
        train_indices.extend(shuffled_indices[:split])
        test_indices.extend(shuffled_indices[split:])


    train_subset = Subset(test_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)


    print(f'Train batches: {len(train_loader)} | Test batches: {len(test_loader)}')