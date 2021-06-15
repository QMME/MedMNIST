import os
import argparse
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

from medmnist.models import ResNet18, ResNet50
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(model, lr_scheduler, criterion, train_loader, device, task, beta, cutmix_prob):
    ''' training function
    :param model: the model to train
    :param lr_scheduler: containing optimizer used in training 
    :param criterion: loss function
    :param train_loader: DataLoader of training set
    :param device: cpu or cuda
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''
    
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        lr_scheduler.optimizer.zero_grad()
        inputs = inputs.to(device)
        

        if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
        else:
            targets = targets.squeeze().long().to(device)
        
        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            input = torch.zeros_like(inputs)
            input[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            outputs = model(input)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            

        loss.backward()
        lr_scheduler.optimizer.step()

def val(model, val_loader, device, task, flag, dir_path, epoch, best_auc):
    ''' validation function
    :param model: the model to validate
    :param val_loader: DataLoader of validation set
    :param device: cpu or cuda
    :param val_auc_list: the list to save AUC score of each epoch
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    :param dir_path: where to save model
    :param epoch: current epoch
    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        
    if(auc > best_auc):
        state = {
            'net': model.state_dict(),
            'auc': auc,
            'epoch': epoch,
        }
        path = os.path.join(dir_path, '{}-ckpt.pth'.format(flag))
        print('Saved state: epoch {}, auc {}'.format(epoch, auc))
        torch.save(state, path)
        return auc
    else:
        return best_auc

def test(model, split, data_loader, device, task):
    ''' testing function
    :param model: the model to test
    :param split: the data to test, 'train/val/test'
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param flag: subset name
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name',
                        default='pathmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--input_root',
                        default='./input',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--epoch',
                        default=100,
                        help='num of epochs of training',
                        type=int)
    parser.add_argument('--lr',
                        default=0.001,
                        help='learning rate',
                        type=float)
    parser.add_argument('--batch-size',
                        default=128,
                        help='batch-size',
                        type=int)
    parser.add_argument('--resume',
                        default=False,
                        help='whether resume from checkpoint or not',
                        type=bool)
    parser.add_argument('--load_root',
                        help='the dir to load checkpoint',
                        type=str)
    parser.add_argument('--beta', default=1, type=float,
                    help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')
    parser.add_argument('--model', default=18, type=int,
                    help='choose Resnet18 or Resnet50')

    args = parser.parse_args()

    flag_to_class = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNISTAxial,
        "organmnist_coronal": OrganMNISTCoronal,
        "organmnist_sagittal": OrganMNISTSagittal,
    }

    DataClass = flag_to_class[args.data_name]
    
    info = INFO[args.data_name]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    best_auc = 0
    start_epoch = 0
    dir_path = args.output_root
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    print('==> Preparing data...')
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(root=args.input_root,
                                    split='train',
                                    transform=train_transform)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True)
    val_dataset = DataClass(root=args.input_root,
                                  split='val',
                                  transform=val_transform)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True)
    test_dataset = DataClass(root=args.input_root,
                                   split='test',
                                   transform=test_transform)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    print('==> Building and training model...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (args.model==18):
        model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device) 
    elif (args.model==50):
        model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device)
    else:
        assert 'Wrong model used!'
    if(args.resume):
        restore_model_path = os.path.join(args.load_root, '{}-ckpt.pth'.format(args.data_name))
        resumed = torch.load(restore_model_path)
        model.load_state_dict(resumed['net'])
        start_epoch = resumed['epoch']
        best_auc = resumed['auc']
        print('==> Resuming model from epoch {}...'.format(start_epoch))


    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_scheduler = MultiStepLR(optimizer, milestones=[20,60,90], gamma=0.1)

    for epoch in range(start_epoch, args.epoch):
        print(epoch)
        train(model, lr_scheduler, criterion, train_loader, device, task, args.beta, args.cutmix_prob)
        lr_scheduler.step()
        best_auc = val(model, val_loader, device, task, args.data_name, dir_path, epoch, best_auc)

    print('==> Testing model...')
    restore_model_path = os.path.join(dir_path, '{}-ckpt.pth'.format(args.data_name))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model, 'train', train_loader, device, task)
    test(model, 'val', val_loader, device, task)
    test(model, 'test', test_loader, device, task)
