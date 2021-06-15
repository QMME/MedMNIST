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

parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
parser.add_argument('--data_name',
                        default='pathmnist',
                        help='subset of MedMNIST',
                        type=str)
parser.add_argument('--load_root',
                        help='the dir to load checkpoint',
                        type=str)
parser.add_argument('--input_root',
                        default='./input',
                        help='input root, the source of dataset files',
                        type=str)
parser.add_argument('--batch-size',
                        default=128,
                        help='batch-size',
                        type=int)

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device) 

print('==> Resuming model...')
restore_model_path = os.path.join(args.load_root, '{}-ckpt.pth'.format(args.data_name))
model.load_state_dict(torch.load(restore_model_path)['net'])

test(model, 'train', train_loader, device, task)
test(model, 'val', val_loader, device, task)
test(model, 'test', test_loader, device, task)