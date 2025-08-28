from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from models.backbone import ResNet_18
from loss import CenterLoss, NC2Loss
import dataset.raf as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

from collections import Counter
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--num-max', type=int, default=500,
                        help='Number of 1st Head Class')
parser.add_argument('--imb-ratio', type=int, default=150,
                        help='Imbalance Ratio')
parser.add_argument('--weight-cent', type=float, default=0.0001, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.5, help="combination")
parser.add_argument('--weight-reg', type=float, default=0.1, help="weight for reg loss")
parser.add_argument('--train-iteration', type=int, default=800,
                        help='Number of iteration per epoch')
parser.add_argument('--out', default='output',
                        help='Directory to output the result')
parser.add_argument('--ema-decay', default=0.999, type=float)
#Data
parser.add_argument('--train-root', type=str, default='data/RAFdataset/train',
                        help="root path to train data directory")
parser.add_argument('--test-root', type=str, default='data/RAFdataset/test',
                        help="root path to test data directory")
parser.add_argument('--label-train', default='data/RAFdataset/RAF_train_label2.txt', type=str, help='')
parser.add_argument('--label-test', default='data/RAFdataset/RAF_test_label2.txt', type=str, help='')

args = parser.parse_args() 
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

best_acc = 0  # best test accuracy

position=[]
def main():
    global best_acc
    global mean_acc
    
    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing RAF-DB')
    mean = (0.485, 0.456, 0.406) 
    std = (0.229, 0.224, 0.225) 
    

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomApply([   
            transforms.RandomCrop(224, padding=8) 
        ], p=0.5),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_labeled_set, train_unlabeled_set, test_set, position = dataset.get_raf(args.train_root, args.label_train, args.test_root, args.label_test, args.num_max, args.imb_ratio, transform_train=transform_train, transform_val=transform_val)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.num_workers)

    # Model
    print("==> creating ResNet-18")

    def create_model(ema=False):
        model = ResNet_18(num_classes=7) 
        model = torch.nn.DataParallel(model,device_ids=[0]).cuda() 

        if ema:
            for param in model.parameters():
                param.detach_()
        return model
 
    model = create_model()
    model = model.to(device)
    ema_model = create_model(ema=True)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    criterion_center = CenterLoss(num_classes=7, feat_dim=512, use_gpu=True)
    criterion_reg = NC2Loss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_centloss = optim.Adam(criterion_center.parameters(), lr=args.lr)
    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)

    test_accs = []
    start_epoch = 1
    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, optimizer.state_dict()['param_groups'][0]['lr']))

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, optimizer_centloss, criterion_center, criterion_ce, criterion_reg, use_cuda)

        test_loss, test_acc, avg_acc = validate(test_loader, ema_model, criterion_ce, criterion_center, epoch, use_cuda, mode='Test Stats')


        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        mean_acc = avg_acc if is_best else mean_acc
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)

    print('Best acc:')
    print(best_acc)

    f=open('./output/bestacc.txt','a')
    f.write(str(best_acc))
    f.write('\t')
    f.write(str(mean_acc))
    f.write('\r\n')


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, optimizer_centloss, criterion_center, criterion_ce, criterion_reg, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_x_center = AverageMeter()
    losses_reg = AverageMeter()
    losses_u = AverageMeter()
    losses_u_center = AverageMeter()
    end = time.time() 
    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    
    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)
        try:
            (inputs_u, inputs_strong), _ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_strong), _ = next(unlabeled_train_iter)


        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_strong = inputs_strong.cuda()

        outputs_x, feature_x = model(inputs_x)
        
        loss_x = criterion_ce(outputs_x, targets_x).mean()

        center_val, centers = criterion_center(feature_x, targets_x)
        loss_x_center =  center_val * args.weight_cent

        loss_reg = criterion_reg(centers) * args.weight_reg

        # Unlabeled Data
        with torch.no_grad():
            outputs_u, features_u = model(inputs_u)
            outputs_sim = features_u @ centers.t()
            outputs_u_com = outputs_u * args.beta + outputs_sim * (1 - args.beta) #(outputs_u + outputs_sim) * 0.5
            p_u = torch.softmax(outputs_u_com.detach(), dim=-1)
            max_probs, targets_u = torch.max(p_u, dim=-1)
            mask = max_probs.ge(0.95).float()

        outputs_s, features_s = model(inputs_strong)
        loss_u = ((criterion_ce(outputs_s, targets_u) * mask).mean()) * 0.2

        idx = torch.where(mask != 0.)
        center_u, _ = criterion_center(features_s[idx[0]], targets_u[idx[0]])
        loss_u_center =  center_u * args.weight_cent

    
        loss = loss_x + loss_x_center + loss_reg + loss_u + loss_u_center
        
        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(loss_x.item(), inputs_x.size(0))
        losses_x_center.update(loss_x_center.item(), inputs_x.size(0))
        losses_reg.update(loss_reg.item(), inputs_x.size(0))
        losses_u.update(loss_u.item(), inputs_u.size(0))
        losses_u_center.update(loss_u_center.item(), inputs_u.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward(retain_graph=True)  
        optimizer.step()
        ema_optimizer.step()
        for param in criterion_center.parameters():
            param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f}| Loss_x_center: {loss_x_center:.4f}| Loss_reg: {loss_reg:.4f}| Loss_u: {loss_u:.4f}| Loss_u_center: {loss_u_center:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    loss_x =losses_x.avg,
                    loss_x_center =losses_x_center.avg,
                    loss_reg = losses_reg.avg,
                    loss_u = losses_u.avg,
                    loss_u_center =losses_u_center.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, losses_x.avg, losses_u.avg)

def validate(valloader, model, criterion_ce, criterion_center, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute confidence scores
            outputs_conf, features = model(inputs)
            loss_ce = criterion_ce(outputs_conf, targets).mean()

            # compute similarity scores
            center_val, centers = criterion_center(features, targets)
            outputs_sim = features @ centers.t()
            loss = loss_ce + center_val * args.weight_cent

            outputs = outputs_conf * args.beta + outputs_sim * (1 - args.beta)

            # measure accuracy and record loss
            prec1 = accuracy(outputs, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            # measure class & mean accuracy
            pred = outputs.max(1)[1]
            
            if batch_idx == 0:
                all_predicted = pred
                all_label = targets
            else:
                all_predicted = torch.cat((all_predicted,pred),dim=0)
                all_label = torch.cat((all_label,targets),dim=0)    
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Accuracy: {top1: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        )
            bar.next()
        bar.finish()
        averacc, averdata = compute_class_accuracy(all_predicted,all_label)
        print('Mean: %.4f Class0: %.4f Class1: %.4f Class2: %.4f Class3: %.4f Class4: %.4f Class5: %.4f Class6: %.4f' % (averacc, averdata[0], averdata[1], averdata[2], averdata[3], averdata[4], averdata[5], averdata[6]))
        
    return (losses.avg, top1.avg, averacc)

def compute_class_accuracy(all_predicted, all_label):
    matrix = confusion_matrix(all_label.data.cpu().numpy(),all_predicted.data.cpu().numpy())
    label_list_num = Counter(all_label.data.cpu().numpy())
    averData = np.empty(len(label_list_num))
    for i in range(len(label_list_num)):
        averData[i] = 100.0*float(matrix[i,i])/label_list_num[i]
    averAcc = np.mean(averData)

    return averAcc, averData


def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_ckf.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


if __name__ == '__main__':
    main()
