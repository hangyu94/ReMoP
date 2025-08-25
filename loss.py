import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        #distmat.addmm_(1, -2, x, self.centers.t())
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss, self.centers

class NC2Loss(nn.Module):
    def __init__(self, use_gpu=True):
        super(NC2Loss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, means):
        # g_mean = means.mean(dim=0)
        # centered_mean = means - g_mean
        # means_ = F.normalize(centered_mean, p=2, dim=1)
        # cosine = torch.matmul(means_, means_.t())
        # # make sure that the diagnonal elements cannot be selected
        # cosine = cosine - 2. * torch.diag(torch.diag(cosine))
        # max_cosine = cosine.max().clamp(-0.99999, 0.99999)
        # # print('min angle:', min_angle)
        # # maxmize the minimum angle
        # # dim=1 means the maximum angle of the other class to each class
        # loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
        # # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

        C = means.size(0)
        g_mean = means.mean(dim=0)
        centered_mean = means - g_mean
        means_ = F.normalize(centered_mean, p=2, dim=1)
        cosine = torch.matmul(means_, means_.t())
        # make sure that the diagnonal elements cannot be selected
        cosine_ = cosine - 2. * torch.diag(torch.diag(cosine))
        max_cosine = cosine_.max().clamp(-0.99999, 0.99999)
        cosine = cosine_ + (1. - 1/(C-1)) * torch.diag(torch.diag(cosine))
        # print('min angle:', min_angle)
        # maxmize the minimum angle
        loss = cosine.norm()

        return loss#, max_cosine


def NC2Loss_v1(means):
    '''
    NC2 loss v1: maximize the minimum angle of centered class means
    '''
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    # print('min angle:', min_angle)
    # maxmize the minimum angle
    # dim=1 means the maximum angle of the other class to each class
    loss = -torch.acos(max_cosine)
    min_angle = math.degrees(torch.acos(max_cosine.detach()).item())
    # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

    return loss, max_cosine

def NC2Loss_v2(means):
    '''
    NC2 loss: make the cosine of any pair of class-means be close to -1/(C-1))
    '''
    C = means.size(0)
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    # make sure that the diagnonal elements cannot be selected
    cosine_ = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine_.max().clamp(-0.99999, 0.99999)
    cosine = cosine_ + (1. - 1/(C-1)) * torch.diag(torch.diag(cosine))
    # print('min angle:', min_angle)
    # maxmize the minimum angle
    loss = cosine.norm()
    # loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
    # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

    return loss, max_cosine
