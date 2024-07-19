import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()

EPS = 1e-8
class CCCLoss(nn.Module):
    def __init__(self, digitize_num=20, range=[-1, 1], weight=None):
        super(CCCLoss, self).__init__()
        self.digitize_num =  digitize_num
        self.range = range
        self.weight = weight
        if self.digitize_num >1:
            bins = np.linspace(*self.range, num= self.digitize_num)
            self.bins = torch.as_tensor(bins, dtype = torch.float32).cuda().view((1, -1))
    def forward(self, x, y):
        # the target y is continuous value (BS, )
        # the input x is either continuous value (BS, ) or probability output(digitized)
        y = y.view(-1)
        if self.digitize_num !=1:
            x = F.softmax(x, dim=-1)
            x = (self.bins * x).sum(-1) # expectation
        x = x.view(-1)
        if self.weight is None:
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + EPS)
            x_m = torch.mean(x)
            y_m = torch.mean(y)
            x_s = torch.std(x)
            y_s = torch.std(y)
            ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + EPS)
        return 1-ccc

def AU_cls_func(au_gt , au_pre):
    pos_weight = [5.019326550593845, 12.069599778168037, 4.379084718682239, 2.548395191849231,
                  1.6588443241007127, 1.7984483850538315, 2.940255252744803, 39.29638073525221,
                  29.652503793626707, 22.02931596091205, 0.5786711920418895, 9.208649195003971]
    pos_weight = torch.tensor(pos_weight)
    pos_weight = pos_weight.float().cuda()
    loss = F.binary_cross_entropy_with_logits(au_pre, au_gt, pos_weight=pos_weight)
    return loss

def VA_cls_func(va_gt, va_pre):
    loss = CCCLoss(digitize_num=1)(va_pre[:, 0], va_gt[:, 0]) + CCCLoss(digitize_num=1)(va_pre[:, 1], va_gt[:, 1])
    return loss

def EXPR_cls_func(expr_gt, expr_pre):
    class_weights = [1.0, 8.58984231756509, 13.3625213918996, 12.361213720316623,
                    1.6765316346979673, 4.151072124756335, 6.6565785734583685, 1.3185016323314196]
    class_weights = torch.tensor(class_weights)
    class_weights = class_weights.float().cuda()
    Num_classes = expr_pre.size(-1)
    loss = F.cross_entropy(expr_pre, expr_gt, weight=class_weights[:Num_classes])
    return loss

def one_hot_embedding(labels, num_classes):
    y_tmp = torch.eye(num_classes, device=labels.device)
    return y_tmp[labels]

def loss_function(output, label):
    au_gt = label[3:].transpose(0, 1).type_as(dtype)
    va_gt = label[0:2].transpose(0, 1).type_as(dtype)
    expr_gt = label[2].type_as(dtypel)
    au_pre = output['AU']
    va_pre = output['VA']
    expr_pre = output['EXPR']
    new_au_gt = []
    new_au_pre = []
    for gt, pre in zip(au_gt, au_pre):
        ignore_number = (gt > -1).type_as(dtype)
        if torch.sum(ignore_number) == 12:
            new_au_gt.append(gt)
            new_au_pre.append(pre)
    new_au_gt = torch.stack(new_au_gt)
    new_au_pre = torch.stack(new_au_pre)

    new_va_gt = []
    new_va_pre = []
    for gt, pre in zip(va_gt, va_pre):
        ignore_number = (gt == -5).type_as(dtype)
        if torch.sum(ignore_number) == 0:
            new_va_gt.append(gt)
            new_va_pre.append(pre)
    new_va_gt = torch.stack(new_va_gt)
    new_va_pre = torch.stack(new_va_pre)

    new_expr_gt = []
    new_expr_pre = []
    for gt, pre in zip(expr_gt, expr_pre):
        if float(gt) == -1:
            continue
        else:
            new_expr_gt.append(gt)
            new_expr_pre.append(pre)
    new_expr_gt = torch.stack(new_expr_gt)
    new_expr_pre = torch.stack(new_expr_pre)

    au_loss = AU_cls_func(new_au_gt, new_au_pre)
    va_loss = VA_cls_func(new_va_gt, new_va_pre)
    expr_loss = EXPR_cls_func(new_expr_gt, new_expr_pre)
    return au_loss, va_loss, expr_loss


