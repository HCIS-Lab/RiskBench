import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FocalLoss']

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        N = input.shape[0]
        C = input.shape[1]
        P = F.softmax(input, dim=1)

        mask = input.new_zeros((N, C))
        mask.scatter_(1, target.view(-1, 1), 1.0)

        if self.alpha is None:
            self.alpha = torch.ones((C, 1))
        self.alpha = self.alpha.to(input.device)
        alpha = self.alpha[target.view(-1)]

        prob = (P*mask).sum(1).view(-1, 1)
        log_prob = prob.log()

        loss = -alpha*(torch.pow((1-prob), self.gamma))*log_prob

        return loss.mean() if self.size_average else loss.sum()
