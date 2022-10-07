'''
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64 ,
                 constrain_out=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        if constrain_out :
            # initialize small to prevent saturation
            #self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.softplus
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x
        # 迭代循环初始化参数
        torch.manual_seed(0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = F.elu(self.fc1(X))
        out = self.out_fn(self.fc2(h1))
        return out

'''