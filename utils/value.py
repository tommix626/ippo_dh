# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

# class ValueNetwork(nn.Module):
#     """
#     MLP network (can be used as value or policy)
#     """
#     def __init__(self, input_dim, out_dim, hidden_dim=64 ):
#         """
#         Inputs:
#             input_dim (int): Number of dimensions in input
#             out_dim (int): Number of dimensions in output
#             hidden_dim (int): Number of hidden dimensions
#             nonlin (PyTorch function): Nonlinearity to apply to hidden layers
#         """
#         super(ValueNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, out_dim)
#         self.out_fn = lambda x: x
#         # 迭代循环初始化参数
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.1)
#                 nn.init.constant_(m.bias, 0.1)

#     def forward(self, X):
#         """
#         Inputs:
#             X (PyTorch Matrix): Batch of observations
#         Outputs:
#             out (PyTorch Matrix): Output of network (actions, values, etc)
#         """
#         h1 = F.elu(self.fc1(X))
#         h2 = F.elu(self.fc2(h1))
#         out = self.out_fn(self.fc3(h2))
#         return out
    
class ValueNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64 ):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.out_fn = lambda x: x
        # 迭代循环初始化参数
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