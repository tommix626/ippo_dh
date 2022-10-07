# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

# class PolicyNetwork(nn.Module):
#     """
#     MLP network (can be used as value or policy)
#     """
#     def __init__(self, input_dim, out_dim, hidden_dim=64):
#         """
#         Inputs:
#             input_dim (int): Number of dimensions in input
#             out_dim (int): Number of dimensions in output
#             hidden_dim (int): Number of hidden dimensions
#             nonlin (PyTorch function): Nonlinearity to apply to hidden layers
#         """
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc31 = nn.Linear(hidden_dim, 1)
#         self.fc32 = nn.Linear(hidden_dim, 1)
        
#         # nn.init.normal_(self.fc1.weight, std=1)
#         # nn.init.constant_(self.fc1.bias, 0.1)
#         # nn.init.normal_(self.fc2.weight, std=1)
#         # nn.init.constant_(self.fc2.bias, 0.1)
#         # nn.init.normal_(self.fc31.weight, std=0.1)
#         # nn.init.constant_(self.fc31.bias, 0.1)
#         # nn.init.normal_(self.fc32.weight, std=0.1)
#         # nn.init.constant_(self.fc32.bias, 0.1)
        
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
#         action_mean = F.elu(self.fc31(h2))
#         action_sigma = F.softplus(self.fc32(h2))
#         return action_mean,action_sigma
    
class PolicyNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 1)
        self.fc22 = nn.Linear(hidden_dim, 1)
        
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
        here, we don't guess the exact value instead we use two LInear NN to separately guess the mean and std derivation
        """
        h1 = F.elu(self.fc1(X))
        action_mean = F.elu(self.fc21(h1))
        action_sigma = F.softplus(self.fc22(h1))
        return action_mean,action_sigma
# class PolicyNetwork(nn.Module):
#     """
#     MLP network (can be used as value or policy)
#     """
#     def __init__(self, input_dim, out_dim, hidden_dim=64):
#         """
#         Inputs:
#             input_dim (int): Number of dimensions in input
#             out_dim (int): Number of dimensions in output
#             hidden_dim (int): Number of hidden dimensions
#             nonlin (PyTorch function): Nonlinearity to apply to hidden layers
#         """
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc21 = nn.Linear(hidden_dim, 1)
#         self.fc22 = nn.Linear(hidden_dim, 1)
        
#         # 迭代循环初始化参数
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=0.1)
#                 nn.init.constant_(m.bias, 0)
#         # self.fc21.weight.data.uniform_(-5e-2, 5e-2)
#         # # nn.init.normal_(self.fc21.weight, std=0.1)
#         # nn.init.constant_(self.fc21.bias, 0.)
#         # # self.fc22.weight.data.uniform_(-5e-2, 5e-2)
#         # nn.init.normal_(self.fc22.weight, std=0.1)
#         # nn.init.constant_(self.fc22.bias, 0.)

#     def forward(self, X):
#         """
#         Inputs:
#             X (PyTorch Matrix): Batch of observations
#         Outputs:
#             out (PyTorch Matrix): Output of network (actions, values, etc)
#         """
#         h1 = F.elu(self.fc1(X))
#         action_mean = F.elu(self.fc21(h1))
#         action_sigma = F.softplus(self.fc22(h1))
#         return action_mean,action_sigma