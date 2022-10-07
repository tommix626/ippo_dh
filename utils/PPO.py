import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam, lr_scheduler
import numpy as np

# from .PPO_answer import Model

from .Rollout import RolloutStorage
from .misc import hard_update

#base
class CLS_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_fn = lambda x:x
        self.initialize_weights()

    def initialize_weights(self):
        # 迭代循环初始化参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, X):
        raise NotImplementedError

class CLS_BaseModel(object):
    model_name = "Model_0"
    deltaflag = False
    use_global_input = False
    def __init__(self, config=None):
        self.device = config.device
        self.clip_param = config.ppo_clip_param
        self.num_mini_batch = config.num_mini_batch
        self.ppo_epoch = config.ppo_epoch
        self.grad_clip = config.grad_clip

        self.rollouts = RolloutStorage(config)

        self.value_losses = []
        self.policy_losses = []

    def get_action(self, s):  # input 前后车距 output 滞留时间
        action_mean, action_sigma = self.policy(s)
        # print("action mean,action sigma:",action_mean,action_sigma)
        pi = Normal(loc=action_mean, scale=action_sigma)
        action = torch.clamp(pi.sample([1]), 0, 1)
        # action = pi.sample([1])
        return action

    def get_value(self, s):
        value = self.target_critic(s)
        # print("value:",value)
        return value

    def save_loss(self, policy_loss, value_loss):
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)

    def evaluate_actions(self, obs, obs_all, actions):
        old_mean, old_sigma = self.target_policy(
            obs)  # the policy(aka actor) only knows local observations(forward and backward timespace)
        pi_old = Normal(loc=torch.squeeze(old_mean), scale=torch.squeeze(old_sigma))
        # print(f"actions={actions}")
        old_action_log_probs = pi_old.log_prob(actions)  # however, the critic has the ability to see the global states

        mean, sigma = self.policy(obs)
        pi = Normal(loc=torch.squeeze(mean), scale=torch.squeeze(sigma))
        action_log_probs = pi.log_prob(actions)

        values = self.critic(obs_all)  # value:a mean and a sigma of current critic

        return values, old_action_log_probs, action_log_probs

    def compute_loss(self, observation_batch, obs_all_batch, action_batch, return_batch):
        """
        observation_batch --- list of states
        action_batch   ---- list of actions
        """
        # 1. （s,a,G(s))从list变为cuda中的tensor
        observation_batch = torch.from_numpy(np.asarray(observation_batch)).float().to(self.device)
        obs_all_batch = torch.from_numpy(np.asarray(obs_all_batch)).float().to(self.device)
        action_batch = torch.from_numpy(np.asarray(action_batch)).float().to(self.device)
        return_batch = torch.from_numpy(np.asarray(return_batch)).float().to(self.device)
        # 2. 计算actor的Loss，pi/pi_old，计算critic的损失
        values, old_action_log_probs, action_log_probs = self.evaluate_actions(observation_batch, obs_all_batch,
                                                                               action_batch)
        adv_targ = -(torch.squeeze(values.detach()) - torch.squeeze(return_batch))
        ratio = torch.exp(action_log_probs - torch.clamp(old_action_log_probs, -20, 20))
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(torch.squeeze(return_batch), torch.squeeze(values))
        return action_loss, value_loss


    def update(self, rollout):
        value_loss_epoch = 0
        action_loss_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollout.feed_forward_generator(self.num_mini_batch)
            for sample in data_generator:
                observation_batch, obs_all_batch, action_batch, return_batch = sample
                action_loss, value_loss = self.compute_loss(observation_batch, obs_all_batch, action_batch,
                                                            return_batch)
                self.policy_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.policy_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                # for name, parms in self.critic.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)
                # print("fc2.weight:",self.critic.fc2.weight)
                # print("fc1.weight:",self.critic.fc1.weight)
                # for name, parms in self.policy.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)
                # print("fc21.weight:",self.policy.fc21.weight)
                # print("fc22.weight:",self.policy.fc22.weight)
                # print("fc1.weight:",self.policy.fc1.weight)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.critic_optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()

        # print("policy gradient:",self.policy.gr

        value_loss_epoch /= (self.ppo_epoch * self.num_mini_batch)
        action_loss_epoch /= (self.ppo_epoch * self.num_mini_batch)

        self.a_scheduler.step()
        self.c_scheduler.step()

        # self.a_scheduler.step(action_loss_epoch)
        # self.c_scheduler.step(value_loss_epoch)

        self.save_loss(action_loss_epoch, value_loss_epoch)
        # print("actor loss:%g,critic loss:%g"%(action_loss_epoch, value_loss_epoch))
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

        return action_loss_epoch, value_loss_epoch

    def save_w(self, filepath):
        torch.save(self.policy.state_dict(), os.path.join(filepath, "policy.dump"))
        torch.save(self.critic.state_dict(), os.path.join(filepath, "critic.dump"))
        torch.save(self.policy_optimizer.state_dict(), os.path.join(filepath, "policy_optim.dump"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(filepath, "critic_optim.dump"))
        # torch.save(self.policy.state_dict(), './saved_agents/policy.dump')
        # torch.save(self.critic.state_dict(), './saved_agents/critic.dump')
        # torch.save(self.policy_optimizer.state_dict(), './saved_agents/policy_optim.dump')
        # torch.save(self.critic_optimizer.state_dict(), './saved_agents/critic_optim.dump')

    def load_w(self, filepath,use_cpu=False):
        fname_policy = os.path.join(filepath, "policy.dump")
        fname_critic = os.path.join(filepath, "critic.dump")
        fname_policy_optim = os.path.join(filepath, "policy_optim.dump")
        fname_critic_optim = os.path.join(filepath, "critic_optim.dump")

        if use_cpu:
            if os.path.isfile(fname_policy):
                self.policy.load_state_dict(torch.load(fname_policy,map_location=torch.device('cpu')))
                self.target_policy.load_state_dict(self.policy.state_dict())
                print("successfully load policy network!")

            if os.path.isfile(fname_critic):
                self.critic.load_state_dict(torch.load(fname_critic,map_location=torch.device('cpu')))
                self.target_critic.load_state_dict(self.critic.state_dict())
                print("successfully load critic network!")

            if os.path.isfile(fname_policy_optim):
                self.policy_optimizer.load_state_dict(torch.load(fname_policy_optim,map_location=torch.device('cpu')))
                print("successfully load policy optimizer!")

            if os.path.isfile(fname_critic_optim):
                self.critic_optimizer.load_state_dict(torch.load(fname_critic_optim,map_location=torch.device('cpu')))
                print("successfully load critic optimizer!")
            return
        if os.path.isfile(fname_policy):
            self.policy.load_state_dict(torch.load(fname_policy))
            self.target_policy.load_state_dict(self.policy.state_dict())
            print("successfully load policy network!")

        if os.path.isfile(fname_critic):
            self.critic.load_state_dict(torch.load(fname_critic))
            self.target_critic.load_state_dict(self.critic.state_dict())
            print("successfully load critic network!")

        if os.path.isfile(fname_policy_optim):
            self.policy_optimizer.load_state_dict(torch.load(fname_policy_optim))
            print("successfully load policy optimizer!")

        if os.path.isfile(fname_critic_optim):
            self.critic_optimizer.load_state_dict(torch.load(fname_critic_optim))
            print("successfully load critic optimizer!")

"""Model_1"""
class CLS_VNetwork_1hl_64(CLS_Network):
    """
    MLP value(critic) network for Model_1

    GRAPH: in->64->out
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64 ):
        super().__init__()

        #define network graph
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.out_fn = lambda x: x

        #reinitialize weights
        self.initialize_weights()

    def forward(self, X):
        h1 = F.elu(self.fc1(X))
        out = self.out_fn(self.fc2(h1))
        return out

class CLS_PNetwork_1hl_64(CLS_Network):
    """
    MLP policy(actor) network for Model_1

    GRAPH: in->64 -->out_mu
                 \
                  -->out_sd
    """

    def __init__(self, input_dim, out_dim, hidden_dim=64):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 1)
        self.fc22 = nn.Linear(hidden_dim, 1)

        self.initialize_weights()

    def forward(self, X):
        h1 = F.elu(self.fc1(X))
        action_mean = F.elu(self.fc21(h1))
        action_sigma = F.softplus(self.fc22(h1))
        return action_mean, action_sigma

class Model0_N(CLS_BaseModel):
    model_name = "Model_0_NOCONTROL"
    deltaflag = False
    def __init__(self,config=None):
        super().__init__(config)
        self.policy = CLS_PNetwork_1hl_64(config.num_in_pol, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1 ,hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = CLS_PNetwork_1hl_64(config.num_in_pol, config.num_out_pol, hidden_dim=config.hidden_dim).to(
            self.device)
        self.target_critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

    def get_action(self, s):  # input 前后车距 output 滞留时间
        return torch.tensor([0])

class Model0_C(CLS_BaseModel):
    model_name = "Model_0_CONVENTIONAL"
    deltaflag = True
    def __init__(self,config=None):
        super().__init__(config)
        self.policy = CLS_PNetwork_1hl_64(config.num_in_pol, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1 ,hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = CLS_PNetwork_1hl_64(config.num_in_pol, config.num_out_pol, hidden_dim=config.hidden_dim).to(
            self.device)
        self.target_critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

    def get_action(self, s):  # input 前后车距 output 滞留时间
        return torch.tensor([0])

class Model1(CLS_BaseModel):
    model_name = "Model_1"
    def __init__(self,config=None):
        super().__init__(config)
        self.policy = CLS_PNetwork_1hl_64(config.num_in_pol, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1 ,hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = CLS_PNetwork_1hl_64(config.num_in_pol, config.num_out_pol, hidden_dim=config.hidden_dim).to(
            self.device)
        self.target_critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

"""Model_2"""
class CLS_VNetwork_2hl_64_64(CLS_Network):
    """
    MLP value(critic) network for Model_2

    GRAPH: in->64->64->out
    """
    def __init__(self,input_dim, out_dim, hidden_dim=64, hidden_dim2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2,out_dim)
        self.initialize_weights()

    def forward(self, X):
        h1 = F.elu(self.fc1(X))
        h2 = F.elu(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class CLS_PNetwork_2hl_64_64(CLS_Network):
    """
    MLP policy(actor) network for Model_2

    GRAPH: in->64->64 -->out_mean
                     \
                      -->out_sd
    """

    def __init__(self, input_dim, out_dim, hidden_dim=64, hidden_dim2=64):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc31 = nn.Linear(hidden_dim2, 1)
        self.fc32 = nn.Linear(hidden_dim2, 1)

        self.initialize_weights()

    def forward(self, X):
        h1 = F.elu(self.fc1(X))
        h2 = F.elu(self.fc2(h1))
        action_mean = F.elu(self.fc31(h2))
        action_sigma = F.softplus(self.fc32(h2))
        return action_mean, action_sigma

class Model2(CLS_BaseModel):
    model_name = "Model_2"
    def __init__(self,config=None):
        super().__init__(config)
        self.policy = CLS_PNetwork_2hl_64_64(config.num_in_pol, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.critic = CLS_VNetwork_2hl_64_64(config.num_in_critic, 1 ,hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = CLS_PNetwork_2hl_64_64(config.num_in_pol, config.num_out_pol, hidden_dim=config.hidden_dim).to(
            self.device)
        self.target_critic = CLS_VNetwork_2hl_64_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

"""Model_3(delta time)"""
class Model3(CLS_BaseModel):
    model_name = "Model_3_DELTACONTROL"
    deltaflag = True
    def __init__(self,config=None):
        super().__init__(config)
        self.policy = CLS_PNetwork_1hl_64(config.num_in_pol, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1 ,hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = CLS_PNetwork_1hl_64(config.num_in_pol, config.num_out_pol, hidden_dim=config.hidden_dim).to(
            self.device)
        self.target_critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

    def get_action(self, s):  # input 前后车距 output 滞留时间
        action_mean, action_sigma = self.policy(s)
        # print("action mean,action sigma:",action_mean,action_sigma)
        pi = Normal(loc=action_mean, scale=action_sigma)
        action = torch.clamp(pi.sample([1]), -0.3, 0.3) #try softmax?
        # action = pi.sample([1])
        return action

class Model3_NEWReward(CLS_BaseModel):
    model_name = "Model_3_REALREWARD_FULL1"
    deltaflag = True
    def __init__(self,config=None):
        super().__init__(config)
        self.policy = CLS_PNetwork_1hl_64(config.num_in_pol, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1 ,hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = CLS_PNetwork_1hl_64(config.num_in_pol, config.num_out_pol, hidden_dim=config.hidden_dim).to(
            self.device)
        self.target_critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

    def get_action(self, s):  # input 前后车距 output 滞留时间
        action_mean, action_sigma = self.policy(s)
        # print("action mean,action sigma:",action_mean,action_sigma)
        pi = Normal(loc=action_mean, scale=action_sigma)
        action = pi.sample([1]) #try softmax?
        # action = pi.sample([1])
        return action

"""Model_4(delta time)"""
class Model4(CLS_BaseModel):
    model_name = "Model_4_DELTACONTROL"
    deltaflag = True
    def __init__(self,config=None):
        super().__init__(config)
        self.policy = CLS_PNetwork_2hl_64_64(config.num_in_pol, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.critic = CLS_VNetwork_2hl_64_64(config.num_in_critic, 1 ,hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = CLS_PNetwork_2hl_64_64(config.num_in_pol, config.num_out_pol, hidden_dim=config.hidden_dim).to(
            self.device)
        self.target_critic = CLS_VNetwork_2hl_64_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

    def get_action(self, s):  # input 前后车距 output 滞留时间
        action_mean, action_sigma = self.policy(s)
        # print("action mean,action sigma:",action_mean,action_sigma)
        pi = Normal(loc=action_mean, scale=action_sigma)
        action = torch.clamp(pi.sample([1]), -0.3, 0.3) #try softmax?
        # action = pi.sample([1])
        return action

"""Model_4_1(delta time without 0.3)"""
class Model4_1(CLS_BaseModel):
    model_name = "Model_4_DELTACONTROL_Full1"
    deltaflag = True
    def __init__(self,config=None):
        super().__init__(config)
        self.policy = CLS_PNetwork_2hl_64_64(config.num_in_pol, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.critic = CLS_VNetwork_2hl_64_64(config.num_in_critic, 1 ,hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = CLS_PNetwork_2hl_64_64(config.num_in_pol, config.num_out_pol, hidden_dim=config.hidden_dim).to(
            self.device)
        self.target_critic = CLS_VNetwork_2hl_64_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

    def get_action(self, s):  # input 前后车距 output 滞留时间
        action_mean, action_sigma = self.policy(s)
        # print("action mean,action sigma:",action_mean,action_sigma)
        pi = Normal(loc=action_mean, scale=action_sigma)
        action = pi.sample([1]) #try softmax?
        # action = pi.sample([1])
        return action

"""Model_5(global control)"""
class Model5(CLS_BaseModel):
    model_name = "Model_5_GlobalControl"
    use_global_input = True
    def __init__(self, config=None):
        super().__init__(config)
        self.policy = CLS_PNetwork_2hl_64_64(config.num_in_pol, config.num_out_pol,hidden_dim=config.hidden_dim*2,hidden_dim2=config.hidden_dim).to(self.device)
        self.critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = CLS_PNetwork_2hl_64_64(config.num_in_pol, config.num_out_pol,
                                                    hidden_dim=config.hidden_dim*2,hidden_dim2=config.hidden_dim).to(
            self.device)
        self.target_critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

"""Model_6(global delta control)"""
class M6_PNetwork(CLS_PNetwork_1hl_64):
    def forward(self, X):

        h1 = F.elu(self.fc1(X[...,[0,-1]]))
        action_mean = F.elu(self.fc21(h1))
        action_sigma = F.softplus(self.fc22(h1))
        return action_mean, action_sigma

class Model6(CLS_BaseModel):
    model_name = "Model_6_GlobalDeltaControl"
    deltaflag = True
    use_global_input = True
    def __init__(self, config=None):
        super().__init__(config)
        self.policy = M6_PNetwork(2, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = M6_PNetwork(2, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.target_critic = CLS_VNetwork_1hl_64(config.num_in_critic, 1, hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

    def get_action(self, s):  # input 前后车距 output 滞留时间
        #local_s=[s[0],s[-1]]
        local_s = torch.tensor(s).to(self.device)
        action_mean, action_sigma = self.policy(local_s)
        # print("action mean,action sigma:",action_mean,action_sigma)
        pi = Normal(loc=action_mean, scale=action_sigma)
        action = torch.tanh(pi.sample([1]))*0.3  # try softmax?
        # action = pi.sample([1])
        return action

    def compute_loss(self, observation_batch, obs_all_batch, action_batch, return_batch):
        """
        observation_batch --- list of states
        action_batch   ---- list of actions
        """

        observation_batch = torch.tensor(observation_batch)
        # print(observation_batch.shape)
        # print(f"observation_batch=={observation_batch}")
        # 1. （s,a,G(s))从list变为cuda中的tensor
        # print(f"observation_batch_post=={observation_batch[:,[0,-1]]}")
        observation_batch = torch.from_numpy(np.asarray(observation_batch[:,[0,-1]])).float().to(self.device)
        obs_all_batch = torch.from_numpy(np.asarray(obs_all_batch)).float().to(self.device)
        action_batch = torch.from_numpy(np.asarray(action_batch)).float().to(self.device)
        return_batch = torch.from_numpy(np.asarray(return_batch)).float().to(self.device)
        # 2. 计算actor的Loss，pi/pi_old，计算critic的损失
        values, old_action_log_probs, action_log_probs = self.evaluate_actions(observation_batch, obs_all_batch,
                                                                             action_batch)
        adv_targ = -(torch.squeeze(values.detach()) - torch.squeeze(return_batch))
        ratio = torch.exp(action_log_probs - torch.clamp(old_action_log_probs, -20, 20))
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(torch.squeeze(return_batch), torch.squeeze(values))
        return action_loss, value_loss