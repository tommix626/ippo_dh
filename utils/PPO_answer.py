# -*- coding: UTF-8 -*-
from torch.distributions.normal import Normal
import torch.nn.functional as F
#from .networks import MLPNetwork
from .policy import PolicyNetwork
from .value import ValueNetwork
from .misc import hard_update
from .Rollout import RolloutStorage
from torch.optim import Adam
import torch.nn as nn

import torch
import numpy as np
import os
import torch.optim.lr_scheduler as lr_scheduler
class Model(object):
    model_name = "Model_0"
    def __init__(self,config=None):

        self.device = config.device
        self.policy = PolicyNetwork(config.num_in_pol, config.num_out_pol,hidden_dim=config.hidden_dim).to(self.device)
        self.critic = ValueNetwork(config.num_in_critic, 1,hidden_dim=config.hidden_dim).to(self.device)
        self.target_policy = PolicyNetwork(config.num_in_pol, config.num_out_pol, hidden_dim=config.hidden_dim).to(self.device)
        self.target_critic = ValueNetwork(config.num_in_critic, 1,hidden_dim=config.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.a_lr)
        self.a_scheduler = lr_scheduler.StepLR(self.policy_optimizer, step_size=20, gamma=0.9)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.c_lr)
        self.c_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=20, gamma=0.9)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

        self.clip_param = config.ppo_clip_param
        self.num_mini_batch = config.num_mini_batch
        self.ppo_epoch = config.ppo_epoch
        self.grad_clip = config.grad_clip

        self.rollouts = RolloutStorage(config)

        self.value_losses = []
        self.policy_losses = []


    def get_action(self,s): #input 前后车距 output 滞留时间
        action_mean, action_sigma = self.policy(s)
        # print("action mean,action sigma:",action_mean,action_sigma)
        pi = Normal(loc=action_mean,scale=action_sigma)
        action = torch.clamp(pi.sample([1]),0,1)
        #action = pi.sample([1])
        return action

    def get_value(self,s):
        value = self.target_critic(s)
        # print("value:",value)
        return value

    def save_loss(self, policy_loss, value_loss):
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)

    def evaluate_actions(self, obs, obs_all,actions):
        # old_p_out = self.target_policy(s)
        # old_mean = old_p_out[:,0]
        # old_sigma = old_p_out[:,1]
        # old_mean,old_sigma = self.target_policy(s)
        # pi_old = Normal(loc=old_mean, scale=old_sigma)
        
        old_mean,old_sigma = self.target_policy(obs) #the policy(aka actor) only knows local observations(forward and backward timespace)
        pi_old = Normal(loc=torch.squeeze(old_mean), scale=torch.squeeze(old_sigma))
        
        old_action_log_probs = pi_old.log_prob(actions) #however, the critic has the ability to see the global states

        # new_p_out = self.policy(s)
        # mean = new_p_out[:,0]
        # sigma = new_p_out[:,1]
        #pi = Normal(loc=mean,scale=sigma)
        mean,sigma = self.policy(obs)
        pi = Normal(loc=torch.squeeze(mean), scale=torch.squeeze(sigma))
        # print(mean.shape)
        # print(sigma.shape)
        # print(actions.shape)
        action_log_probs = pi.log_prob(actions)

        values = self.critic(obs_all) # value:a mean and a sigma of current critic

        return values, old_action_log_probs, action_log_probs
    # def compute_loss(self, observation_batch,action_batch,return_batch):
    #     """
    #     observation_batch --- list of states
    #     action_batch   ---- list of actions
    #     """
    #     #1. （s,a,G(s))从list变为cuda中的tensor
    #     observation_batch  = torch.from_numpy(np.array(observation_batch)).float().to(self.device)
    #     action_batch = torch.from_numpy(np.array(action_batch)).float().to(self.device)
    #     return_batch = torch.from_numpy(np.array(return_batch)).float().to(self.device)
    #     #2. 计算actor的Loss，pi/pi_old，计算critic的损失
    #     values, old_action_log_probs,action_log_probs = self.evaluate_actions(observation_batch, action_batch)
    #     #print(values.shape)
    #     #print(return_batch.shape)
    #     #adv_targ = -(torch.squeeze(values)[:-1] - return_batch)#此处会将critic网络和actor网络的计算图连在一起，因此做values的co0y分开两张图
    #     adv_targ = -(torch.squeeze(values.detach())[:-1] - return_batch)

    #     ratio = torch.exp(action_log_probs - old_action_log_probs)[:-1]
    #     #print(adv_targ.shape)
    #     #print(ratio.shape)
    #     surr1 = ratio * adv_targ
    #     surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
    #     action_loss = -torch.min(surr1, surr2).mean()

    #     value_loss = F.mse_loss(return_batch, torch.squeeze(values)[:-1])#这里应该是不合适的，相当于把Q(s,a)逼近V(s)
    #     return action_loss, value_loss

    def compute_loss(self, observation_batch,obs_all_batch,action_batch,return_batch):
        """
        observation_batch --- list of states
        action_batch   ---- list of actions
        """
        #1. （s,a,G(s))从list变为cuda中的tensor
        observation_batch  = torch.from_numpy(np.asarray(observation_batch)).float().to(self.device)
        obs_all_batch  = torch.from_numpy(np.asarray(obs_all_batch)).float().to(self.device)
        action_batch = torch.from_numpy(np.asarray(action_batch)).float().to(self.device)
        return_batch = torch.from_numpy(np.asarray(return_batch)).float().to(self.device)
        #2. 计算actor的Loss，pi/pi_old，计算critic的损失
        values, old_action_log_probs,action_log_probs = self.evaluate_actions(observation_batch, obs_all_batch,action_batch)
        # print(values.shape)
        # print(return_batch.shape)
        # print(action_log_probs.shape)
        #adv_targ = -(torch.squeeze(values)[:-1] - return_batch)#此处会将critic网络和actor网络的计算图连在一起，因此做values的co0y分开两张图
        adv_targ = -(torch.squeeze(values.detach()) - torch.squeeze(return_batch))
        # print("actions:", action_batch)
        # print("action prob:",old_action_log_probs)
        ratio = torch.exp(action_log_probs - torch.clamp(old_action_log_probs,-20,20))
        # print("ratio:",ratio)
        # print(adv_targ.shape)
        # print(ratio.shape)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(torch.squeeze(return_batch), torch.squeeze(values))
        return action_loss, value_loss

    # def mix_all_experience(self, rollout):
    #     agent_num = len(rollout.observations)
    #     for i in range(agent_num):
    #         rollout.observations_all_mix.extend(rollout.observations_all[i][:len(rollout.returns[i])])
    #         rollout.observations_mix.extend(rollout.observations[i][:len(rollout.returns[i])])
    #         rollout.actions_mix.extend(rollout.actions[i][:len(rollout.returns[i])])
    #         rollout.returns_mix.extend(rollout.returns[i])
    #     rollout.observations_mix = np.array(rollout.observations_mix)
    #     rollout.observations_all_mix = np.array(rollout.observations_all_mix)
    #     rollout.actions_mix = np.array(rollout.actions_mix)
    #     rollout.returns_mix = np.array(rollout.returns_mix )

    
    # def update(self, rollout):
    #     agent_num = len(rollout.rewards)
    #     value_loss_epoch = 0
    #     action_loss_epoch = 0
    #     #把(s,a,return)混到一起方面mini-Batch梯度下降
    #     # self.mix_all_experience(rollout)

    #     for i in range(agent_num):
    #         observation_batch = rollout.observations[i][:len(rollout.returns[i])]
    #         action_batch = rollout.actions[i][:len(rollout.returns[i])]
    #         return_batch = rollout.returns[i]
    #         # print("return batch:",return_batch)

    #         action_loss, value_loss = self.compute_loss(observation_batch,action_batch,return_batch)
    #         self.policy_optimizer.zero_grad()
    #         action_loss.backward()
    #         self.policy_optimizer.step()

    #         self.critic_optimizer.zero_grad()
    #         value_loss.backward()
    #         self.critic_optimizer.step()

    #         value_loss_epoch += value_loss.item()
    #         action_loss_epoch += action_loss.item()

    #     value_loss_epoch /= agent_num
    #     action_loss_epoch /= agent_num
        
    #     self.a_scheduler.step()
    #     self.c_scheduler.step()
        
    #     # self.a_scheduler.step(action_loss_epoch)
    #     # self.c_scheduler.step(value_loss_epoch)
        
    #     self.save_loss(action_loss_epoch, value_loss_epoch)
    #     print("actor loss:%g,critic loss:%g"%(action_loss_epoch, value_loss_epoch))
    #     hard_update(self.target_policy, self.policy)
    #     hard_update(self.target_critic, self.critic)
    #     return action_loss_epoch, value_loss_epoch

    def  update(self, rollout):
        value_loss_epoch = 0
        action_loss_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollout.feed_forward_generator(self.num_mini_batch)
            for sample in data_generator:
                observation_batch, obs_all_batch,action_batch, return_batch= sample
                action_loss, value_loss = self.compute_loss(observation_batch,obs_all_batch,action_batch,return_batch)
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

    def save_w(self,filepath):
        torch.save(self.policy.state_dict(), os.path.join(filepath,"policy.dump"))
        torch.save(self.critic.state_dict(), os.path.join(filepath,"critic.dump"))
        torch.save(self.policy_optimizer.state_dict(), os.path.join(filepath,"policy_optim.dump"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(filepath,"critic_optim.dump"))
        # torch.save(self.policy.state_dict(), './saved_agents/policy.dump')
        # torch.save(self.critic.state_dict(), './saved_agents/critic.dump')
        # torch.save(self.policy_optimizer.state_dict(), './saved_agents/policy_optim.dump')
        # torch.save(self.critic_optimizer.state_dict(), './saved_agents/critic_optim.dump')
    
    def load_w(self,filepath):
        fname_policy = os.path.join(filepath,"policy.dump")
        fname_critic = os.path.join(filepath,"critic.dump")
        fname_policy_optim = os.path.join(filepath,"policy_optim.dump")
        fname_critic_optim = os.path.join(filepath,"critic_optim.dump")

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

