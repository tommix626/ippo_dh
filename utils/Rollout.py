# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
class RolloutStorage(object):
    def __init__(self, config,gae_tau=0.95):
        self.num_agents = config.num_agents

        self.observations = [[] for _ in range(self.num_agents)]
        self.observations_all = [[] for _ in range(self.num_agents)]
        self.rewards = [[] for _ in range(self.num_agents)]
        self.rewards_p1 = [[] for _ in range(self.num_agents)]
        self.rewards_p2 = [[] for _ in range(self.num_agents)]
        self.actions = [[] for _ in range(self.num_agents)]
        self.returns = [[] for _ in range(self.num_agents)]# 等同于v_target
        self.value_preds = [[] for _ in range(self.num_agents)]
        self.time = [[] for _ in range(self.num_agents)]

        # all experience mix
        self.observations_mix = []
        self.observations_all_mix = []
        self.actions_mix = []
        self.returns_mix = []

        # self.value_preds = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        # self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        # self.action_log_probs = torch.zeros(num_steps, num_processes, 1).to(device)
        self.gae = config.USE_GAE
        self.gae_tau = gae_tau
        self.n_step = config.N_STEPS

    def after_epoch(self):
        self.observations = [[] for _ in range(self.num_agents)]
        self.observations_all = [[] for _ in range(self.num_agents)]
        self.rewards = [[] for _ in range(self.num_agents)]
        self.rewards_p1 = [[] for _ in range(self.num_agents)]
        self.rewards_p2 = [[] for _ in range(self.num_agents)]
        self.actions = [[] for _ in range(self.num_agents)]
        self.returns = [[] for _ in range(self.num_agents)]# 等同于v_target
        self.value_preds = [[] for _ in range(self.num_agents)]
        self.time = [[] for _ in range(self.num_agents)]
        #把(s,a,return)混到一起方面mini-Batch梯度下降
        # self.mix_all_experience()
    
    def after_update(self):
        # all experience mix
        self.observations_mix = []
        self.observations_all_mix = []
        self.actions_mix = []
        self.returns_mix = []
        
    def mix_all_experience(self):
        agent_num = len(self.observations)
        for i in range(agent_num):
            self.observations_all_mix.extend(self.observations_all[i][:len(self.returns[i])])
            self.observations_mix.extend(self.observations[i][:len(self.returns[i])])
            self.actions_mix.extend(self.actions[i][:len(self.returns[i])])
            self.returns_mix.extend(self.returns[i])
            

    def insert(self, t,current_obs,current_obs_all,actions,w1,w2):
        for agent_idx in range(len(actions)):
            if actions[agent_idx] != -2:
                if len(self.observations[agent_idx][:])>0:
                    #TODO: change this function only: p2 p1+p2
                    r_p1 = np.exp(-self.actions[agent_idx][-1]) #reward function part1
                    # r_p1 = -self.actions[agent_idx][-1]
                    # dis = np.array(current_obs[agent_idx][0:6])
                    # r_p2 = -np.var(dis)/np.square(np.mean(dis))
                    # print("r_p1,r_p2:",r_p1,r_p2)
                    # print("current_obs[agent_idx][0]:",current_obs[agent_idx][0])
                    # print("current_obs[agent_idx][1]:",current_obs[agent_idx][1])
                    r_p2 = np.exp(- abs((current_obs[agent_idx][0] - current_obs[agent_idx][1]))) -np.exp(-abs(current_obs[agent_idx][0] + current_obs[agent_idx][1])) #part 2
                    # print("r_p1,r_p2:",r_p1,r_p2)
                    r = w1 * r_p1 + w2 * r_p2
                    # if self.actions[agent_idx][-1]!= 0.0 and self.actions[agent_idx][-1]!=1.0:
                    #     print("action:",self.actions[agent_idx][-1])
                    #     print("headway:",current_obs[agent_idx][-1],current_obs[agent_idx][-2])
                    #     print("reward：",r)
                    self.rewards_p1[agent_idx].append(r_p1)
                    self.rewards_p2[agent_idx].append(r_p2)
                    self.rewards[agent_idx].append(r)
                self.observations[agent_idx].append(current_obs[agent_idx])
                self.observations_all[agent_idx].append(current_obs_all[agent_idx])
                self.actions[agent_idx].append(actions[agent_idx])
                self.time[agent_idx].append(t)
                # self.value_preds[agent_idx].append(values[agent_idx])

    def compute_returns(self, gamma):
        agent_num = len(self.returns)
        if self.gae:
            for i in range(agent_num):
                gae = 0
                # print(len(self.value_preds[i]),len(self.rewards[i]))
                for step in reversed(range(len(self.rewards[i])-1)):
                    delta = self.rewards[i][step] + gamma*self.value_preds[i][step + 1] - self.value_preds[i][step]
                    gae = delta + gamma*self.gae_tau * gae  #with discount factor:tau*gamma,accumulate reward!
                    r = gae + self.value_preds[i][step]
                    self.returns[i].append(r)
                self.returns[-i].reverse()
        else:
            for i in range(agent_num):
                for j in range(len(self.rewards[i])):
                    if j+self.n_step<= len(self.rewards[i]):
                        temp = 0
                        for k in reversed(range(self.n_step)):
                            temp = self.rewards[i][j+k]+gamma*temp
                            # temp = temp + gamma * self.rewards[i][j+k]
                        r = temp + (gamma**self.n_step)*self.value_preds[i][j+self.n_step]
                        self.returns[i].append(r)
        self.mix_all_experience()

    def feed_forward_generator(self, num_mini_batch):
        #
        self.observations_mix = np.array(self.observations_mix)
        self.observations_all_mix = np.array(self.observations_all_mix)
        self.actions_mix = np.array(self.actions_mix)
        self.returns_mix = np.array(self.returns_mix )
        
        batch_size = len(self.observations_mix)
        # print("batch size=",batch_size)
        assert batch_size >= num_mini_batch, (
            "PPO requires the experience size (",batch_size,") "
            "to be greater than or equal to the number of PPO mini batches (",num_mini_batch,").")
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
        for indices in sampler:
            observations_batch = self.observations_mix[indices]
            obs_all_batch = self.observations_all_mix[indices]
            actions_batch = self.actions_mix[indices]
            return_batch = self.returns_mix[indices]

            yield observations_batch, obs_all_batch,actions_batch, return_batch