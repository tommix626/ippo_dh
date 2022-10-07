#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


# class RlConfig(object):
#     def __init__(self):

class EnvConfig(object):
    def __init__(self):
        self.Hold_Strategy = 4

        self.Stop_Num = 17  # 6
        self.bus_num_each_dir = 8  # 每個方向
        # self.mu_time = 3.5 # min
        self.mu_v = 27  # km/h
        self.sigma_v = 4.5  # km/h
        self.headway = 8 * 60  # /s
        # self.road_length = np.pi*4# np.pi*2
        self.road_length = 9210  # m
        self.alight_rate = 0.55
        self.board_rate = 0.33

        self.bus_omega = self.mu_v / 3.6
        # self.bus_omega = self.road_length / (self.Stop_Num-1)/(self.mu_time*60)
        self.sim_num = 1
        self.Sim_Horizon = 60 * 60 * 7
        self.num_episodes = 100
        self.Dir_Num = 2
        self.warm_up_time = 10 * 60
        self.bus_num = self.bus_num_each_dir * self.Dir_Num

        # arr_rates = [0.5 / 60 / 2, 0.5 / 60 / 2,   0.5 / 60 / 1.2, 0.5 / 60,      \
        #               0.5 /60 * 1.2,  0.5 * 2 /60,    1.5 / 60,   0.5 / 60 * 3, 0.5 / 60 * 3.5, 0.5 / 60 * 3.5, 0.5 / 60 * 4, 0.5 / 60 * 3.2,0.5 / 60 * 3, 0.5 / 60 * 2.5, 0.5 / 60 * 2,   0.5 / 60 * 2,   0.5 / 60 * 1.2, 0.5 / 60 * 1.2, 0.5 / 60,   0.5 / 60,     0.5 / 60 / 1.5, 0.5 / 60 / 1.8, 0.5 / 60 / 2, 0]
        arr_rates = [0.5 / 60 / 2, 0.5 / 60 / 1.8, 0.5 / 60 * 1.5, 0.5 * 2 / 60, 0.5 / 60 * 3, 0.5 / 60 * 3.5,
                     0.5 / 60 * 4, \
                     0.5 / 60 * 3.2, 0.5 / 60 * 3, 0.5 / 60 * 2.5, 0.5 / 60 * 2, 0.5 / 60 * 1.2, 0.5 / 60,
                     0.5 / 60 / 1.5, 0.5 / 60 / 2, 0.5 / 60 / 4, 0]

        arr_rates_rev = arr_rates
        self.Arr_Rates = [arr_rates, arr_rates_rev]
        # 0.5 / 60 * 3,0.5 / 60 / 2
        # self.Arr_Rates = [[0.5 / 60 / 2, 0.5 / 60 / 2, 0.5 / 60 / 1.2, 0.5 / 60, 1.5 / 60, 0],[0.5 / 60 * 4,0.5 / 60 * 2, 0.5 / 60,0.5 / 60 / 1.5, 0.5 / 60 / 1.8, 0]]
        # self.Arr_Rates = [[0.5 / 60 / 2/2, 0.5 / 60 / 2/2, 0.5 / 60 / 1.2/2, 0.5 / 60/2, 1.5 / 60/2, 0],[0.5 / 60 * 4/2,0.5 / 60 * 2/2, 0.5 / 60/2,0.5 / 60 / 1.5/2, 0.5 / 60 / 1.8/2, 0]]

        node_loc = [0, 160, 460, 550, 950, 1440, 1700, 2080, 2340, 3030, 3350, 3470, 3760, 4230, 4560, 5040, 5220, 5820,
                    6340, 7080, 7300, 8010, 8150, 8540, 9020, 9210]
        intersec_flag = [False, True, True, False, False, True, False, True, False, False, True, False, True, True,
                         False, False, False, False, False, False, False, False, True, True, False, False]

        init_phase = [0, 1, 1, 0, 0, 1, 0, 0, 1]
        pass_time = [20, 20, 20, 20, 20, 20, 20, 20, 20]
        p1_time = [30, 30, 30, 30, 30, 30, 30, 30, 30]
        p2_time = [30, 30, 30, 30, 30, 30, 30, 30, 30]

        init_phase_rev = init_phase[::-1]
        self.Init_Phase_List = [init_phase, init_phase_rev]
        pass_time_rev = pass_time[::-1]
        self.Pass_Time_List = [pass_time, pass_time_rev]
        p1_time_rev = p1_time[::-1]
        self.P1_Time_List = [p1_time, p1_time_rev]
        p2_time_rev = p2_time[::-1]
        self.P2_Time_List = [p2_time, p2_time_rev]

        stop_loc = []
        intersec_loc = []
        for i in range(len(node_loc)):
            if intersec_flag[i] == True:
                intersec_loc.append(node_loc[i])
            else:
                stop_loc.append(node_loc[i])
        node_loc_rev = node_loc[::-1]
        self.Node_Loc_List = [node_loc, node_loc_rev]

        intersec_loc_rev = intersec_loc[::-1]
        self.Intersec_Loc_List = [intersec_loc, intersec_loc_rev]

        # print(stop_loc)
        stop_loc_rev = stop_loc[::-1]
        self.Stop_Loc_List = [stop_loc, stop_loc_rev]
        stop_name = [str(i + 1) for i in range(self.Stop_Num)]
        stop_name = np.array(stop_name)
        # stop_name = np.array(['1', '2', '3', '4', '5', '6', '7', '8','9', '10', '11', '12',
        # '13','14','15','16','17','18','19','20','21','22','23','24'])
        # stop_name_rev = stop_name[::-1]
        stop_name_rev = stop_name
        self.stop_names = [stop_name, stop_name_rev]

        # 隨機性
        self.pax_saturate = False
        self.ran_travel = True


import torch
import math


class Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # bus
        self.num_agents = 16  # 6

        # PPO controls
        self.ppo_epoch = 1
        self.num_mini_batch = 16  # 6
        self.ppo_clip_param = 0.5

        # a2c controls
        self.USE_GAE = True
        self.gae_tau = 0.5
        self.a_lr = 0.0002  # 0.0002
        self.c_lr = 0.0002

        # network input and output
        self.num_in_pol = 16
        self.num_out_pol = 2
        self.num_in_critic = 16 * 2 - 1  # 6*2
        self.hidden_dim = 16

        # reward weight
        self.w1 = 0.6 #for less holding time
        self.w2 = 1  #for headway equalization

        # optimizer
        self.Patience = 10
        self.grad_clip = 1

        # Multi-step returns
        self.N_STEPS = 10

        # epsilon variables
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 30000
        self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (
                    self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

        # misc agent variables
        self.GAMMA = 0.8
