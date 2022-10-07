#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program is used to generate datas of key var for several model. and plot them out
Available Vars:
- total reward(r,r1,r2)
- minimum headways (one for all models,and one for each traditional methods in distinct color)
- pax waiting time,travel time, all time (wt,tt,at)
- pax distributions
- bus average holding time (one for all models,and one for each traditional methods in distinct color)
- headway variance
#FIXME:When changing model:
1. change import model class
2.change model name folder
"""
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import matplotlib
from typing import Type, List, Union

from DrawPicture import plot_tsd
from Env import Env

from utils.PPO import *

# from .singlemodel_datasimulator import collect_main_data

from utils.hyperparameters_origin import EnvConfig,Config
from utils.misc import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# NOTE:matplotlib settings
# matplotlib.rc("font",**{"family":"sans-serif","sans-serif":["Helvetica","Arial"],"size":14})
matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=1)
matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
matplotlib.rc("axes", unicode_minus=False, linewidth=1, labelsize='medium')
matplotlib.rc("axes.formatter", limits=[-7, 7])
# matplotlib.rc('savefig', bbox='tight', format='pdf', frameon=False, pad_inches=0.05)
matplotlib.rc('lines', marker=None, markersize=4)
matplotlib.rc('text', usetex=False)
matplotlib.rc('xtick', direction='in')
matplotlib.rc('xtick.major', size=8)
matplotlib.rc('xtick.minor', size=2)
matplotlib.rc('ytick', direction='in')
matplotlib.rc('lines', linewidth=1)
matplotlib.rc('ytick.major', size=8)
matplotlib.rc('ytick.minor', size=2)
matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
matplotlib.rc('mathtext', fontset='stixsans')
matplotlib.rc('legend', fontsize='small', frameon=False,
              handleheight=0.5, handlelength=1, handletextpad=0.1, numpoints=1)

ModelList: List[Type[CLS_BaseModel]] = [Model0_N, Model0_C, Model1, Model2, Model3, Model4, Model4_1]
ModelSeedList = [[],[0],
                 [1260],
                 [],
                 [],
                 [],
                 list(reversed(range(1,40)))
                ]
ModelSeedList = [[],[0],
                 [1260],
                 [],
                 [],
                 [],
                 [31]
                ]
DisplayNameList = [[],"Dual-Headway",
                 "RL",
                 [],
                 [],
                 [],
                 "IPPO-D"
                ]

def average(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


class CLS_ModelData(object):
    def __init__(self,name,sid):
        self.name = name
        self.sid=sid
        self.datadict = {}

def plot1():

    # baseliney1,baseliney2 = [],[]
    # for t in All_ModelDataList[0].datadict.keys():
    #     [MinHeadway_overT, PaxNumList_overTBus, TotalPaxNum_overT, BusHeadwayVariance_overT, r, r1, r2, wtL, ttL, atL,
    #      hold_actionL] = All_ModelDataList[0].datadict[t]
    #     baseliney1.append(np.log(np.var(wtL)))
    #     [MinHeadway_overT, PaxNumList_overTBus, TotalPaxNum_overT, BusHeadwayVariance_overT, r, r1, r2, wtL, ttL, atL,
    #      hold_actionL] = All_ModelDataList[1].datadict[t]
    #     baseliney2.append(np.log(np.var(wtL)))
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle('Holding Action')
    ax = fig.add_subplot(121)
    ax.set_ylabel("Holding Action Mean")
    ax.set_xlabel("Passenger Density")
    ax2 = fig.add_subplot(122)
    ax2.set_ylabel("Holding Action Variance")
    ax2.set_xlabel("Passenger Density")
    ax2.set_yscale('log')
    # ax.set_yscale('log')
    for md in All_ModelDataList:

        # plt.ylim(0, 15000)
        ploty = []
        plotx = []
        for t in md.datadict.keys():
            plotx.append(t)
            [MinHeadway_overT, PaxNumList_overTBus, TotalPaxNum_overT, BusHeadwayVariance_overT, r, r1, r2, wtL, ttL, atL, hold_actionL] = md.datadict[t]
            # pax_divergence = [np.var(x) for x in PaxNumList_overTBus]
            ploty.append(np.mean(hold_actionL))
        ax.plot(plotx, ploty, label=md.name)

        ploty = []
        plotx = []
        for t in md.datadict.keys():
            plotx.append(t)
            [MinHeadway_overT, PaxNumList_overTBus, TotalPaxNum_overT, BusHeadwayVariance_overT, r, r1, r2, wtL, ttL,
             atL, hold_actionL] = md.datadict[t]
            # pax_divergence = [np.var(x) for x in PaxNumList_overTBus]
            ploty.append(np.var(hold_actionL))
        ax2.plot(plotx, ploty, label=md.name)

    ax.legend()
    ax2.legend()
    fig.savefig(os.path.join(os.getcwd(), "fig_output", "HC.svg"), bbox_inches='tight')
    fig.show()
    plt.close()


if __name__ == "__main__":
    # fig = plt.figure(1)
    # ax1 = plt.subplot(2, 1, 1)
    # ax1.set_ylabel("Passenger Waiting Time")
    # ax1.set_xlabel("Simulation TimeStep")
    # ax2 = plt.subplot(2, 3, 4)
    # ax2.set_ylabel("Minimum Headway")
    # ax2.set_xlabel("Simulation TimeStep")
    # ax3 = plt.subplot(2, 3, 5)
    # ax3.set_ylabel("Bus Holding Time")
    # ax3.set_xlabel("Simulation TimeStep")

    All_ModelDataList: List[CLS_ModelData] = []
    for model_id in range(len(ModelList)):
        model = ModelList[model_id]
        # set root(model) folder //MODEL_X/TRAIN_x_x_x_x
        config = Config()
        hypername = "TRAIN_" + str(config.w1) + "_" + str(config.w2) + "_" + str(config.a_lr) \
                    + "_" + str(config.c_lr) + "_" + str(config.ppo_clip_param)
        root = os.path.join(os.getcwd(), model.model_name, hypername)
        if not os.path.exists(root):
            print(f"model not exist")
            raise RuntimeError
        for seed_id in ModelSeedList[model_id]:
            MD = CLS_ModelData(DisplayNameList[model_id],seed_id)
            print(f"{ModelList[model_id].model_name} (seed {seed_id}):")
            seed_folder = os.path.join(root, str(seed_id))
            if not (os.path.exists(seed_folder)):
                print(f"seed{seed_id} not found.")
                continue
            for t in [x / 1000 for x in range(10, 45, 5)]:
                pkl_file_path = os.path.join(seed_folder, str(t)+"_final_training_data.pickle")
                with open(pkl_file_path, "rb") as f:
                    data = pickle.load(f)
                    MD.datadict[t] = data
            All_ModelDataList.append(MD)

# input()
plot1()

