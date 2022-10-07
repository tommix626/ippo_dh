#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program is used to generate datas of key var for a single model.
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
# from utils import PPO_answer as PPO

# from utils.PPO import Model1 as PPO_Model  # NOTE: Choose model here
from utils.PPO import *


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
ModelSeedList = [[0],[0],
                 [1260],
                 [],
                 [],
                 [],
                 [31]
                ]

# ModelSeedList = [[],[],
#                  [],
#                  [],
#                  [],
#                  [],
#                  [106,103,81,69,52, 31,11]
#                 ]

def average(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


def collect_main_data(seed_folder_name, PPO_Model):  # ploting for one config setting.
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    model_dir = comb_path(seed_folder_name, "saved_agents")

    ####------rl setting ------
    model = PPO_Model(config)
    model.load_w(model_dir,use_cpu=True)

    line = [[] for i in range(envConfig.Dir_Num)]

    MinHeadway_overT = []
    PaxNumList_overTBus = []  # pax number over time. eac element is a list that contains #of pax on wach bus: self[t][bus_id]=PaxNum on bus(bus_id) at at time t
    TotalPaxNum_overT = []  # total pax number at time t
    BusHeadwayVariance_overT = []
    env = Env(envConfig=envConfig)
    step_t = 0
    var = []
    while step_t <= envConfig.Sim_Horizon:
        # update environment
        s = env.update(step_t)  # NOTE: Main Update of the system
        # Data Core
        MinHeadway, PaxNumList, TotalPaxNum, _, _, BusHeadwayVariance = env.collect_data()  # collect plotting data
        MinHeadway_overT.append(MinHeadway)
        PaxNumList_overTBus.append(PaxNumList)
        TotalPaxNum_overT.append(TotalPaxNum)
        BusHeadwayVariance_overT.append(BusHeadwayVariance)
        if step_t > envConfig.warm_up_time:
            actions = [-2 for _ in range(envConfig.bus_num)]
            actions = np.array(actions, dtype=float)
            local_obs = [[] for _ in range(envConfig.bus_num)]
            exist_ctl = False
            for i in range(envConfig.bus_num):

                if len(s[i][:]) > 0:
                    # print("s[i]=",s[i])
                    var.append(1 / (1. + np.var(np.array(s[i]))))
                    exist_ctl = True
                    local_obs[i].extend([s[i][0], s[i][-1]])  # local_obs 只存前后车时距 s存的是对每个车来说所有其他车的时距
                    current_local_obs = torch.from_numpy(np.array(local_obs[i])).float().to(config.device)
                    with torch.no_grad():
                        a = model.get_action(current_local_obs)  # 执行动作
                        # v = model.get_value(current_obs)#获得模型
                    actions[i] = a.view(-1).cpu().numpy()
            if exist_ctl == True:
                # print(s)
                # 把动作，状态，奖励存进memory中
                # model.rollouts.insert(step_t, local_obs, s, actions, config.w1, config.w2)
                action_real = env.control(actions, deltaflag=PPO_Model.deltaflag,
                            step_t=step_t)  # needed, this update the hold_time of the CLS_Bus
                model.rollouts.insert(step_t, local_obs, s, action_real, config.w1,
                                      config.w2)
        step_t += 1

    # get reward
    r, r1, r2 = [], [], []
    for i in range(len(model.rollouts.rewards)):
        # print(i,len(model.rollouts.rewards[i]))
        r.extend(np.array(model.rollouts.rewards[i]))
        r1.extend(np.array(model.rollouts.rewards_p1[i]))
        r2.extend(np.array(model.rollouts.rewards_p2[i]))

    wtL, ttL, atL, hold_actionL = env.collect_final_data()

    env.save_traj(step_t)
    plot_tsd(env.trajectory, envConfig.Sim_Horizon, env.bus_num, envConfig.Stop_Loc_List[0], seed_folder_name, str(t)+"Final")

    # NOTE:Save post training data
    pickle_data = [MinHeadway_overT, PaxNumList_overTBus, TotalPaxNum_overT, BusHeadwayVariance_overT, r, r1, r2, wtL,
                   ttL, atL, hold_actionL]
    pklname = f"{t}_final_training_data.pickle"
    log_dir = os.path.join(seed_folder_name, pklname)
    with open(log_dir, 'wb') as f:
        pickle.dump(pickle_data, f)
    print(f"seed {seed_id} final_training data saved")
    return



"""generation
t=0.04
a=np.random.exponential(t, size=17)
while(abs(np.std(a)-t)>0.0001 or abs(np.mean(a)-t)>0.0001):
    a=np.random.exponential(t, size=17)

print (np.std(a),np.mean(a))
print(a)
"""
def envgenerator():
    # t = 0.01
    # arr_rates = [0.00285457, 0.00019458, 0.00037339, 0.03697479, 0.00134369,
    #              0.00271185, 0.0213734, 0.02208354, 0.00700832, 0.01772659,
    #              0.00131051, 0.01587619, 0.00567951, 0.01559059, 0.00185563,
    #              0.00361886, 0.01256782]
    # # yield t,arr_rates
    # t = 0.015
    # arr_rates = [0.04810887, 0.00940928, 0.00323046, 0.00236753, 0.04436006,
    #              0.00685864, 0.01112956, 0.00632335, 0.01287084, 0.01239672,
    #              0.0352973, 0.00918285, 0.03516291, 0.00396159, 0.00061681,
    #              0.00108315, 0.01262366]
    # # yield t, arr_rates
    t = 0.02
    arr_rates = [0.00767768, 0.0171793, 0.00971969, 0.00594411, 0.03065117,
                 0.00245527, 0.01628628, 0.01238343, 0.07147761, 0.00560562,
                 0.00411756, 0.00144332, 0.04189173, 0.02181648, 0.06237763,
                 0.01880144, 0.00958297]
    yield t, arr_rates
    # t = 0.025
    # arr_rates = [0.02877353, 0.01589829, 0.02762243, 0.03924829, 0.03783017,
    #              0.0080569, 0.02437668, 0.00503229, 0.00253752, 0.0051088,
    #              0.01272469, 0.01739619, 0.05576731, 0.00834145, 0.02519564,
    #              0.00291837, 0.10677463]
    # # yield t, arr_rates
    # t = 0.03
    # arr_rates = [0.05468323, 0.04301393, 0.00034501, 0.00221005, 0.07531798,
    #              0.0091944, 0.02576208, 0.02773578, 0.00788828, 0.0782551,
    #              0.00968824, 0.00178959, 0.00907959, 0.03357214, 0.02209274,
    #              0.10225648, 0.00649759]
    # # yield t, arr_rates
    # t = 0.035
    # arr_rates = [0.01668233, 0.02092653, 0.03598976, 0.05724902, 0.00394979,
    #              0.03212791, 0.00293937, 0.05127478, 0.01307801, 0.00491389,
    #              0.05030086, 0.08966435, 0.06805773, 0.00361388, 0.00076033,
    #              0.13032529, 0.01172887]
    # yield t, arr_rates
    t = 0.04
    arr_rates = [0.09865412, 0.05606727, 0.0010143, 0.06319319, 0.03313204,
                 0.1549655, 0.02618595, 0.06717562, 0.0221591, 0.07138617,
                 0.01530913, 0.0004627, 0.02362975, 0.01536047, 0.00763716,
                 0.00265792, 0.02000699]
    yield t, arr_rates
    # t = 0.045
    # arr_rates = [0.03262064, 0.01450096, 0.1403588 , 0.03212844, 0.03523919,
    #    0.00044912, 0.022066  , 0.06224605, 0.02236775, 0.01109016,
    #    0.06348899, 0.17438002, 0.05566844, 0.02712147, 0.04519498,
    #    0.01000966, 0.01501277]
    # yield t, arr_rates
    yield 0,0

if __name__ == "__main__":

    envConfig = EnvConfig()
    config = Config()
    # arr_rates = np.random.normal(loc=0.01439, scale=0.01, size=17)
    gen = envgenerator()
    t, arr_rates = next(gen)
    while(t!=0):
        print(f"simulating t={t}")
        envConfig.Arr_Rates = [arr_rates, arr_rates]
        model_folders = []
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
                print(f"{ModelList[model_id].model_name} (seed {seed_id}):")
                seed_folder = os.path.join(root, str(seed_id))
                if(os.path.exists(seed_folder)):
                    collect_main_data(seed_folder, model)
                else:
                    print(f"seed{seed_id} not found")


        t, arr_rates = next(gen)