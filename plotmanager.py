#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program is used to generate plots for a seed / multiple seeds.
Available Plots:
- minimum headways (one for all models,and one for each traditional methods in distinct color)
- pax percentage with time (one for each model)
- pax average waiting time (one for each model,cross compare with traditional methods)
- bus average holding time (one for all models,and one for each traditional methods in distinct color)

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

from DrawPicture import plot_tsd
from Env import Env
# from utils import PPO_answer as PPO

from utils.PPO import Model6 as PPO_Model #NOTE: Choose model here
VERSION_ID = 2

from utils.hyperparameters import EnvConfig,Config
from utils.misc import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


def average(l):
    if len(l)==0:
        return 0
    return sum(l)/len(l)

def main_plot(seed_folder_name,seed_id): #ploting for one config setting.
    output_dir = comb_path(os.path.join(seed_folder_name,"fig"),"post_training")
    model_dir = comb_path(seed_folder_name, "saved_agents")
    para_dir = comb_path(seed_folder_name, "para")

    ####------rl setting ------
    model = PPO_Model(config)
    model.load_w(model_dir,use_cpu=True)

    line = [[] for i in range(envConfig.Dir_Num)]


    MinHeadway_overT = []
    PaxNumList_overTBus = []  # pax number over time. eac element is a list that contains #of pax on wach bus: self[i_ep][bus_id]=PaxNum on bus(bus_id) at at time t
    TotalPaxNum_overT = []  # total pax number at time t
    PaxWaitingTime_overT = []  # the average of all pax waiting time at time t
    BusHoldingTime_overT = []  # the average bus holding time at time t
    env = Env(envConfig=envConfig)
    step_t = 0
    var = []
    while step_t <= envConfig.Sim_Horizon:
        # update environment
        s = env.update(step_t)  # NOTE: Main Update of the system
        # Data Core
        MinHeadway,PaxNumList,TotalPaxNum,AvgPaxWaitingTime,AvgBusHoldingTime,_ = env.collect_data() # collect plotting data
        MinHeadway_overT.append(MinHeadway)
        PaxNumList_overTBus.append(PaxNumList)
        TotalPaxNum_overT.append(TotalPaxNum)
        PaxWaitingTime_overT.append(AvgPaxWaitingTime)
        BusHoldingTime_overT.append(AvgBusHoldingTime)

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
                    if PPO_Model.use_global_input == True:
                        local_obs[i].extend(s[i])
                    else:
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
                action_real = env.control(actions,deltaflag=PPO_Model.deltaflag,step_t=step_t) # needed, this update the hold_time of the CLS_Bus
        step_t += 1

    env.save_traj(step_t)

    # NOTE:ploting for headway map
    plot_tsd(env.trajectory, envConfig.Sim_Horizon, env.bus_num, envConfig.Stop_Loc_List[0], output_dir, 100)
    print(f"seed {seed_id} finish simulating")

    #NOTE:Save post training data
    pickle_data = [MinHeadway_overT, PaxNumList_overTBus, TotalPaxNum_overT, PaxWaitingTime_overT, BusHoldingTime_overT]
    pklname = "post_training_data.pickle"
    log_dir = os.path.join(para_dir, pklname)
    with open(log_dir, 'wb') as f:
        pickle.dump(pickle_data, f)
    print(f"seed {seed_id} post_training data saved")

    #NOTE:plotting

    ax =[]
    # NOTE:plot MinHeadway
    f = plt.figure()
    ax = plt.subplot(111)
    ax.autoscale()
    # ax.set_ylim(bottom=0.)
    plt.xlabel('Simulation TimeStep')
    plt.ylabel('Minimum Headway')
    smoothing_window = 300
    # MinHeadway_overT = MinHeadway_overT[5000:] #used to not count the spike at the first 5000 step
    MinHeadway_overT_smoothed = pd.Series(MinHeadway_overT).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(MinHeadway_overT, alpha=0.2)
    plt.plot(MinHeadway_overT_smoothed, color='orange')
    plt.grid()
    #plt.show()
    f.savefig(os.path.join(output_dir, "min_headway.png"), bbox_inches='tight')
    f.savefig(os.path.join(output_dir, "min_headway.pdf"), bbox_inches='tight')

    # NOTE:plot PaxPercentage
    f = plt.figure()
    ax = plt.subplot(111)
    ax.autoscale()
    ax.set_ylim(bottom=0.,top=1.)
    plt.xlabel('Simulation TimeStep')
    plt.ylabel('Bus Load Percentage')
    smoothing_window = 3000 #for the plot to be less chaos
    for bus_idx in range(len(PaxNumList_overTBus[0])):
        PaxPercentage_overT = np.array(PaxNumList_overTBus)[:,bus_idx]/np.array(TotalPaxNum_overT)
        PaxPercentage_overT_smoothed = pd.Series(PaxPercentage_overT).rolling(smoothing_window,
                                                                    min_periods=smoothing_window).mean()
        plt.plot(PaxPercentage_overT_smoothed, alpha=0.2)
        plt.plot(PaxPercentage_overT_smoothed, color='orange')
    plt.grid()
    #plt.show()
    f.savefig(os.path.join(output_dir, "pax_percentage.png"), bbox_inches='tight')
    f.savefig(os.path.join(output_dir, "pax_percentage.pdf"), bbox_inches='tight')

    # NOTE:plot PaxWaitingTime
    f = plt.figure()
    ax = plt.subplot(111)
    ax.autoscale()
    # ax.set_ylim(bottom=0.)
    plt.xlabel('Simulation TimeStep')
    plt.ylabel('Passenger Waiting Time')
    smoothing_window = 300
    PaxWaitingTime_overT_smoothed = pd.Series(PaxWaitingTime_overT).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(PaxWaitingTime_overT, alpha=0.2)
    plt.plot(PaxWaitingTime_overT_smoothed, color='orange')
    plt.grid()
    f.savefig(os.path.join(output_dir, "pax_waiting_time.png"), bbox_inches='tight')
    f.savefig(os.path.join(output_dir, "pax_waiting_time.pdf"), bbox_inches='tight')

    # NOTE:plot BusHoldingTime_overT
    f = plt.figure()
    ax = plt.subplot(111)
    ax.autoscale()
    # ax.set_ylim(bottom=0.)
    plt.xlabel('Simulation TimeStep')
    plt.ylabel('Bus Holding Time')
    smoothing_window = 300
    BusHoldingTime_overT_smoothed = pd.Series(BusHoldingTime_overT).rolling(smoothing_window,
                                                                            min_periods=smoothing_window).mean()
    plt.plot(BusHoldingTime_overT, alpha=0.2)
    plt.plot(BusHoldingTime_overT_smoothed, color='orange')
    plt.grid()
    f.savefig(os.path.join(output_dir, "bus_holding_time.png"), bbox_inches='tight')
    f.savefig(os.path.join(output_dir, "bus_holding_time.pdf"), bbox_inches='tight')

    print("Plot finished")
    return [average(MinHeadway_overT), PaxNumList_overTBus, average(TotalPaxNum_overT), average(PaxWaitingTime_overT), average(BusHoldingTime_overT)]

def save_post_training_var(key_var_file_path_noprefix,data):
    f2 = open(key_var_file_path_noprefix+".pickle","wb")
    pickle.dump(data,f2)
    [AvgMinHeadway, PaxNumList_overTBus, AvgTotalPaxNum, AvgPaxWaitingTime, AvgBusHoldingTime] = data
    f1 = open(key_var_file_path_noprefix+".txt","w")
    f1.write(f"AvgMinHeadway={AvgMinHeadway}\n")
    f1.write(f"AvgPaxWaitingTime={AvgPaxWaitingTime}\n")
    f1.write(f"AvgBusHoldingTime={AvgBusHoldingTime}\n")
    f1.close()
    f2.close()

def compare_seeds(seed_id,key_var_pickle_file_path):
    with open(key_var_pickle_file_path,"rb") as f:
        [AvgMinHeadway, PaxNumList_overTBus, AvgTotalPaxNum, AvgPaxWaitingTime, AvgBusHoldingTime] = pickle.load(f)
    if AvgMinHeadway>best_seed_values["AvgMinHeadway"]:
        best_seed_ids["AvgMinHeadway"] = seed_id
        best_seed_values["AvgMinHeadway"] = AvgMinHeadway

    if AvgPaxWaitingTime<best_seed_values["AvgPaxWaitingTime"]:
        best_seed_ids["AvgPaxWaitingTime"] = seed_id
        best_seed_values["AvgPaxWaitingTime"] = AvgPaxWaitingTime

    if AvgBusHoldingTime<best_seed_values["AvgBusHoldingTime"]:
        best_seed_ids["AvgBusHoldingTime"] = seed_id
        best_seed_values["AvgBusHoldingTime"] = AvgBusHoldingTime

"""
    # NOTE:ploting after all training TODO:no need loss plot
    ax = []

    # PLOT critic loss
    f = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel('Training episode')
    plt.ylabel('Mean squared error')
    smoothing_window = 10
    v_loss_set_smoothed = pd.Series(v_loss_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(v_loss_set, alpha=0.2)
    plt.plot(v_loss_set_smoothed, color='orange')
    plt.grid()
    # plt.show()FIXME
    f.savefig(os.path.join(fig_dir, "critic.pdf"), bbox_inches='tight')

    # plot actor loss
    f = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel('Training episode')
    plt.ylabel('Actor loss')
    smoothing_window = 10
    a_loss_set_smoothed = pd.Series(actor_loss_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(actor_loss_set, alpha=0.2)
    plt.plot(a_loss_set_smoothed, color='orange')
    plt.grid()
    # plt.show()FIXME
    f.savefig(os.path.join(fig_dir, "actor.pdf"), bbox_inches='tight')

    # plot reward
    f = plt.figure()
    ax = plt.subplot(111)
    ax.tick_params(length=4, width=0.5)
    plt.xlabel('Training episode')
    plt.ylabel('Cumulative global reward')
    smoothing_window = 10
    rewards_smoothed = pd.Series(reward_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed1 = pd.Series(reward_p1_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed2 = pd.Series(reward_p2_set).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(reward_set, alpha=0.2)
    plt.plot(rewards_smoothed, color='orange', label='total reward')
    plt.plot(reward_p1_set, alpha=0.2)
    plt.plot(rewards_smoothed1, color='red', label='reward for holding penalty')
    plt.plot(reward_p2_set, alpha=0.2)
    plt.plot(rewards_smoothed2, color='green', label='reward for headway equalization')
    ax.legend(loc='best', fancybox=True, shadow=False, ncol=1, prop={'size': 12})
    # plt.show()FIXME
    f.savefig(os.path.join(fig_dir, "reward_train.pdf"), bbox_inches='tight')

    #NOTE:copy hyperpara file FIXME
    org_file = os.path.join(os.getcwd(), "utils", "hyperparameters.py")
    command = "cp " + org_file + " " + result_path
    os.system(command)
"""

if __name__ == "__main__":
    #model0:null
    # 1:v2
    # 2:v2
    # 3:v2
    # 4:v2
    version_id = VERSION_ID
    print(f"version_id = {version_id},Enter to continue.")
    input()
    envConfig = EnvConfig()
    config = Config()

    # set root(model) folder //MODEL_X/
    root = os.path.join(os.getcwd(), PPO_Model.model_name)
    if not os.path.exists(root):
        print(f"model not exist")
        raise RuntimeError

    hypername = "TRAIN_" + str(config.w1) + "_" + str(config.w2) + "_" + str(config.a_lr) \
                + "_" + str(config.c_lr) + "_" + str(config.ppo_clip_param)
    seedList = os.listdir(comb_path(root, hypername))
    seedList.sort(reverse=True)
    # print(f"seedList={seedList}")
    best_seed_ids = {"AvgMinHeadway":-1,"AvgPaxWaitingTime":-1,"AvgBusHoldingTime":-1}
    best_seed_values = {"AvgMinHeadway": 0, "AvgPaxWaitingTime": 9999999, "AvgBusHoldingTime": 9999999, "reward":0}
    for seed_id in seedList:
        # seed_id = "102"
        seed_folder_name = comb_path(comb_path(root, hypername), seed_id)
        if not os.path.isdir(seed_folder_name):
            continue
        if not os.path.exists(os.path.join(seed_folder_name, "fig", "reward_train.pdf")):
            print(f"seed{seed_id} did not finish successfully, deleting...")
            command = "rm -r " + seed_folder_name
            os.system(command)
            continue
        if os.path.exists(os.path.join(seed_folder_name, "fig", "post_training")):
            version_id_file_path = os.path.join(seed_folder_name, "fig", "post_training","version_id.txt")
            key_var_file_path_noprefix = os.path.join(seed_folder_name, "para","key_var")
            if os.path.exists(version_id_file_path):
                with open(version_id_file_path,"r") as f:
                    id = int(f.readline().strip("\n"))
                    if id == version_id:
                        print(f"seed {seed_id} already up-to-date.")
                        compare_seeds(seed_id,key_var_file_path_noprefix+".pickle")
                        continue
                    else:
                        print(f">>>updating seed {seed_id} from version {id} to version {version_id}...")
                        data = main_plot(seed_folder_name, int(seed_id))
                        save_post_training_var(key_var_file_path_noprefix,data)
                        compare_seeds(seed_id, key_var_file_path_noprefix + ".pickle")



                with open(version_id_file_path,"w") as f:
                    f.write(str(version_id))
                    print("<<<Done.")
            else: #first time processing seed (should not happen normally)
                print(f"Faulty version id case!!! care for seed: {seed_id}")
                with open(version_id_file_path,"w") as f:
                    f.write(str(version_id))
                    data = main_plot(seed_folder_name, int(seed_id))
                    save_post_training_var(key_var_file_path_noprefix, data)
                    compare_seeds(seed_id, key_var_file_path_noprefix + ".pickle")
        else:
            os.makedirs(os.path.join(seed_folder_name, "fig", "post_training"))
            para_dir = comb_path(seed_folder_name, "para")
            version_id_file_path = os.path.join(seed_folder_name, "fig", "post_training","version_id.txt")
            key_var_file_path_noprefix = os.path.join(para_dir, "key_var")
            with open(version_id_file_path, "w") as f:
                f.write(str(version_id))
                data = main_plot(seed_folder_name, int(seed_id))
                save_post_training_var(key_var_file_path_noprefix, data)
                compare_seeds(seed_id, key_var_file_path_noprefix + ".pickle")

    print(best_seed_ids)