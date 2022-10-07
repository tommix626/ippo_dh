#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODOs:
change the bus number and run through the code
check the hyperparas and figure out what is the problem
export MPLBACKEND=Agg
"""
# parse arguments
import sys

if (len(sys.argv) == 4):
    cuda_id = int(sys.argv[1])
    startseed = int(sys.argv[2])
    seedsearchlen = int(sys.argv[3])
    print(sys.argv)
else:
    cuda_id = eval(input("cuda device="))
    startseed = eval(input("startseed="))
    seedsearchlen = eval(input("seedsearchlen="))
import os

from utils.misc import check_seed, comb_path

os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np

np.random.seed(0)
from Env import Env
from DrawPicture import gif_init, gif_update, plot_tsd
import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma
import math
import matplotlib
from utils.PPO import Model3_NEWReward as PPO_Model  # NOTE: Choose model here
import pickle
from IPython.display import display, clear_output, Image

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

from utils.hyperparameters_origin import EnvConfig,Config
import torch
# torch.use_deterministic_algorithms(True)

import random

random.seed(0)
import time

import pickle

"""
Structure:
//Model_1/TRAIN_x_x_x_x/123/
"""
endseed = startseed + seedsearchlen
# set root folder by model name
root = os.getcwd()
model_folder_name = PPO_Model.model_name
root = comb_path(root, model_folder_name)

config = Config()
seed_folder_name = "TRAIN_" + str(config.w1) + "_" + str(config.w2) + "_" + str(config.a_lr) + "_" + str(
    config.c_lr) + "_" + str(config.ppo_clip_param)
root = comb_path(root, seed_folder_name)

# copy hyperparameters file
org_file = os.path.join(os.getcwd(), "utils", "hyperparameters.py")
command = "cp " + org_file + " " + root
os.system(command)

# recording the success seeds
recordfilename = f"record{startseed}-{endseed}.txt"
recordfiledir = os.path.join(root, recordfilename)
recordfile = open(recordfiledir, "w")
recordfile.write("StartRecording:\n")

for seed_id in range(startseed, endseed):
    torch.manual_seed(seed_id)
    np.random.seed(seed_id)
    random.seed(seed_id)
    seed_flag = 1
    print(f"seed{seed_id} simulation started:")
    #######-------参数设置----#######
    envConfig = EnvConfig()

    Hold_Strategy = envConfig.Hold_Strategy  # 0 - no control; 1 - schedule-based 2 - forward headway, 3- two-way headway based # 4- Jiawei

    # result_dname = "result_test_best_replication"
    if Hold_Strategy == 4:
        result_dname = str(seed_id) + "STRAIN_" + str(config.w1) + "_" + str(config.w2) + "_" + str(
            config.a_lr) + "_" + str(config.c_lr) + "_" + str(config.ppo_clip_param)
        # print("the current directory is:", result_dname, " ,is this correct? end the process if not!!")
        # print("(w1,w2,aLR,cLR,clipE):",config.w1,config.w2,config.a_lr,config.c_lr,config.ppo_clip_param)
        # safe_check = input()
        # if safe_check=="remote" or safe_check=="r":
        #     result_dname = "R"+result_dname
        #     print("Chnage to remote. Directory is:", result_dname, "confirm?")
        #     safe_check = input()
        # elif safe_check=="y" or safe_check=="yes":
        #     pass
        # else:
        #     result_dname = result_dname+"_"+safe_check
        #     print("Add Suffix. Directory is:", result_dname)
    else:
        result_dname = "TMP"
        print("Directory is:TMP!!!!!")
    # ---存储结果"/result"---
    # print("the current directory is:",result_dname," ,is this correct? end the process if not!!")
    # safe_check = input()

    if Hold_Strategy == 0:
        # tmp = os.path.join(root,result_dname)
        result_path = os.path.join(root, "no_control")
    elif Hold_Strategy == 1:
        # tmp = os.path.join(root,result_dname)
        result_path = os.path.join(root, "schedule_based")
    elif Hold_Strategy == 2:
        # tmp = os.path.join(root,result_dname)
        result_path = os.path.join(root, "forward_headway")
    elif Hold_Strategy == 3:
        # tmp = os.path.join(root,result_dname)
        result_path = os.path.join(root, "two_way")
    elif Hold_Strategy == 4:
        result_path = os.path.join(root, str(seed_id))
    elif Hold_Strategy == 5:
        # tmp = os.path.join(root,result_dname)
        result_path = os.path.join(root, "me")

    if os.path.exists(result_path):
        print(f"simulation exists at {result_path}, delete it manually to train model again.")
        continue
    else:
        os.makedirs(result_path)
    logname = "log.txt"
    log_dir = os.path.join(result_path, logname)
    logfile = open(log_dir, "w")
    logfile.write(
        f"seedID={seed_id}\n\nw1={config.w1},w2={config.w2},aLR={config.a_lr},cLR={config.c_lr},clipE={config.ppo_clip_param}\n")

    fig_dir = comb_path(result_path, "fig")
    para_dir = comb_path(result_path, "para")
    model_dir = comb_path(result_path, "saved_agents")

    ####------rl setting ------
    if Hold_Strategy == 4:
        # 模型
        model = PPO_Model(config)
        model.load_w(model_dir)
        reward_set = []
        reward_p1_set = []
        reward_p2_set = []
        actor_loss_set = []
        v_loss_set = []

    line = [[] for i in range(envConfig.Dir_Num)]
    ax = []
    buses_plot = []
    pax_bar = []
    rollout_num = 2

    # ave_wait_list,std_wait_list,ave_travel_list,std_travel_list,ave_all_list,std_all_list = [],[],[],[],[],[]
    for i_ep in range(envConfig.num_episodes):
        if seed_flag == 0:
            logfile.write(f"\nnot passing seed_check at ep={i_ep}\nAborting.")
            # print(f"seed{seed_id} failed.Aborting...")
            break
        ave_reward_each_rollout = []
        ave_var_each_rollout = []
        ave_reward_p1_each_rollout = []
        ave_reward_p2_each_rollout = []
        for rolloutidx in range(rollout_num):  # FIXME what is rollout_num

            # env = Env(bus_stop_num=envConfig.Stop_Num,bus_num = envConfig.bus_num_each_dir,\
            #           stop_loc_list=envConfig.Stop_Loc_List,arr_rates = envConfig.Arr_Rates,bus_omega= envConfig.bus_omega,hold_strategy=Hold_Strategy,\
            #               pax_saturate=envConfig.pax_saturate,ran_travel=envConfig.ran_travel,mu_time = envConfig.mu_time,headway=envConfig.headway,\
            #                   alight_rate=envConfig.alight_rate,board_rate=envConfig.board_rate,warm_up_time = envConfig.warm_up_time,road_length=envConfig.road_length,envConfig=envConfig)
            env = Env(envConfig=envConfig)
            step_t = 0
            var = []
            start_time = time.time()
            while step_t <= envConfig.Sim_Horizon:
                # update environment

                s = env.update(step_t)  # NOTE: Main Update of the system

                if step_t > envConfig.warm_up_time:
                    if Hold_Strategy == 4:
                        actions = [-2 for _ in range(envConfig.bus_num)]  # action range -1~1 if controlled
                        actions = np.array(actions, dtype=float)
                        values = np.zeros(envConfig.bus_num, dtype=float)
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
                                current_obs = torch.from_numpy(np.array(s[i])).float().to(config.device)
                                with torch.no_grad():
                                    a = model.get_action(current_local_obs)  # 执行动作
                                    # v = model.get_value(current_obs)#获得模型
                                actions[i] = a.view(
                                    -1).cpu().numpy()  # FIXME what is view:change shape & what num stands for what action
                                # values[i] = v.view(-1).cpu().numpy()
                                # if env.bus_list[i].catch == True:
                                #     actions[i] = 1.0
                                #     with torch.no_grad():
                                #         v = model.get_value(current_obs)
                                #     values[i] = v.view(-1).cpu().numpy()
                                # else:
                                #     with torch.no_grad():
                                #         a = model.get_action(current_local_obs)#执行动作
                                #         v = model.get_value(current_obs)#获得模型
                                #     actions[i] = a.view(-1).cpu().numpy()
                                #     values[i] = v.view(-1).cpu().numpy()
                                # print("bus id ,驻留：",i,actions[i],step_t)

                        if exist_ctl == True:  # if any bus has controling actions
                            # to avoid special case, we update it whether ot not there is bus that need control
                            action_real = env.control(actions, deltaflag=PPO_Model.deltaflag,
                                                      step_t=step_t)  # this update the hold_time of the CLS_Bus
                            # 把动作，状态，奖励存进memory中
                            model.rollouts.insert(step_t, local_obs, s, action_real, config.w1,
                                                  config.w2)  # FIXME what is model.rollouts : savedatas
                step_t += 1
            # 仿真结束
            end_time = time.time()
            # print("cost time:",end_time-start_time)
            logfile.write("cost time:{}\n".format(end_time - start_time))
            print(("cost time:{}\n".format(end_time - start_time)))
            if Hold_Strategy == 4:
                ave_reward = 0
                ave_reward_p1 = 0
                ave_reward_p2 = 0
                length = 0
                # f = plt.figure()

                for i in range(len(model.rollouts.rewards)):
                    # print(i,len(model.rollouts.rewards[i]))
                    temp_r = np.array(model.rollouts.rewards[i])
                    temp_r1 = np.array(model.rollouts.rewards_p1[i])
                    temp_r2 = np.array(model.rollouts.rewards_p2[i])
                    length += temp_r.shape[0]
                    ave_reward_p1 += temp_r1.sum()
                    ave_reward_p2 += temp_r2.sum()
                    ave_reward += temp_r.sum()
                    # plt.plot(list(range(len(model.rollouts.actions[i]))),model.rollouts.actions[i])
                ave_reward /= length
                ave_reward_p1 /= length
                ave_reward_p2 /= length
                ave_var = sum(var) / len(var)
                # print(' num_episodes:%d rollout index:%d  r:%g   realvar:%g' % (i_ep,rolloutidx, ave_reward,ave_var))
                logfile.write(
                    f' num_episodes:{i_ep:d} rollout index:{rolloutidx:d}  r:{ave_reward:g}   realvar:{ave_var:g}')
                ave_reward_each_rollout.append(ave_reward)
                ave_reward_p1_each_rollout.append(ave_reward_p1)
                ave_reward_p2_each_rollout.append(ave_reward_p2)
            env.save_traj(step_t)

            # NOTE:ploting for headway map
            if (i_ep * rollout_num + rolloutidx) % 15 == 0:
                plot_tsd(env.trajectory, envConfig.Sim_Horizon, env.bus_num, envConfig.Stop_Loc_List[0], fig_dir, i_ep)

            # ave_w,std_w,ave_tr,std_tr,ave_all,std_all = env.cal_ave_time()
            # ave_wait_list.append(ave_w)
            # std_wait_list.append(std_w)
            # ave_travel_list.append(ave_tr)
            # std_travel_list.append(std_tr)
            # ave_all_list.append(ave_all)
            # std_all_list.append(std_all)

            # 仿真结束，更新网络
            if Hold_Strategy == 4:
                if (i_ep * rollout_num + rolloutidx) % 30 == 0:
                    seed_flag = check_seed(reward_set, reward_p1_set, reward_p2_set)  # NOTE:Check the seed flag

                for i in range(envConfig.bus_num):
                    for a_idx in range(
                            len(model.rollouts.actions[i]) - 1):  # for every seg(action<->next action) in trajectory
                        start_t = model.rollouts.time[i][a_idx]
                        end_t = model.rollouts.time[i][a_idx + 1]
                        # 其他bus的駐留時間
                        for j in range(1, envConfig.bus_num):
                            for_ind = (i - j) % envConfig.bus_num
                            tmp = np.array(env.hold_record[for_ind][start_t:end_t])
                            hold_t = np.sum(tmp > 0) / 180.0
                            model.rollouts.observations_all[i][a_idx].append(
                                hold_t)  # eg:3buses,4 actions performed:[[[前车停留时长,前前车,...][][][]],[[][][][]],[[][][][]]]
                    model.rollouts.observations_all[i].remove(
                        model.rollouts.observations_all[i][-1])  # ?why remove the last one
                    # caculate value function
                    obs_tmp = torch.from_numpy(np.array(model.rollouts.observations_all[i])).float().to(config.device)
                    with torch.no_grad():
                        v = model.get_value(obs_tmp)
                    v_cpu = v.view(-1).cpu().numpy().tolist()
                    # print(len(v_cpu))
                    model.rollouts.value_preds[i].extend(v_cpu)

                model.rollouts.compute_returns(config.GAMMA)
                model.rollouts.after_epoch()

        if Hold_Strategy == 4:
            reward_set.append(sum(ave_reward_each_rollout) / rollout_num)
            reward_p1_set.append(sum(ave_reward_p1_each_rollout) / rollout_num)
            reward_p2_set.append(sum(ave_reward_p2_each_rollout) / rollout_num)
            value_loss, action_loss = model.update(model.rollouts)  # NOTE:update the network(training)
            model.rollouts.after_update()
            v_loss_set.append(value_loss)
            actor_loss_set.append(action_loss)
            model.save_w(model_dir)

    # ave_wait = np.mean(ave_wait_list)
    # std_wait = np.mean(std_wait_list)
    # ave_travel = np.mean(ave_travel_list)
    # std_travel = np.mean(std_travel_list)
    # ave_alltime = np.mean(ave_all_list)
    # std_alltime = np.mean(std_all_list)
    # print("平均等待时间：",ave_wait)
    # print("标准差：",std_wait)
    # print("平均乘车时间：",ave_travel)
    # print("标准差：",std_travel)
    # print("average journey time:",ave_alltime)
    # print("std:",std_alltime)

    # close logging
    logfile.close()
    if (seed_flag):
        print(f"seed{seed_id} simulation ended successfully:")
        recordfile.write(f"success_seed{seed_id}\n")

        # pickle save reward, loss data
        pickle_data = [reward_set, reward_p1_set, reward_p2_set, actor_loss_set, v_loss_set]
        pklname = "reward_loss_set.pickle"
        log_dir = os.path.join(para_dir, pklname)
        with open(log_dir, 'wb') as f:
            pickle.dump(pickle_data, f)

        if Hold_Strategy == 4:  # ploting after all training
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
            a_loss_set_smoothed = pd.Series(actor_loss_set).rolling(smoothing_window,
                                                                    min_periods=smoothing_window).mean()
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


    else:
        recordfile.write("seed fail\n")

recordfile.close()
