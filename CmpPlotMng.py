import os
import pickle
import pandas as pd
from typing import Type, List, Union

import matplotlib.pyplot as plt
import matplotlib

from utils.PPO import Model0_C, Model1, Model2, Model3, Model4, CLS_BaseModel
from utils.hyperparameters_origin import EnvConfig, Config
from utils.misc import comb_path


ModelList: List[Type[CLS_BaseModel]] = [Model0_C, Model1, Model2, Model3, Model4]
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

ModelSeedList = [[],
                 [1260],
                 [608],
                 [262],
                 [35]
                ]
# ModelSeedList = [[],
#                  [],
#                  [],
#                  [],
#                  [262,1,10,100,101,102,103,106]
#                 ]

if __name__ == "__main__":
    fig = plt.figure(1)
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_ylabel("Passenger Waiting Time")
    ax1.set_xlabel("Simulation TimeStep")
    ax2 = plt.subplot(2, 3, 4)
    ax2.set_ylabel("Minimum Headway")
    ax2.set_xlabel("Simulation TimeStep")
    ax3 = plt.subplot(2, 3, 5)
    ax3.set_ylabel("Bus Holding Time")
    ax3.set_xlabel("Simulation TimeStep")

    # # 选择ax1
    # plt.sca(ax1)
    # plt.plot()   #(x, y1, 'r-.')
    # plt.ylim(-1.5, 1.5)  # 限定y axis
    #
    # # 选择ax2
    # plt.sca(ax2)
    # plt.plot(x, y2, 'g--')
    # plt.ylim(-1.5, 1.5)
    #
    # # 选择ax3
    # plt.sca(ax3)
    # plt.plot(x, y3, 'b--')
    # plt.ylim(-1.5, 1.5)

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
            seed_folder = os.path.join(root,str(seed_id))
            pkl_file_path =  os.path.join(seed_folder,"para","post_training_data.pickle")
            with open(pkl_file_path,"rb") as f:
                data = pickle.load(f)
                [MinHeadway_overT, PaxNumList_overTBus, TotalPaxNum_overT, PaxWaitingTime_overT, BusHoldingTime_overT] = data
                smoothing_window = 5000
                warming_period = 1000
                PaxWaitingTime_overT_smoothed = pd.Series(PaxWaitingTime_overT).rolling(smoothing_window,
                                                                                        min_periods=smoothing_window).mean()
                MinHeadway_overT_smoothed = pd.Series(MinHeadway_overT).rolling(smoothing_window,
                                                                                min_periods=smoothing_window).mean()
                BusHoldingTime_overT_smoothed = pd.Series(BusHoldingTime_overT).rolling(smoothing_window,
                                                                                        min_periods=smoothing_window).mean()
                plt.sca(ax1)
                plt.plot(PaxWaitingTime_overT_smoothed[warming_period:], label=model.model_name + str(seed_id))
                plt.sca(ax2)
                plt.plot(MinHeadway_overT_smoothed[warming_period:], label=model.model_name + str(seed_id))
                plt.sca(ax3)
                plt.plot(BusHoldingTime_overT_smoothed[warming_period:], label=model.model_name + str(seed_id))
                # break
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    print(lines, labels)
    fig.legend(lines, labels, loc = 'lower right')
    plt.show()


if __name__ == "__main__":
    fig = plt.figure(1)
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_ylabel("Passenger Waiting Time")
    ax1.set_xlabel("Simulation TimeStep")
    ax2 = plt.subplot(2, 3, 4)
    ax2.set_ylabel("Minimum Headway")
    ax2.set_xlabel("Simulation TimeStep")
    ax3 = plt.subplot(2, 3, 5)
    ax3.set_ylabel("Bus Holding Time")
    ax3.set_xlabel("Simulation TimeStep")

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
        DataArray = [[],[],[]]
        for seed_id in ModelSeedList[model_id]:
            seed_folder = os.path.join(root,str(seed_id))
            pkl_file_path =  os.path.join(seed_folder,"para","key_var.pickle")
            with open(pkl_file_path,"rb") as f:
                data = pickle.load(f)
                [AvgMinHeadway_overT, _, _, AvgPaxWaitingTime_overT, AvgBusHoldingTime_overT] = data
                DataArray[0].append(AvgMinHeadway_overT)
                DataArray[1].append(AvgPaxWaitingTime_overT)
                DataArray[2].append(AvgBusHoldingTime_overT)
                # break
    plt.plot(DataArray[0])
    plt.plot(DataArray[1])
    plt.plot(DataArray[2])
    plt.show()