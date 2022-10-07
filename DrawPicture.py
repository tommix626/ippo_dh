#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 11:41:00 2020

@author: Tom
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output, Image
import time
import os


def gif_init(fig,i,stop_pos,buses,pax_at_stop,stop_name):
    
    #fig = plt.figure(i+1)
    #print(i+1)
    if i==0:
        ax = fig.add_subplot(211)
    else:
        ax = fig.add_subplot(212)
    ax.plot(stop_pos, np.zeros(len(stop_pos)), '.-')
    buses_plot = []
    bus_num = len(buses)
    for i in range(bus_num):
        _, = ax.plot(buses[i].loc, 0, '*', markersize=16,label='%d' %buses[i].id)
        buses_plot.append(_)
    pax_bar = plt.bar(stop_pos, pax_at_stop, width=0.1, align='center', color='indianred', alpha=0.5,tick_label=stop_name )
    ax.set_title('Simulation using Artificial Data', fontsize=20)
    ax.set_xlabel('Distance along route M1', fontsize=16)
    ax.set_ylabel('Pax at Stop', fontsize=16)
    ax.set_xlim(-0.1,6.6)
    ax.set_ylim(-1, 60)
    # ax.legend()
    # plt.legend(loc=2)
    # plt.show()
    return ax,buses_plot,pax_bar
    
def gif_update(dir,fig,ax,buses_plot,buses,pax_bar,pax_at_stop,t):
    #    ---画图并刷新----
    bus_num = len(buses)
    print("bus num:",bus_num)
    if bus_num > len(buses_plot):
        _, = ax.plot(buses[-1].loc, 0, '*', markersize=16,label='%d' %buses[-1].id)
        buses_plot.append(_)
    elif bus_num < len(buses_plot):
        buses_plot.pop(0).remove()
    for i in range(bus_num):
        buses_plot[i].set_xdata(buses[i].loc)
    for px, h in zip(pax_bar, pax_at_stop):
        px.set_height(h)
    ax.legend()
    plt.legend(loc=2)
    #plt.show()
    print("direction:",dir)
    print("Time elapsed: %i seconds"%(t))
    if dir == 1:
        print("Pax at each stop: ", pax_at_stop[::-1])
    else:
        print("Pax at each stop: ", pax_at_stop)
    #time.sleep(1/100000000000) # set a global time equivalent parameter
    
    
def plot_tsd(traj,sim_horizon,bus_num,stop_loc,fig_dir,episode=None):
    #palette=sns.color_palette("hls", bus_num)
    # select a random set of colours from the xkcd palette
    xkcd = []
    for i in range(bus_num):
        xkcd.append(list(sns.xkcd_rgb.keys())[i])
    #print(xkcd)
    for i in range(len(traj)): # traj:: self.trajectory[dir][b.id].append([b.emit_time,b.trajectory,b.hold_action,b.hold_loc])
        f = plt.figure()
        ax = plt.gca()
        #colors = plt.cm.jet(np.linspace(0, 1,bus_num))
        f.set_size_inches((6, 4))
        plt.xlim([0, sim_horizon + 3])
        plt.ylim(0, 3.2)
        # plt.yticks(stop_loc,
        #                 ['1', '2', '3', '4', '5', '6', '7', '8',
        #                 '9', '10', '11', '12','13','14','15','16','17','18','19','20','21','22','23','24'])
        plt.yticks(stop_loc,
                        ['1', '2', '3', '4', '5', '6', '7', '8',
                        '9', '10', '11', '12','13','14','15','16','17']) #TODO:also change here
        # plt.yticks(stop_loc,
        #                 ['1', '2', '3', '4', '5', '6'])
        #单个图例
        labels = []
        for j in range(len(traj[i])):
            # print("i=",i,"j=",j)
            for data in traj[i][j]:
                start_t = data[0]
                # end_t = data[1]
                tr = np.array(data[1])
                # if end_t >= start_t and data[1]!=[]: 
                #print(y)
                #plt.scatter(x,y,c=x, cmap=cmp, norm=norm, alpha=0.7)
                #plt.plot([i for i in range(start_t,end_t+1)], y, '-', label='bus  ' + str(j),c = sns.color_palette("hls", bus_num)[j])
                if j not in labels: 
                    plt.plot([i_ for i_ in range(start_t,start_t+len(tr))], tr, '-', label='bus  ' + str(j),c=sns.xkcd_rgb[xkcd[j]],lw=1)
                    labels.append(j)
                else:
                    plt.plot([i_ for i_ in range(start_t,start_t+len(tr))], tr, '-',c=sns.xkcd_rgb[xkcd[j]],lw=1)
                tr_len = len(data[1])
                hold_len = len(data[2])
                assert tr_len == hold_len,("trajectory length(",tr_len,")should be equal to the length of holding action (",hold_len,").")
                # try:
                #     assert tr_len == hold_len,(f"trajectory length({tr_len})should be equal to the length of holding action ({hold_len}).")
                #     # print(tr_len,hold_len)
                # except:
                #     # print(data[2])
                maskscatter = np.ma.array(np.array(data[3]), mask=np.array(data[3])== 0.)
                normalize = mpl.colors.Normalize(vmin=0, vmax=180)
                plt.scatter([i_ for i_ in range(start_t,start_t+len(data[3]))], maskscatter, c=data[2], norm=normalize,cmap='binary' )
        plt.xlabel('Time step' )
        plt.ylabel('Station' )
        plt.yticks()
        plt.xticks()
        ax.legend(labelspacing=0.05)
        #plt.legend(loc=2)
        plt.legend(bbox_to_anchor=(1,0),loc=3,borderaxespad=0)
        if episode:
            f.savefig(os.path.join(fig_dir,"Tr_d"+str(i)+"_ep"+str(episode)+".svg"), bbox_inches='tight')
        else:
            plt.show()
        


# def gif_init(stop_pos,buses,pax_at_stop,stop_name):
#     fig = plt.figure(figsize=(20,8))
#     ax = fig.add_subplot(111)

#     ax.plot(stop_pos, np.zeros(len(stop_pos)), '.-')
#     buses_plot = []
#     bus_num = len(buses)
#     for i in range(bus_num):
#         _, = ax.plot(buses[i].loc, 0, '*', markersize=16,label='%d' %buses[i].id)
#         buses_plot.append(_)
#     pax_bar = plt.bar(stop_pos, pax_at_stop, width=0.1, align='center', color='indianred', alpha=0.5,tick_label=stop_name )
#     ax.set_title('Simulation using Artificial Data', fontsize=20)
#     ax.set_xlabel('Distance along route M1', fontsize=16)
#     ax.set_ylabel('Pax at Stop', fontsize=16)
#     ax.set_xlim(-0.1,6.6)
#     ax.set_ylim(-1, 60)
#     ax.legend()
#     plt.legend(loc=2)
#     plt.show()
#     return fig,ax,buses_plot,pax_bar
    
# def gif_update(fig,buses_plot,buses,pax_bar,pax_at_stop,t):
#     #    ---画图并刷新----
#     bus_num = len(buses_plot)
#     print("bus num:",bus_num)
#     for i in range(bus_num): 
#         buses_plot[i].set_xdata(buses[i].loc)
#     for px, h in zip(pax_bar, pax_at_stop):
#         px.set_height(h)

#     clear_output(wait=True)
#     display(fig)
    
#     print("Time elapsed: %i seconds"%(t))
#     print("Pax at each stop: ", pax_at_stop)
#     time.sleep(1/10000000) # set a global time equivalent parameter
