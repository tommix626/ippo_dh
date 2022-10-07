#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:18:49 2020

@author: Tom
"""
import random

import numpy as np
import scipy.stats as stats
import math 
class CLS_bus():
    def __init__(self, w, dispatch_loc,stop_nums,stop_loc_list,node_loc_list,dir,hold_strategy, id, emit_time,alight_rate=0.55,board_rate=0.33):
        self.w = w
        self.loc = dispatch_loc
        self.dispatch_loc = dispatch_loc
        self.stop_nums = stop_nums
        self.dir = dir
        self.hold_strategy = hold_strategy
        self.alight_rate = alight_rate
        self.board_rate = board_rate
        self.node_loc_list = node_loc_list
        
        self.nextstop_ind = 0
        self.nextnode_ind = 0
        
        self.is_close = False
        self.hold_stop = None
        self.operate = True
        self.terminal = False
        self.hold_time = 0
        self.catch = False
        self.pax_num = 0# 车上乘客数 not int, use len(self.pax_list) instead to get accurate pax number
        self.capacity = 72
        
        self.move_flag = True
        self.at_stop = True
        self.Arrive = True
        # self.new_link = False
        self.need_RLcontrol = False
        
        self.pax_list = []
        self.alight_list = [0 for i in range(stop_nums)]
        self.phase = -1

        self.id = id
        self.trajectory = []
        self.hold_action = []
        self.hold_loc = []
        self.hold_record = [] #append every frame its current hold_time eg:0,0,4,3,2,1,0,0,0,2,1,0,0,...
        self.state = []
        self.emit_time = emit_time
        self.arr_time = 0
        self.first_dec = True
        self.first_arrive = 0
        if self.dir == 0:
            self.nextstop_ind = np.sum(self.loc > np.asarray(stop_loc_list)+self.w/2)
            self.nextnode_ind = np.sum(self.loc > np.asarray(node_loc_list)+self.w/2)
        else:
            self.nextstop_ind = np.sum(self.loc < np.asarray(stop_loc_list)-self.w/2)
            self.nextnode_ind = np.sum(self.loc < np.asarray(node_loc_list)+self.w/2)

        self.curtime = 0 #used for time update of pax
        # print(self.nextstop_ind)
        # print(self.id,self.loc)
                
    def board(self,stop_pax):
        self.pax_num += self.board_rate
        board_num = int(self.pax_num)-len(self.pax_list)
        ##Debug##
        if board_num == -1:
            print("board num:",board_num)
            print("pax num:",self.pax_num)
            print(len(self.pax_list))

        # board_num = min(int(self.pax_num)-len(self.pax_list),len(stop_pax))
        for i in range(board_num):
            self.pax_list.append(stop_pax[0])
            stop_pax[0].onboarding(self.curtime)
            self.alight_list[stop_pax[0].des_stop]+=1
            stop_pax.remove(stop_pax[0])
            
    def alight(self):
        # if self.nextstop_ind == 0:
        #     print("alighting")
        #下客：1. alight_list减 2.pax_list减 
        stop_id = self.nextstop_ind
        # alight_sum = math.ceil(self.alight_list[stop_id])
        # self.alight_list[stop_id] -= self.alight_rate
        # alight_num = alight_sum - math.ceil(self.alight_list[stop_id])
        self.alight_list[stop_id] -= self.alight_rate #self.alight_list[stop_id] is a number(counter) of how many ppl wants to go down at stop_id
        alight_sum = 0
        for i in range(len(self.pax_list)):
            if self.pax_list[i].des_stop == stop_id:
                alight_sum += 1
        alight_num = alight_sum - math.ceil(self.alight_list[stop_id])
        alight_num = int(alight_num)
        # print("alight_num:",alight_num)
        wait_time = []
        travel_time = []
        for i in range(alight_num):
            for j in range(len(self.pax_list)):
                if self.pax_list[j].des_stop == stop_id:
                    # if stop_id == 0:
                        # print("alighting , bus.id:",self.id)
                    self.pax_list[j].alighting(self.curtime)

                    if self.pax_list[j].label == 1:
                        # print(self.pax_list[j].wait_time)
                        wait_time.append(self.pax_list[j].wait_time)
                        travel_time.append(self.pax_list[j].travel_time)
                    self.pax_list.remove(self.pax_list[j])
                    self.pax_num -= 1
                    break
        wait_time = np.array(wait_time)
        travel_time = np.array(travel_time)
        all_time = wait_time+travel_time
        return wait_time,travel_time,all_time

    def at_stop_jdg(self,stop_loc_list):
        if abs(stop_loc_list[self.nextstop_ind] - self.loc) <= self.w/2:
            self.at_stop = True
        else:
            self.at_stop = False
    def at_intersec_jdg(self,intersec_loc_list):
        self.at_intersec = False
        self.intersec_ind = -1
        for i in range(len(intersec_loc_list)):
            if abs(intersec_loc_list[i]-self.loc)<=self.w/2:
                self.at_intersec = True
                self.intersec_ind = i
    def board_alight(self,pax_saturate,stop_pax):
        stop_id = self.nextstop_ind #當前所在站點
        wait_time = [] #记录等车时间
        travel_time = [] #记录乘车时间
        all_time = []
        if len(stop_pax) != 0:
            if pax_saturate == False or self.pax<self.capacity:
                self.board(stop_pax)
        if self.alight_list[stop_id] > 0:
            wait_time,travel_time,all_time = self.alight()
            # print(wait_time,travel_time)
        #------上下客结束了，关门，不允许新增乘客上车-------
        if self.alight_list[stop_id] <= 0 and stop_pax == []:
            # print("bus close, id,time:",self.id)
            self.pax_num = int(self.pax_num)#上下客結束後把車上乘客弄成整數
            self.alight_list[self.nextstop_ind] = 0
            self.is_close = True
        # print(self.pax_num)
        # print(len(self.pax_list))
        return wait_time,travel_time,all_time
    def restart(self,dir,stop_nums,t):
        # print("restart")
        ##---mark一下，這裏需要把不同變量分屬於哪些策略標注一下---
        self.nextstop_ind = 0
        self.nextnode_ind = 0
        self.dir = dir
        self.terminal = False
        self.is_close = False
        self.move_flag = True
        self.at_stop = True
        self.Arrive = True
        # self.new_link = True
        self.first_dec = True
        self.first_arrive = 0
        self.need_RLcontrol = False
        self.trajectory = []
        self.hold_action = []
        self.hold_loc = []
        self.emit_time = t
        # print("restart busid:",self.id)
        # print(len(self.pax_list))
        
        # print(self.alight_list)
        for pax in self.pax_list:
            # print(pax.des_stop)
            # if pax.des_stop != 5:
            #     print("not 5 pax des:",pax.des_stop)
            # else:
            #     print("5 pax des:",pax.des_stop)
            pax.des_stop = 0
        # print(self.pax_num)
        self.alight_list = [0 for i in range(stop_nums)]
        self.alight_list[0] = len(self.pax_list)
    

    def move(self,stop_loc_list,step_t):
        if self.dir == 0:
            self.loc += self.w
            self.nextstop_ind = np.sum(self.loc > np.asarray(stop_loc_list)+self.w/2)
            self.nextnode_ind = np.sum(self.loc > np.asarray(self.node_loc_list)+self.w/2)
        else:
            self.loc -= self.w
            self.nextstop_ind = np.sum(self.loc < np.asarray(stop_loc_list)-self.w/2)
            self.nextnode_ind = np.sum(self.loc < np.asarray(self.node_loc_list)+self.w/2)
        self.trajectory.append(self.loc)
        self.is_close = False
        self.first_dec = True
        self.need_RLcontrol = False
        	# self.hold_stop = None

    def stop(self,stop_loc_list,intersec_loc_list):
        # print("stop before:",self.id,self.loc)
        if self.at_stop == True:
            self.loc = stop_loc_list[self.nextstop_ind]
        if self.at_intersec == True:
            self.loc = intersec_loc_list[self.intersec_ind]
        # print("stop",self.id,self.loc)
        self.trajectory.append(self.loc)
    
    def update(self, stop_loc_list,intersec_loc_list,step_t,warm_up_time,curtime):
        self.curtime = curtime
        if step_t <= warm_up_time or self.need_RLcontrol == False:
            self.hold_action.append(0.)
            self.hold_loc.append(0.)
        self.hold_record.append(self.hold_time)
        if self.move_flag == True:
            self.move(stop_loc_list,step_t)
        else:
            self.stop(stop_loc_list,intersec_loc_list)
        # print(self.id,self.loc)
        # for pax in self.pax_list:
        #     pax.travel_time += 1
        # self.terminal(stop_loc_list,step_t)
    
    def move_judge(self,depart_times,t):
        ##(不在站点->移动+不需要控制；在站点，上下客未结束->不动+不需要控制;在站点，上下客结束了，如果hold策略为“no”，(可能前车下客没有结束)，如果超车->不动，如果不超车->移动)；移动+不需要控制;
        ###上下客结束了，如果hold策略不为“no”，hold还没有决策->不动+需要控制；在站点，上下客结束了，hold决策了，hold时间没有超过->不动+不需要控制;
        ####(在站点，上下客结束了，hold决策了，hold时间超过,但移动会超车->不动+需要控制;否则->移动+不需要控制)
        if self.at_stop == False:
            if self.at_intersec == False:
                self.move_flag = True
            else:
                if self.phase == 0:
                    self.move_flag = False
                else:
                    if self.catch == True:
                        self.move_flag = False
                    else:
                        self.move_flag = True
        elif self.is_close == False:
            self.move_flag = False
        elif self.hold_strategy == 0:
            if self.catch == True:
                self.move_flag = False
            if self.catch == False:
                self.move_flag = True
        elif self.hold_strategy == 1:
            #----如果没到离开时间，驻留->不动，如果到了->移动
            if depart_times[self.nextstop_ind]< t:
            	if self.catch == False:
            		self.move_flag = True
            	else:
            		self.move_flag = False
            else:
                self.move_flag = False
        elif self.hold_strategy == 2 or self.hold_strategy == 3:
            #----如果dwell time够了，移动，如果不够->不动
            if self.hold_time <= 0:
                if self.catch == False:
                    self.move_flag = True
                else:
                    self.move_flag = False
            else:
                self.move_flag = False
            #print("bus的dwell time为：",self.hold_time)
            self.hold_time -= 1
        elif self.hold_strategy == 4:
            #-------到站后位进行第一次决策->不动;
            if self.first_dec == True:
                self.move_flag=False
            else:
            #-------不是到站后第一次决策，如果dwell time够了且不超车，移动，否则不动并需要控制；如果dwell time不够则不动，驻留时间-1
                if self.hold_time <= 0:
                    if self.catch == False:
                        self.move_flag = True
                    else:
                        self.move_flag=False
                else:
                    self.move_flag = False
                    self.hold_time -= 1
    def new_arrive(self,):
        #-----用self.arrive记录是否新到站------
        if self.move_flag == True and self.at_stop == True:
            self.Arrive = True
        else:
            self.Arrive = False
    def wait_control_jdg(self,):
        self.need_RLcontrol = False
        if self.at_stop == True and self.is_close == True:
            if self.first_dec == True:
                self.need_RLcontrol = True
                self.first_dec = False
                self.first_arrive = 1
            # else:
            #     if self.hold_time<=0 and self.catch == True:
            #         self.need_RLcontrol = True

            
class CLS_bus_stop():
    def __init__(self, loc, i,wait_num,arr_rate,depart_schedule,stop_num):
        self.loc = loc
        self.i = i
        self.arr_rate = arr_rate
        # self.bus_pass_flag = False
        self.latest_bus_pass_time = 0
        self.depart_schedule = depart_schedule
        #-----初始化等车乘客数-----
        self.pax_list = []
        self.stop_num = stop_num
        for i in range(wait_num):
            p = CLS_pax(self.i, self.stop_num, label=0,curtime = 0)
            self.pax_list.append(p)
        self.curtime = 0
    def arrive(self):
        k = np.random.poisson(self.arr_rate, 1)[0]
        #self.wait_num += k
        for i in range(k):
            p = CLS_pax(self.i, self.stop_num,curtime=self.curtime) #FIXME: the last stop must not generate pax (for each line)
            self.pax_list.append(p)
        
    def proceed(self,curtime):
        # if self.bus_pass_flag == True:
        self.curtime =curtime
        self.arrive()
            
class CLS_pax(object):
    def __init__(self,o,stop_num,label=1,curtime = 0):
        self.org_stop = o
        if(o==stop_num-1):
            print("ERRERE!!")
        s = random.randint(1, stop_num-self.org_stop-1)
        self.des_stop = np.array(self.org_stop + s)
        if self.des_stop == 0:
            print("error:",0)
        if self.des_stop <= self.org_stop:
            print("error:",self.des_stop,self.org_stop)
        self.wait_time = 0
        self.travel_time = 0
        self.label = label#0--初始化的乘客， 1-正常随机到达的乘客
        self.spawn_time = curtime
    def onboarding(self,curtime):
        self.wait_time = curtime - self.spawn_time
    def alighting(self,curtime):
        self.travel_time = curtime - self.spawn_time -self.wait_time

class CLS_intersec(object):
    def __init__(self,loc,phase,origin_time,p1_time,p2_time):
        self.loc = loc 
        self.phase = phase
        self.pass_t = origin_time
        self.p1_time = p1_time
        self.p2_time = p2_time
        
    def proceed(self,):
        self.pass_t += 1
        if self.phase == 0:
            all_time = self.p1_time
        else:
            all_time = self.p2_time
        if self.pass_t > all_time:
            self.phase = 1- self.phase
            self.pass_t = 0
            

                    
