#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 14:26:07 2020

@author: mendy
"""

import numpy as np
from Bus_plus_Stop import CLS_bus,CLS_bus_stop
from Line import Line
import datetime
import scipy.stats as stats
import copy
import time


class Env():
    def __init__(self,envConfig=None):
        self.bus_stop_num = envConfig.Stop_Num
        self.hold_strategy = envConfig.Hold_Strategy 
        self.warm_up_time = envConfig.warm_up_time
        #--------乘客过饱和------
        self.pax_saturate = envConfig.pax_saturate
        #-------travel time 随机-------
        self.ran_travel = envConfig.ran_travel
        # self.emit_time = headway ## 每8分钟发一趟车
        self.headway = envConfig.headway 
        self.bus_omega = envConfig.bus_omega
        self.dir_num = 2
        self.lines = []
        self.bus_num_each_dir = envConfig.bus_num_each_dir
        self.bus_num = self.bus_num_each_dir * self.dir_num
        self.stop_loc_list = envConfig.Stop_Loc_List
        self.road_length = envConfig.road_length
        # for datacore
        # self.envConfig = envConfig
        self.bus_hold_action_List = []

        # Initialize line and bus
        for i in range(self.dir_num):
            if i == 0:
                dispatch_loc = self.road_length- self.road_length/(2*self.bus_num_each_dir)
            else:
                dispatch_loc =  self.road_length/(2*self.bus_num_each_dir)     
            l = Line(i,envConfig=envConfig)#初始化两条线路的下一个发车时间点
            self.lines.append(l)
            #初始化12辆车
            for j in range(self.bus_num_each_dir):
                id = j+i*self.bus_num_each_dir
                # print(id,dispatch_loc)
                b = CLS_bus(w=self.bus_omega, dispatch_loc=dispatch_loc, stop_nums=self.bus_stop_num, stop_loc_list=self.stop_loc_list[i], node_loc_list= envConfig.Node_Loc_List[i], \
                            id = id, emit_time=0, hold_strategy=self.hold_strategy, dir=i, alight_rate=self.lines[i].alight_rate, board_rate=self.lines[i].board_rate)
                self.lines[i].bus_list.append(b)
                if i == 0:
                    dispatch_loc= max(0.0,dispatch_loc-self.road_length/self.bus_num_each_dir)
                else:
                    dispatch_loc= min(dispatch_loc+self.road_length/self.bus_num_each_dir,self.road_length)
                    
        self.trajectory = [[] for i in range(self.dir_num)]
        for i in range(self.dir_num):
            for j in range(self.bus_num):
                self.trajectory[i].append([])
        self.memory = []
        self.hold_record = [[] for i in range(self.bus_num)]
    
    def save_traj(self,step_t):
        for i in range(self.dir_num):
            for b in self.lines[i].bus_list:
                self.trajectory[i][b.id].append([b.emit_time,b.trajectory,b.hold_action,b.hold_loc])
                self.memory.append(b.state)
                self.hold_record[b.id].extend(b.hold_record)
    
    def cal_ave_time(self,):
        wait_time,travel_time,all_time= [],[],[]
        for i in range(self.dir_num):
            wait_time.extend(self.lines[i].wtime_list)
            travel_time.extend(self.lines[i].ttime_list)
            all_time.extend(self.lines[i].alltime_list)
        ave_wait = np.mean(wait_time)
        std_wait = np.std(wait_time)
        
        ave_travel = np.mean(travel_time)
        std_travel = np.std(travel_time)
        
        ave_all = np.mean(all_time)
        std_all = np.std(all_time)
        return ave_wait,std_wait,ave_travel,std_travel,ave_all,std_all

    def dispatch_bus(self,step_t):
        """
        發車
        1. 判斷當前是否有到終點站的車輛
        2. 到達終點站的車輛重新初始化爲另一方向位於起始站的車輛
        Parameters
        ----------
        step_t : int
            當前時刻
        Returns
        -------
        None.

        """
        for dir in range(self.dir_num):
            stop_locs = self.stop_loc_list[dir]
            for b in self.lines[dir].bus_list:
                if (dir == 0 and b.loc+b.w/2 >= stop_locs[-1]) or (dir == 1 and b.loc-b.w/2<=stop_locs[-1]):
                    b.terminal = True
                    #b.arrive_time = step_t
                    self.trajectory[dir][b.id].append([b.emit_time,b.trajectory,b.hold_action,b.hold_loc])
                    opposite_dir = (dir+1)%self.dir_num
                    b.restart(opposite_dir,self.bus_stop_num,step_t)
                    b.loc = stop_locs[-1]
                    self.lines[opposite_dir].bus_list.append(b)
                    # print("before,bus num:",len(self.lines[dir].bus_list))
                    self.lines[dir].bus_list.remove(b)
                    # print("after,bus num:",len(self.lines[dir].bus_list))
                    # print(f"bus{b.id} change dir from line{dir} to line{opposite_dir}")
    
    def get_busstate(self,dir,j):
        """
        獲得bus的狀態

        Parameters
        ----------
        dir : TYPE
            線路方向，可選 0，1
        j : TYPE
            line.bus_list中的所處index

        Returns
        -------
        s : TYPE
            [前車距離，前前車距離，...]，共bus_num個數
            dis = realdis/800
        """
        s = []
        current_id = j
        current_dir = dir
        for i in range(self.bus_num):
            if current_id == 0:
                if current_dir == 0:
                    # terminal_loc = np.pi*2
                    terminal_loc = self.stop_loc_list[0][-1]
                else:
                    terminal_loc = 0.0
                opposite_dir = (current_dir+1)%self.dir_num
                if len(self.lines[opposite_dir].bus_list) > 0 :
                    dis_for = abs(terminal_loc-self.lines[current_dir].bus_list[current_id].loc)+abs(terminal_loc-self.lines[opposite_dir].bus_list[-1].loc)
                    #
                    current_id = len(self.lines[opposite_dir].bus_list)-1
                    current_dir = opposite_dir
                else:
                    # dis_for = np.pi*4-abs(self.lines[current_dir].bus_list[current_id].loc-self.lines[current_dir].bus_list[-1].loc)
                    dis_for = 2*self.stop_loc_list[0][-1]-abs(self.lines[current_dir].bus_list[current_id].loc-self.lines[current_dir].bus_list[-1].loc)
                    #
                    current_id = len(self.lines[current_dir].bus_list)-1
            else:
                for_ind = current_id-1
                dis_for = abs(self.lines[current_dir].bus_list[current_id].loc - self.lines[current_dir].bus_list[for_ind].loc)

                #
                current_id = for_ind

            s.append(dis_for/800.)
        return s

    def update(self,step_t):
        """
        1. 发车
        2. 更新等车乘客数
        3. 当前已在站点的车辆进行上下客

        Returns
        -------
        state: list[list[int]], state[bus_id] = bus_state  -> [前車距離，前前車距離，...]，共bus_num個數 when need control; [] when not need control

        """
        
        state = [[] for _ in range(self.bus_num)]
        ###------1. 发车---------(如果始發站存在還init_bus的車子，把這些車子隔8min發出去一趟，或者從終點站發一趟)
        self.dispatch_bus(step_t)
        
        ###------2. 乘客到站------(车站是否第一趟车经过了，如果是，则按照泊松分布随机生成等车乘客)
        
        for i in range(self.dir_num):
            self.lines[i].stop_gen_pax(step_t)
       
        
        
        ###------3. 路口信號燈更新----
        for i in range(self.dir_num):
        	self.lines[i].signal_update()
        
        
        ###------4. 判斷車輛是否在站點/是否在路口處-----
        for i in range(self.dir_num):
        	self.lines[i].locate_sec_stop()
        
        
        
        ###------4. 当前已在站点的车辆进行上下客--------
        for i in range(self.dir_num):
            self.lines[i].buses_alight_board()
        
        
        
        ###------4.判断车辆是否应该移动-----------------
        for i in range(self.dir_num):
            self.lines[i].move_judge(step_t)
        
            
        
        ###------5. 爲離開車站的車輛生成新路段的速度----
        if self.ran_travel == True:
            for i in range(self.dir_num):
                self.lines[i].Gen_speed()
        
        
        # start_time = time.time()
        ###-----6. 如果策略爲RL，判斷車輛是否需要控制，需要則返回車輛當前狀態-------
        if self.hold_strategy == 4:
            for i in range(self.dir_num):
                for j in range(len(self.lines[i].bus_list)):
                    self.lines[i].bus_list[j].wait_control_jdg()
                    if self.lines[i].bus_list[j].need_RLcontrol == True:
                        state[self.lines[i].bus_list[j].id]=self.get_busstate(i,j)
        # end_time = time.time()
        
        #-------7. 更新車輛位置-----
        for line in self.lines:
            for j in range(len(line.bus_list)):
                line.bus_list[j].update(line.stop_loc_list,line.intersec_loc_list,step_t,self.warm_up_time,step_t)
        # print("cost time:",end_time-start_time)
        
        # ### ------5. 到達終點站的車輛重新初始化爲另一方向位於起始站的車輛-------
        # self. terminal_bus_restart_from_other_direction(step_t)
        # ###------6. 删除到达终点站的车辆并记录车辆的轨迹---------------
        # self.del_not_operate_save_traj(step_t)
        # ###------5.判断车辆是否应该触发holding decision,需要控制的车辆记录当前状态------
        # for i in range(self.dir_num):
        #     for j in range(len(self.lines[i].bus_list)):
        #         self.lines[i].bus_list[j].control_or_not()
        
        # print("cost time:",end_time-start_time)
        
        return state

    def control(self, action, max_holding=180,deltaflag=0, step_t=0):
        action_real = b = [action[i] for i in range(len(action))] # copy action
        for i in range(self.dir_num):
            # print("dir:",i)
            for j in range(len(self.lines[i].bus_list)):
                b = self.lines[i].bus_list[j]
                # print("bus id, loc:",b.id,b.loc)
                if action[b.id] != -2:
                    # print("驻留:",b.id,a[b.id])
                    # print(b.catch,b.is_close,b.operate,b.hold_time,b.move_flag,b.at_stop,b.Arrive,b.need_RLcontrol)
                    if deltaflag:
                        s_ind = b.nextstop_ind
                        ht = self.lines[i].cal_holdingtime(step_t, s_ind, j, strategy=3)
                        # print(f"bus{b.id} on line{i} hold time={ht}, t={step_t}, busstopid={s_ind}")
                        b.hold_time += action[b.id] * max_holding
                        b.hold_time = min(max_holding,max(0,b.hold_time))
                    else:
                        b.hold_time = action[b.id] * max_holding
                    action_real[b.id] = b.hold_time / max_holding
                    b.hold_action.append(b.hold_time)
                    b.hold_loc.append(b.loc)
                    self.bus_hold_action_List.append(b.hold_time)
                    # print(b.first_dec)
                    # print(b.hold_loc)
        # print("end control")
        return action_real

    def collect_data(self):
        # print("start data collection:")
        MinHeadway = -1
        PaxNumList = [0]*self.bus_num  # pax number. eac element is a list that contains #of pax on wach bus: self[i_ep][bus_id]=PaxNum on bus(bus_id) at at time t
        TotalPaxNum = 0  # total pax number at time t
        AvgPaxWaitingTime = 0  # the average of all pax waiting time at time t
        AvgBusHoldingTime = 0  # the average bus holding time at time t
        for i in range(self.dir_num):
            if MinHeadway!=-1:
                break
            for j in range(len(self.lines[i].bus_list)):
                headways = self.get_busstate(i, j)
                MinHeadway = min(headways) * 800 #800: scaling, see the get_busstate function
                # print(f"minheadways={MinHeadway}")
                break

        wtime_list = []
        for line in self.lines:
            for bus in line.bus_list:
                PaxNumList[bus.id] = len(bus.pax_list)
                TotalPaxNum += len(bus.pax_list)
            for bus_stop in line.stop_list:
                TotalPaxNum += len(bus_stop.pax_list)
            wtime_list.extend(line.wtime_list[-20:])
        if(len(wtime_list)>0):
            AvgPaxWaitingTime = sum(wtime_list) / len(wtime_list)
        if (len(self.bus_hold_action_List) > 0):
            AvgBusHoldingTime = sum(self.bus_hold_action_List[max(-20,-len(self.bus_hold_action_List)):])/min(20,len(self.bus_hold_action_List))
        # print(f"MinHeadway={MinHeadway}\nPaxNumList={PaxNumList}\n"
        #       f"TotalPaxNum={TotalPaxNum}\nAvgPaxWaitingTime={AvgPaxWaitingTime}\n"
        #       f"wtime_list={wtime_list}\nAvgBusHoldingTime={AvgBusHoldingTime}")
        debug_cnt = 0
        for i in range(self.dir_num):
            for j in range(len(self.lines[i].bus_list)):
                BusHeadwayVariance = np.var(np.array(self.get_busstate(i, j)))
                # print(f"var={BusHeadwayVariance}, cnt={debug_cnt}")
                debug_cnt+=1

        return MinHeadway,PaxNumList,TotalPaxNum,AvgPaxWaitingTime,AvgBusHoldingTime,BusHeadwayVariance
    def collect_final_data(self):
        wt,tt,at = [],[],[]
        for line in self.lines:
            wt.extend(line.wtime_list)
            tt.extend(line.ttime_list)
            at.extend(line.alltime_list)

        hold_action = self.bus_hold_action_List
        return wt,tt,at,hold_action
                    
                
                
                
        
                
        
            
            
            






