#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import math
from Bus_plus_Stop import CLS_bus,CLS_bus_stop,CLS_intersec
import time
class Line():
    def __init__(self,dir,envConfig=None):
        self.dir = dir
        self.node_loc = envConfig.Node_Loc_List[dir]
        self.links = len(self.node_loc)
        self.stop_nums = envConfig.Stop_Num
        self.stop_loc_list = envConfig.Stop_Loc_List[dir] #onlyonedirection
        self.hold_strategy = envConfig.Hold_Strategy
        self.pax_saturate = envConfig.pax_saturate
        self.warm_up_time = envConfig.warm_up_time
        self.road_length = envConfig.road_length

        self.init_busid = [] # 初始化站點的所有bus id
        
        self.bus_list = []
        self.stop_list = []
        #初始化各个站点等车乘客数
        wait_nums = [5 for _ in range(self.stop_nums)]
        # wait_nums[0] = 5
        wait_nums[-1]=0
        self.terminal_busid_time = []
        self.start_stop_busid = []
        self.headway = envConfig.headway
        self.alight_rate = envConfig.alight_rate
        self.board_rate = envConfig.board_rate
        ##计算乘客平均等车时间------
        self.wtime_list = []
        self.ttime_list = []
        self.alltime_list = []
        #设置各个策略的参数
        ####hold strategy为1，schedule based------
        mu_t = 4
        sigma_t=0.25 # 暫時隨便取個數
        slack_time=4*sigma_t*60
        self.depart_times = [0 for _ in range(self.stop_nums)]
        ####hold strategy为2，forward headway based------
        self.alpha = 0.2
        hd_var = 0.95*sigma_t/(np.sqrt(self.alpha*(1-self.alpha)))
        self.FH_ds = [0 for _ in range(self.stop_nums)]# headway_based的方法的slack time, forward_based和two-way_based共用

        arr_rates = envConfig.Arr_Rates[self.dir]
        for i in range(self.stop_nums): #FIXME: stop_num*2 因为有逆序了一遍 NOOO! a Line is single directed
            ave_cost_time = [arr_rates[j]*self.headway/self.board_rate if j==0 else \
                             arr_rates[j]*self.headway/self.board_rate+mu_t*60+slack_time for j in range(i+1)]
            depart_schedule = sum(ave_cost_time)
            bs = CLS_bus_stop(loc=self.stop_loc_list[i], i=i, wait_num=wait_nums[i], arr_rate=arr_rates[i], \
                              depart_schedule=depart_schedule, stop_num=self.stop_nums)
            self.stop_list.append(bs)
            self.depart_times[i] = depart_schedule
            
            fh_ds = 3*(self.alpha+arr_rates[i]/self.board_rate)*hd_var*60
            self.FH_ds[i] = fh_ds

        self.intersec_loc_list = envConfig.Intersec_Loc_List[dir]
        self.intersec_list = []
        init_phase = envConfig.Init_Phase_List[self.dir]
        pass_time = envConfig.Pass_Time_List[self.dir]
        p1_time = envConfig.P1_Time_List[self.dir]
        p2_time = envConfig.P2_Time_List[self.dir]
        ##----傳入路口信息----
        for i in range(len(self.intersec_loc_list)):
            sec = CLS_intersec(loc=self.intersec_loc_list[i], phase=init_phase[i], origin_time=pass_time[i], p1_time=p1_time[i], p2_time=p2_time[i])
            self.intersec_list.append(sec)
        
        #----------parameter setting-------------
        # self.ran_travel = ran_travel
        # lower, upper = mu_t - 2 * sigma_t, mu_t + 2 * sigma_t  # 截断在[μ-2σ, μ+2σ]
        # self.timeDis = stats.truncnorm((lower - mu_t) / sigma_t, (upper - mu_t) / sigma_t, loc=mu_t, scale=sigma_t)
        self.ran_travel = envConfig.ran_travel
        lower, upper = envConfig.mu_v - 2 * envConfig.sigma_v, envConfig.mu_v + 2 * envConfig.sigma_v
        self.speedDis = stats.truncnorm((lower - envConfig.mu_v) / envConfig.sigma_v, (upper - envConfig.mu_v) / envConfig.sigma_v, loc=envConfig.mu_v, scale=envConfig.sigma_v)
            
    def stop_gen_pax(self,step_t):
        """
        站點更新

        Returns
        -------
        None.

        """
        
        for i in range(self.stop_nums-1): # "-1" is to ban the last stop generate pax
            self.stop_list[i].proceed(step_t)
            # for j in range(len(self.stop_list[i].pax_list)):
            #     self.stop_list[i].pax_list[j].wait_time += 1
            # def func(x):
            #  	x.wait_time += 1
            # map(func,self.stop_list[i].pax_list)
        # no need
        # for i in range(len(self.bus_list)):
        #     if self.bus_list[i].at_stop == True:
        #         self.stop_list[self.bus_list[i].nextstop_ind].bus_pass_flag = True
    def signal_update(self,):
        for sec in self.intersec_list:
            sec.proceed()
    def locate_sec_stop(self,):
        for b in self.bus_list:
            b.at_stop_jdg(self.stop_loc_list)
            b.at_intersec_jdg(self.intersec_loc_list)
            if b.at_intersec == True:
                b.phase = self.intersec_list[b.intersec_ind].phase
            else:
                b.phase = -1
    def buses_alight_board(self,):
        """
        当前已在站点的车辆进行上下客

        Returns
        -------
        None.

        """
        for b in self.bus_list:
            stop_id = b.nextstop_ind
            s = self.stop_list[stop_id]
            #b.at_stop_jdg(self.stop_loc_list)
            if b.at_stop == True and b.is_close == False:
                wait_time,travel_time,all_time = b.board_alight(self.pax_saturate,s.pax_list)
                self.wtime_list.extend(list(wait_time))
                self.ttime_list.extend(list(travel_time))
                self.alltime_list.extend(list(all_time))

    def checkCatch(self):
    	#得先到站下完客再考慮catch的事情，所以沒有問題
        if self.bus_list != []:
            self.bus_list[0].catch = False
        for j in range(1,len(self.bus_list)):
            if self.dir == 0:
                if self.bus_list[j].loc + self.bus_list[j].w >= self.bus_list[j-1].loc:
                    self.bus_list[j].catch = True
                else:
                    self.bus_list[j].catch = False
            else:
                if self.bus_list[j].loc - self.bus_list[j].w <= self.bus_list[j-1].loc:
                    self.bus_list[j].catch = True
                else:
                    self.bus_list[j].catch = False
    
    
    def cal_holdingtime(self,t,stop_id,b_index,strategy=None):
        """

        :param t:global time stamp
        :param stop_id: the stop bus is at
        :param b_index: bus id
        :param strategy:
        :return:
        """
        if strategy is None:
            strategy = self.hold_strategy
        #计算驻留时间
        if strategy == 2:
            h= t - self.stop_list[stop_id].latest_bus_pass_time
            self.bus_list[b_index].hold_time = self.FH_ds[stop_id]+\
            (self.alpha+ self.stop_list[stop_id].arr_rate/self.board_rate)*(self.headway-h)
#             print("前后车时距：",self.headway-h)
#             print("驻留时间：",self.bus_list[b_index].hold_time)
        elif strategy == 3:
            h_for = t - self.stop_list[stop_id].latest_bus_pass_time
            if b_index >= len(self.bus_list)-1:#如果没有后车
                h_back = self.headway
            else:
                j = len(self.bus_list[b_index].trajectory)-1 # 從最近更新的位置開始搜索
                search_val = self.bus_list[b_index+1].loc # 後車就是list裡的後面一輛車
                if self.dir == 0:
                    while j>=0:
                        if self.bus_list[b_index].trajectory[j]>=search_val:
                            j = j-1
                        else:
                            break
                    pass_time = self.bus_list[b_index].emit_time+j
                    h_back = t - pass_time
                else:
                    while j>=0:
                        if self.bus_list[b_index].trajectory[j]<=search_val:
                            j = j-1
                        else:
                            break
                    pass_time = self.bus_list[b_index].emit_time+j
                    h_back = t - pass_time
                # print("forward time:", h_for)
                # print("back time:",h_back)
            self.bus_list[b_index].hold_time = self.FH_ds[stop_id]+\
            (self.alpha+ self.stop_list[stop_id].arr_rate/self.board_rate)*(self.headway-h_for)-self.alpha*(self.headway-h_back)
        return self.bus_list[b_index].hold_time
    def move_judge(self,step_t):
        #####1.检查是否catch 2.对于策略2和3，新到站计算驻留时间 3.根据catch情况和hold时间决定是否移动 4.对于策略1新离站需要更新离站时间表 5. 新离站生成下一Link的新速度 6. 更新位置######
        self.checkCatch()
        #1. --計算駐留時間--
        #schedule-based方法只需時刻表，策略FH和TH需要計算,RL方法默認等待決策,但也计算传统方法。
        # RL 为了deltatime所以要更新latest_bus_pass_time
        if self.hold_strategy == 4:
            for j in range(len(self.bus_list)):
                # self.bus_list[j].new_arrive()
                if self.bus_list[j].first_arrive == True: #when the door close for the first time
                    s_ind = self.bus_list[j].nextstop_ind
                    self.stop_list[s_ind].latest_bus_pass_time = step_t
                    self.bus_list[j].first_arrive = 0
                    # print(f"set stop{s_ind} latesttime={step_t}")
        if self.hold_strategy == 2 or self.hold_strategy == 3:
            for j in range(len(self.bus_list)):
                self.bus_list[j].new_arrive()
                if self.bus_list[j].Arrive == True:
                    s_ind = self.bus_list[j].nextstop_ind
                    if self.stop_list[s_ind].latest_bus_pass_time == 0:
                        self.bus_list[j].hold_time = 0
                    else:
                        self.cal_holdingtime(step_t,s_ind,j)
                    self.stop_list[s_ind].latest_bus_pass_time = step_t
        #2.--判斷是否移動--
        for b in self.bus_list:
            b.move_judge(self.depart_times,step_t)
        #當策略爲schedule-based時，車輛離開更新時刻表
        if self.hold_strategy == 1:
            for b in self.bus_list:
                if b.at_stop == True and b.move_flag == True:
                    self.depart_times[b.nextstop_ind] += self.headway
        

    def Gen_speed(self,):
        # start_time = time.time()
        bus_link = [[] for i in range(self.links)]#bus_link:每一条路上都有哪些车bus_link[i]表示到站点i这条路上的车
        need_Gen = False
        for b in self.bus_list:
            if (b.at_stop == True or b.at_intersec == True) and b.move_flag == True and b.nextnode_ind+1 <= self.links-1:
                need_Gen = True
        if need_Gen == True:
            for b in self.bus_list:
                if b.at_stop == True or b.at_intersec == True:
                    if b.move_flag == True and b.nextnode_ind+1 <= self.links-1:
                        bus_link[b.nextnode_ind+1].append([b.id,b.w])
                else:
                    bus_link[b.nextnode_ind].append([b.id,b.w])
            for b in self.bus_list: #同一路段上所有车速度保持相等
                if b.at_stop == True or b.at_intersec == True:
                    if b.move_flag == True and b.nextnode_ind+1 <= self.links-1:
                        if bus_link[b.nextnode_ind+1]!=[[b.id,b.w]]:
                            for _ in range(len(bus_link[b.nextnode_ind+1])):
                                if bus_link[b.nextnode_ind+1][_][0]!=b.id:
                                    b.w = bus_link[b.nextnode_ind+1][_][1] 
                    else:
                        
                        speed = self.speedDis.rvs(1)[0]
                        b.w = speed/3.6
        # end_time = time.time()
        # print("cost time:",end_time-start_time)