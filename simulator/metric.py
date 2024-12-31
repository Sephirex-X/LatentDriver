import numpy as np 
from prettytable import PrettyTable
class MetricBaseValue():
    def __init__(self,arrival_thres):
        self.PR = []
        self.OR = []
        self.CR = []
        self.arrival_thres = arrival_thres
        for thre in arrival_thres:
            setattr(self,'AR'+str(thre),[])
        self.R = []
    def update(self, info, i):
        self.PR.append(info['metric/progress_rate'].flatten()[i])
        self.OR.append(info['metric/offroad_rate'].flatten()[i])
        self.CR.append(info['metric/collision_rate'].flatten()[i])
        self.R.append(info['reward/reward_mean'].flatten()[i])
        for thre in self.arrival_thres:
            temp = getattr(self,'AR'+str(thre))
            temp.append(info['metric/arrival_rate'+str(int(thre*100))].flatten()[i])
            setattr(self,'AR'+str(thre),temp)
    def mean(self):
        mAR = []
        metric ={'number of episodes': len(self.PR),
                'metric/offroad_rate': np.array(self.OR).mean(),
                'metric/collision_rate': np.array(self.CR).mean(),
                'metric/progress_rate': np.array(self.PR).mean(),
                'reward/reward_mean': np.array(self.R).mean()}
        for thre in self.arrival_thres:
            value = np.array(getattr(self,'AR'+str(thre))).mean()
            metric.update({'metric/arrival_rate'+str(int(thre*100)):value})
            mAR.append(value)
        metric.update({'metric/AR[{}:{}]'.format(int(min(self.arrival_thres)*100),int(max(self.arrival_thres)*100)): np.array(mAR).mean()})
        return metric
    def get_info(self):
        return self.mean()
class Metric():
    def __init__(self,
                 arrival_thres,
                 intention_label_path,
                 batch_dims
                 ):
        self.intention_label_path = intention_label_path
        self.batch_dims = batch_dims
        self.reset_times = 0
        self.arrival_thres = arrival_thres
        
        self.global_ = MetricBaseValue(arrival_thres)
        self.label_class = ['stationary_','straight_','turning_left','turning_right','U-turn_left']
        self.info_by_class_global = {}
        for k in self.label_class:
            self.info_by_class_global.update({k:MetricBaseValue(arrival_thres)})
            
    def reset(self, intention_list=None):
        self.waymo_offroad_num_t = np.zeros(self.batch_dims)
        self.waymo_overlap_num_t = np.zeros(self.batch_dims)
        self.waymo_arrivated_num_t = [np.zeros(self.batch_dims) for _ in range(len(self.arrival_thres))]
        self.overlap_after_arrival = np.zeros(self.batch_dims)
        self.offroad_after_arrival = np.zeros(self.batch_dims)
        self.sdc_kinematic_infeasibility_t = np.zeros(self.batch_dims)
        self.sdc_wrongway_t = np.zeros(self.batch_dims)
        self.reward_all = np.zeros(self.batch_dims)
        self.reset_times += 1
        
        self.intention_list = intention_list

        

    def update(self, rewards:dict, rew):
        self.sdc_wrongway_t += rewards['sdc_wrongway']
        self.progress_reward = rewards['sdc_progression']
        self.ADE = rewards['log_divergence']
        self.sdc_kinematic_infeasibility_t += rewards['sdc_kinematic_infeasibility']
        for thres in self.arrival_thres:
            self.waymo_arrivated_num_t[self.arrival_thres.index(thres)] += np.where(rewards['sdc_progression'] > thres, 1, 0)
        self.waymo_offroad_num_t += np.where(rewards['offroad'] > 0, 1, 0)
        self.waymo_overlap_num_t += np.where(rewards['overlap'] > 0, 1, 0)

        self.reward_all += rew
    def collect_batch(self, info):
        info['metric/progress_rate'] = self.progress_reward
        info['metric/offroad_rate'] = np.where(self.waymo_offroad_num_t > 0, 1, 0)
        info['metric/collision_rate'] = np.where(self.waymo_overlap_num_t > 0, 1, 0)
        for theres in self.arrival_thres:
            index = self.arrival_thres.index(theres)
            mask = np.logical_and.reduce((self.waymo_arrivated_num_t[index] > 0, self.waymo_offroad_num_t == 0, self.waymo_overlap_num_t == 0))
            info['metric/arrival_rate'+str(int(theres*100))] = np.where(mask, 1, 0)
        info['reward/reward_mean'] = self.reward_all
        # for vis hack here
        self.info_hack = info.copy()
        # self.update_by_class(info)
        for i in range(len(self.intention_list)):
            self.global_.update(info, i)
            intention = self.intention_list[i]
            self.info_by_class_global[intention].update(info, i)
            
        # this api is for tensorboard
        for k,v in info.items():
            info[k] = v.mean()
    def get_global_info(self):
        global_info = self.global_.get_info()
        table = PrettyTable()
        table.add_column('metric',[k.split('/')[-1] for k in global_info.keys()])
        # calculate average over class here
        avg_over_class = []
        for k, v in self.info_by_class_global.items():
            avg_over_class.append([round(np.nan_to_num(vv),4) for kk, vv in v.get_info().items()])
        avg_over_class = np.array(avg_over_class).T
        non_zero_type_num = len(avg_over_class[0].nonzero()[0])
        avg_over_class_sum = np.sum(avg_over_class, axis=1)
        # cal average for metrics
        divider = np.ones_like(avg_over_class_sum) * non_zero_type_num
        # putting num of eposide as the same
        divider[0] = 1
        avg_over_class = np.round(avg_over_class_sum/divider,4)
        avg_over_class = avg_over_class.tolist()
        avg_over_class[0] = int(avg_over_class[0])   
        table.add_column('Average_over_class', avg_over_class) 
        table.add_column('Average',[round(v,4) for k, v in global_info.items()])
        for k, v in self.info_by_class_global.items():
            table.add_column(k, [str(round(vv,4)) for kk, vv in v.get_info().items()])
        return table

