
import numpy as np
import os
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
from waymax import dynamics
from simulator.waymo_base import WaymoBaseEnv
from simulator.metric import Metric     
from simulator.utils import combin_traj,build_discretizer,get_cache_polylines_baseline
from simulator.observation import get_obs_from_routeandmap_saved,preprocess_data_dist_jnp
def get_obs(*args):
    data_dict,sdc_obs = get_obs_from_routeandmap_saved(*args)
    obs = preprocess_data_dist_jnp(data_dict)
    return obs,data_dict

class WaymoEnv():
    def __init__(self,
                 waymax_conf,
                 data_conf,
                 env_conf,
                 batch_dims,
                 ego_control_setting:dict,
                 metric_conf:dict,):
        '''
        ego_control_setting:
            ego_policy_type:
                'expert'
                'idm': not implemented
                'custom'
            action_type: bicycle or waypoints
            action_space = dict(
                type = 'continuous', # can be 'continuous' or 'discrete',
                action_ranges = action_ranges,
                bins = [13,39] #for acc and steer only for discrete
            )
            npc_policy_type:
                'expert'
                'idm'
        metric_conf:
            arrival_thres = 0.9
            intention_label_path =/root/intention_label_val
        
        '''
        self.ego_policy_type = ego_control_setting.ego_policy_type
        action_type = ego_control_setting.action_type
        action_space = ego_control_setting.action_space
        npc_type = ego_control_setting.npc_policy_type
        self.data_conf = data_conf
        if action_type == 'bicycle':
            env_dynamic_model = dynamics.InvertibleBicycleModel()
        elif action_type == 'waypoint':
            env_dynamic_model = dynamics.DeltaLocal()
        
        if action_space.type == 'continuous':
            pass
        elif action_space.type == 'discrete':
            raise Warning('discrete action space has not been verified!')
            self.discretizer = build_discretizer(action_space)
        self.action_space_type = action_space.type
        self.com_traj = jax.pmap(combin_traj)
        self.dynamic_inverse = jax.pmap(env_dynamic_model.inverse)
        # waymax_conf.update({'batch_dims': batch_dims})
        self.env = WaymoBaseEnv(
                    waymax_conf=waymax_conf,
                    env_conf=env_conf,
                    action_space=action_space,
                    action_type=action_type,
                    dynamics_model=env_dynamic_model,
                    npc = npc_type)
        
        self.metric = Metric(**metric_conf, batch_dims=batch_dims)
        self.log_rew_dict = {}
        self.batch_dims = batch_dims
        
        self.path_to_map = os.path.join(data_conf.path_to_processed_map_route,'map')
        self.path_to_route = os.path.join(data_conf.path_to_processed_map_route,'route')
        
    def get_expert_action(self)->np.ndarray:
        current_state = self.states[-1]
        traj = self.com_traj(current_state)
        action = self.dynamic_inverse(traj, current_state.object_metadata,jnp.zeros(self.batch_dims[0],dtype=jnp.int32))
        action = np.array(action.data[current_state.object_metadata.is_sdc])
        return action
    
    def reset(self):
        self.scenario = next(self.env.data_iter)
        self.states = self.env.pmap_reset(self.scenario)
        cur_state = self.states[-1]
        self.road_np, self.route_np, self.intention_label = get_cache_polylines_baseline(cur_state, self.path_to_map, self.path_to_route, self.metric.intention_label_path)
        self.metric.reset(self.intention_label)
        obs, obs_dict = get_obs(cur_state,self.road_np,self.route_np)
        return obs, obs_dict

    def step(self,action=None, show_global=False):
        # check ego agent control mode
        if self.ego_policy_type == 'custom':
            pass
        elif self.ego_policy_type == 'stationary':
            action = np.zeros_like(action)
        elif self.ego_policy_type == 'expert':
            action = self.get_expert_action()
        else:
            raise ValueError(f'ego_policy_type {self.ego_policy_type} not supported, only support expert, stationary and custom')
        
        current_state = self.states[-1]
        info = {}
        if self.action_space_type =='discrete':
            action = self.discretizer.make_continuous(action)
            # print(action)
        # (N,B,action_space)
        action = action.reshape(self.batch_dims[0],self.batch_dims[1], -1)
        # (N,action_space,B)
        actions = np.transpose(action,(0,2,1))
        
        
        rewards,rew,next_state = self.env.pmap_sim(actions,current_state)
        obs, obs_dict = get_obs(next_state,self.road_np,self.route_np)
        done = np.array(list(next_state.is_done) * self.batch_dims[-1]).flatten()
        self.states.append(next_state)
        self.metric.update(rewards,rew)
        # logger
        for k,_ in rewards.items():
            if k not in self.log_rew_dict:
                self.log_rew_dict[k] = rewards[k]
            else:
                self.log_rew_dict[k] += rewards[k]      
                  
        if done[-1]:
            # logger
            for k,v in self.log_rew_dict.items():
                info['reward/'+k] = v.mean()
            # metric
            self.metric.collect_batch(info)
            if show_global:
                print('\n',self.metric.get_global_info())
        return (obs, obs_dict,np.array(rew).reshape(self.num_envs,), done, info)
            
        
    @property
    def observation_space(self) -> gym.spaces.Space:
        raise NotImplementedError
        # return gym.spaces.Box(low= -float('inf'), high=float('inf'), shape=(self.cfg.obs_shape), dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space_
    
    @property
    def reward_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(1, ), dtype=np.float32
            )
    @property
    def num_envs(self)->int:
        return self.batch_dims[0] * self.batch_dims[1]
    
    def get_env_idx(self,batch_id=None):
        id_bank = np.array(self.states[-1]._scenario_id.reshape(-1), dtype=np.uint64).tolist()
        if batch_id == None:
            return id_bank
        else:
            return id_bank[batch_id]