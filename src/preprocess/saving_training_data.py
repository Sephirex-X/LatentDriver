import src.utils.init_default_jax
import jax
import hydra
import os
from omegaconf import OmegaConf
from src.utils.utils import update_waymax_config, saving_data
import time
from simulator.waymo_env import WaymoEnv
import numpy as np
from waymax import dynamics
import jax.numpy as jnp

class TrainingDataCollector():
    def __init__(self,
                 config,
                 ):
        self.env = WaymoEnv(
            waymax_conf=config.waymax_conf,
            env_conf=config.env_conf,
            batch_dims=config.batch_dims,
            ego_control_setting=config.ego_control_setting,
            metric_conf=config.metric_conf,
            data_conf=config.data_conf,
        )
        self.save_path = config.save_path
        self.batch_dims = config.batch_dims
        self.size = self.batch_dims[0] * self.batch_dims[1]
        dynamics_model_bicycle = dynamics.InvertibleBicycleModel()
        dynamics_model_waypoints = dynamics.DeltaLocal()
        self.get_action_bicycle = jax.pmap(dynamics_model_bicycle.inverse)
        self.get_action_waypoints = jax.pmap(dynamics_model_waypoints.inverse)
    def run(self):
        self.idx = 0
        while True:
            try:
                obs, obs_dict = self.env.reset()
                obs = obs.reshape(self.env.num_envs,-1,7)
                obs_depth,obs_dim = obs.shape[1],obs.shape[-1]
                states = (obs).reshape(self.env.num_envs,-1,obs_depth,obs_dim)
                # format_data(obs_dict)
                actions_bicycle = np.zeros((self.env.num_envs, 1,2))
                actions_waypoints = np.zeros((self.env.num_envs, 1,3))
                rewards = np.zeros((self.env.num_envs, 1,1))
                done_ = False
                self.T = 1
                a = time.time()
                
                while not done_:
                    actions_bicycle = np.concatenate(
                        [
                            actions_bicycle,
                            np.zeros((self.env.num_envs, 2)).reshape(
                                self.env.num_envs, -1, 2
                            ),
                        ],
                        axis=1,
                    )
                    actions_waypoints = np.concatenate(
                        [
                            actions_waypoints,
                            np.zeros((self.env.num_envs, 3)).reshape(
                                self.env.num_envs, -1, 3
                            ),
                        ],
                        axis=1,
                    )
                    rewards = np.concatenate(
                        [
                            rewards,
                            np.zeros((self.env.num_envs, 1)).reshape(self.env.num_envs, -1, 1),
                        ],
                        axis=1,
                    )             
                    
                    data_ = {}
                    actions_to_collect = self.collect_actions(self.env.states[-1])
                    actions_bicycle[:,-1] = actions_to_collect['bicycle_actions']
                    actions_waypoints[:,-1] = actions_to_collect['waypoints_actions']
                    
                    obs, obs_dict,rew, done, info = self.env.step(self.env.get_expert_action(),show_global=False)
                    obs = obs.reshape(self.env.num_envs,-1,7)
                    state = (obs.reshape(self.env.num_envs,-1,obs_depth,obs_dim))
                    states = np.concatenate([states,state],axis=1)
                    rewards[:,-1] = rew.reshape(self.env.num_envs,1)
                    self.T+=1
                    done_ =done[-1]
                    
                for ii in range(self.env.num_envs):
                    scen_id = str(self.env.get_env_idx(ii))
                    sub_folder = os.path.join(self.save_path,'data')
                    os.makedirs(sub_folder,exist_ok=True)
                    terminals = np.zeros(81)
                    terminals[-1] = 1
                    traj = {
                        'obs': states[ii],
                        'waypoints_actions': actions_waypoints[ii],
                        'bicycle_actions': actions_bicycle[ii],
                        'rewards': rewards[ii],
                        'terminals':terminals,
                    }
                    saving_data(traj,name=os.path.join(sub_folder,scen_id))
                    with open(os.path.join(self.save_path,'name.txt'),'a') as f:
                        f.write(f'{scen_id}\n')


                                
                self.idx += 1
                print('Processed: ', self.idx, 'th batch, Time: ', time.time()-a, 's')

            except StopIteration:
                print("StopIteration")
                break

    def collect_actions(self,next_state):
        action_collected = {}
        traj = self.env.com_traj(next_state)
        '''for bicycles'''
        action = self.get_action_bicycle(traj,metadata=next_state.object_metadata, timestep=jnp.zeros(self.batch_dims[0],dtype=jnp.int32))
        action = np.array(action.data[next_state.object_metadata.is_sdc])
        
        # make acc steer here become (B,1)
        # acc,steer = action[...,0:1],action[...,1:2]
        action_collected.update(bicycle_actions = action)
        '''for waypoints'''
        action = self.get_action_waypoints(traj,metadata=next_state.object_metadata, timestep=jnp.zeros(self.batch_dims[0],dtype=jnp.int32))
        action = np.array(action.data[next_state.object_metadata.is_sdc])               
        # dx,dy,dyaw = action[...,0:1],action[...,1:2],action[...,2:3]
        action_collected.update(waypoints_actions = action)
        return action_collected

@hydra.main(version_base=None, config_path="../../configs", config_name="simulate")
def run(cfg):
    """
    Entry point for the data collection script.

    Args:
        cfg (OmegaConf): The configuration object.
    """
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = update_waymax_config(cfg)
    cfg = OmegaConf.merge(cfg, cfg.method)
    collector = TrainingDataCollector(cfg)
    collector.run()

if __name__ == '__main__':
    run()