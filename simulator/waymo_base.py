
import numpy as np
import gymnasium as gym
import numpy as np
import waymax
import jax
import jax.numpy as jnp
from waymax import config as _config
from waymax import dynamics
from waymax import env as _env
from waymax import datatypes
from waymax import agents
from waymax import dataloader
from waymax.config import DatasetConfig
from simulator.actor import create_control_actor
from src.dataloader.waymax_scenloader import WaymoScenLoader
class WaymoBaseEnv():
    def __init__(self,
                 waymax_conf,
                 env_conf,
                 dynamics_model,
                 action_space,
                 action_type,
                 npc:str) -> None:
        is_customized = waymax_conf.pop('customized')
        if not is_customized:
            self.data_iter = dataloader.simulator_state_generator(config=DatasetConfig(**waymax_conf))
        elif is_customized:
            raise NotImplementedError("Customized dataset is not supported yet")
            self.data_iter = WaymoScenLoader(waymax_conf)
            
        if action_type == 'waypoint':
            # for dx dy dyaw
            shape = (3,)
            # ranges = [(-0.14, 6), (-0.35, 0.35), (-0.15,0.15)]
            ranges = action_space.action_ranges 
            low = np.array([r[0] for r in ranges])
            high = np.array([r[1] for r in ranges])
            self.action_space_ = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
        elif action_type == 'bicycle':
            # for acc steers
            shape = (2,)
            ranges = action_space.action_ranges #orignal -1
            low = np.array([r[0] for r in ranges])
            high = np.array([r[1] for r in ranges])
            self.action_space_ = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
        else:
            raise ValueError("action_type should be waypoint or bicycle")
        
        self.dynamics_model = dynamics_model
        # create actors
        actor = create_control_actor(is_controlled_func = lambda state: state.object_metadata.is_sdc)
        actors = [actor]
        if npc=='idm':
            controlled_object = _config.ObjectType.VALID
            npc = agents.IDMRoutePolicy(
            is_controlled_func=lambda state: ~state.object_metadata.is_sdc,
            # additional_lookahead_points = 40,
            # additional_lookahead_distance = 40.0,
            )
            actors.append(npc)
        elif npc=='expert':
            controlled_object = _config.ObjectType.SDC
        else:
            raise ValueError("npc should be idm or expert")
        env_conf.update({'controlled_object': controlled_object})
        self.select_action_list = [actor.select_action for actor in actors]
        
        self.env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics.StateDynamics(),
        config=_config.EnvironmentConfig(**env_conf)
        )          
        self.pmap_sim = jax.pmap(self.simulate)
        self.pmap_reset = jax.pmap(self.reset_func)
        
    def reset_func(self,scenario: datatypes.SimulatorState):
        states = [self.env.reset(scenario)]
        return states
    def simulate(self,actions:np.array,current_state: datatypes.SimulatorState):
        outputs = [
            select_action({'actions':actions}, current_state, None, None)
            for select_action in self.select_action_list
        ]
        traj = datatypes.dynamic_slice(
                    inputs=current_state.sim_trajectory, start_index=current_state.timestep.flatten()[0], slice_size=1, axis=-1
                )

        action_transformed = self.dynamics_model.compute_update(outputs[0].action, traj).as_action()
        outputs[0].action.data = action_transformed.data.astype(jnp.float32)
        # outputs[0].action.valid = action_transformed.valid  
        # jax.debug.print("sdc: {x} \n , anothers: {y}",x = outputs[0].action.valid, y=outputs[1].action.valid)
        action = waymax.agents.merge_actions(outputs)
        reward = self.env.reward(current_state, action)
        rewards,rew =datatypes.select_by_onehot(
            reward, current_state.object_metadata.is_sdc, keepdims=False
        )         
        next_state = self.env.step(current_state, action)
        return rewards,rew,next_state