import numpy as np
import jax.numpy as jnp
import jax
from waymax import datatypes
from src.utils.utils import loading_data
from src.utils.discretizer import Discretizer
import os

def get_cache_polylines_baseline(cur_state:datatypes.SimulatorState,
                                 path_to_map:str,
                                 path_to_route:str,
                                 intention_label_path:str=None):
    cur_id = cur_state._scenario_id.reshape(cur_state.shape)
    whole_map_by_device_id = []
    whole_route_by_device_id = []
    intention_label = []
    for device_id in range(cur_id.shape[0]):
        whole_map_by_batch = []
        whole_route_by_batch = [] 
        for batch_id in range(cur_id.shape[1]):
            whole_map_by_batch.append(loading_data(os.path.join(path_to_map,'{}.npy'.format(cur_id[device_id][batch_id])), mode='np'))
            whole_route_by_batch.append(loading_data(os.path.join(path_to_route,'{}.npy'.format(cur_id[device_id][batch_id])), mode = 'np'))
            if intention_label_path is not None:
                with open(os.path.join(intention_label_path,'{}.txt'.format(cur_id[device_id][batch_id])),'r')as f:
                    intention_label.append(f.readlines()[0])
                    
        whole_map_by_device_id.append(np.stack(whole_map_by_batch,axis=0))
        # print(whole_map_by_device_id[-1].shape)
        whole_route_by_device_id.append(np.stack(whole_route_by_batch,axis=0))
    road_np = np.stack(whole_map_by_device_id,axis=0)
    route_np = np.stack(whole_route_by_device_id,axis=0)
    return road_np,route_np,intention_label


def build_discretizer(action_space, seperate=False):
    if seperate:
        discretizer_list = []
        action_ranges = np.array(action_space.action_ranges)
        for i in range(len(action_ranges)):
            discretizer_range = action_ranges[i:i+1]
            discretizer = Discretizer(
                        min_value=discretizer_range[...,0], 
                        max_value=discretizer_range[...,1],
                        bins = np.array(action_space.bins[i:i+1],dtype=np.int32),
                        )        
            discretizer_list.append(discretizer)
        return discretizer_list
    else:
        discretizer_range = np.array(action_space.action_ranges)
        discretizer = Discretizer(
                    min_value=discretizer_range[...,0], 
                    max_value=discretizer_range[...,1],
                    bins = np.array(action_space.bins,dtype=np.int32),
                    )
        return discretizer

def combin_traj(
    simulator_state: datatypes.SimulatorState,
):
  """Infers an action from sim_traj[timestep] to log_traj[timestep + 1].

  Args:
    simulator_state: State of the simulator at the current timestep. Will use
      the `sim_trajectory` and `log_trajectory` fields to calculate an action.
    dynamics_model: Dynamics model whose `inverse` function will be used to
      infer the expert action given the logged states.

  Returns:
    Action that will take the agent from sim_traj[timestep] to
      log_traj[timestep + 1].
  """
  prev_sim_traj = datatypes.dynamic_slice(  # pytype: disable=wrong-arg-types  # jax-ndarray
      simulator_state.sim_trajectory, simulator_state.timestep, 1, axis=-1
  )
  next_logged_traj = datatypes.dynamic_slice(  # pytype: disable=wrong-arg-types  # jax-ndarray
      simulator_state.log_trajectory, simulator_state.timestep + 1, 1, axis=-1
  )
  combined_traj = jax.tree_map(
      lambda x, y: jnp.concatenate([x, y], axis=-1),
      prev_sim_traj,
      next_logged_traj,
  )
  return combined_traj