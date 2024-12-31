import jax
from jax import numpy as jnp
import dataclasses
# import torch
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import agents
from waymax.config import DatasetConfig,DataFormat
import os
import torch
print("torch on gpu is available? {}".format(torch.cuda.is_available()))
def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        raise Exception("Jax does not have GPU support")
print("jax on gpu is available? {}".format(jax_has_gpu()))
# Config dataset:
max_num_objects = 12
conf = DatasetConfig(
    path=os.environ['WOMD_VAL_PATH'],
    max_num_rg_points=20000,
    data_format=DataFormat.TFRECORD,
    max_num_objects=max_num_objects,
    num_shards = 12,
    repeat= 1,
    batch_dims=(1,),
    distributed=True
)
'''
batch_dims – List of size of batch dimensions. 
Multiple batch dimension can be used to provide inputs for multiple devices. 
E.g. [jax.local_device_count(), batch_size_per_device].
'''
data_iter = dataloader.simulator_state_generator(config=conf)
dynamics_model = dynamics.DeltaLocal()
# Expect users to control all valid object in the scene.
obj_idx = jnp.arange(max_num_objects)
npc_actor = agents.create_expert_actor(
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: ~state.object_metadata.is_sdc,
)
# sdc using constant speed
# actor = agents.create_constant_speed_actor(
#     speed=0.0,
#     dynamics_model=dynamics_model,
#     is_controlled_func = lambda state: state.object_metadata.is_sdc,
# )
# sdc using IDM
# actor = agents.IDMRoutePolicy(
#   is_controlled_func=lambda state: state.object_metadata.is_sdc,
# )
# sdc using expert
actor = agents.create_expert_actor(
    dynamics_model=dynamics_model,
    is_controlled_func = lambda state: state.object_metadata.is_sdc,
    )
env = _env.MultiAgentEnvironment(
    dynamics_model=dynamics_model,
    config=dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=max_num_objects,
        controlled_object=_config.ObjectType.VALID,
        compute_reward = True    ),)
actors = [actor]
jit_step = jax.jit(env.step)
jit_reward = jax.jit(env.reward)
jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]
total_reward = 0.
@jax.pmap
def simulate(scenario: datatypes.SimulatorState):
    outputs = [
        jit_select_action({}, scenario, None, None)
        for jit_select_action in jit_select_action_list
    ]
    action = agents.merge_actions(outputs)
    reward = jit_reward(scenario, action)
    rewards,rew =datatypes.select_by_onehot(
        reward, scenario.object_metadata.is_sdc, keepdims=False
    )         
    next_state = jit_step(scenario, action)
    return rew,next_state
@jax.pmap
def fun1(scenario: datatypes.SimulatorState):
    states = [env.reset(scenario)]
    return states

for scen_idx,scenario in enumerate(data_iter):
    states = fun1(scenario)
    # log_states = fun1(scenario)
    for _ in range(states[0].remaining_timesteps[0]):
        current_state = states[-1]
        reward_sdc,next_state = simulate(current_state)
        total_reward += reward_sdc.mean()
        states.append(next_state)
    print('simulated {}th scenario'.format(scen_idx))
    if scen_idx == 5:
        print('Jax works fine on simulation')
        break

    
