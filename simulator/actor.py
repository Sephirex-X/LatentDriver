from typing import Callable

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.agents import actor_core

def create_control_actor(
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array]
) -> actor_core.WaymaxActorCore:
  """Creates an actor controlling its steer and acc.

  Args:
    dynamics_model: The dynamics model the actor is using that defines the
      action output by the actor.
    is_controlled_func: Defines which objects are controlled by this actor.
    speed: Speed of the actor, if None, speed from previous step is used.

  Returns:
    An statelss actor that drives the controlled objects with constant speed.
  """

  def select_action(  # pytype: disable=annotation-type-mismatch
      params: actor_core.Params,
      state: datatypes.SimulatorState,
      actor_state=None,
      rng: jax.Array = None,
  ) -> actor_core.WaymaxActorOutput:
    """Computes the actions using the given dynamics model and speed."""
    del actor_state, rng  # unused.
    is_controlled = is_controlled_func(state)
    shape = state.shape + (state.num_objects,1)
    actions = datatypes.Action
    wrapped_action = [wrap_action(action,state) for action in params['actions']]
    wrapped_data = jnp.concatenate(wrapped_action,axis=-1)
    actions = datatypes.Action(data=wrapped_data, 
                               valid=jnp.ones(shape, dtype=jnp.bool_))

    # Note here actions' valid could be different from is_controlled, it happens
    # when that object does not have valid trajectory from the previous
    # timestep.
    return actor_core.WaymaxActorOutput(
        actor_state=None,
        action=actions,
        is_controlled=is_controlled,
    )

  return actor_core.actor_core_factory(
      init=lambda rng, init_state: None,
      select_action=select_action,
      name=f'muzero_agent',
  )
def wrap_action(data,state):
    '''
        input data is [bs,]
        wraping to [1,bs,num_objects,1]
    '''
    return (jnp.ones(state.shape) * data)[...,jnp.newaxis,jnp.newaxis].repeat(state.num_objects,-2)