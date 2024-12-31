# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics relating to route."""

import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric

# The value of the progress metric if the simulated agent is at the same
# point along the route as the expert agent was.
FULL_PROGRESS_VALUE = 1.0


class ProgressionMetric(abstract_metric.AbstractMetric):
  """Route progression metric for SDC.

  This metric returns a non-negative value representing the progression
  towards the final logged position from the initial logged position along the
  route. It first finds the closest on-route path to the SDC's current xy
  position according to Euclidean distance and uses that as the reference path
  to compute the progress (`arc_length`) relative to the logged SDC's initial
  and final xy positions.

  If the SDC trajectory is invalid or there are no valid on-route paths, it
  returns zero.
  """

  @jax.named_scope('ProgressionMetric.compute')
  def compute(
      self, simulator_state: datatypes.SimulatorState
  ) -> abstract_metric.MetricResult:
    """Computes the progression metric.

    Args:
      simulator_state: The current simulator state of shape.

    Returns:
      A (...) MetricResult containing the metric result described above.

    Raises:
      ValueError: If `simulator_state.sdc_paths` is undefined.
    """
    # Shape: (..., num_paths, num_points_per_path)
    # sdc_paths = simulator_state.sdc_paths
    # if sdc_paths is None:
    #   raise ValueError(
    #       'SimulatorState.sdc_paths required to compute the route progression '
    #       'metric.'
    #   )

    # Shape: (..., num_objects, num_timesteps=1, 2)
    obj_xy_curr = datatypes.dynamic_slice(
        simulator_state.sim_trajectory.xy,
        simulator_state.timestep,
        1,
        axis=-2,
    )

    # Shape: (..., 2)
    sdc_xy_curr = datatypes.select_by_onehot(
        obj_xy_curr[..., 0, :],
        simulator_state.object_metadata.is_sdc,
        keepdims=False,
    )
    sdc_xy_start = datatypes.select_by_onehot(
        simulator_state.log_trajectory.xy[..., 0, :],
        simulator_state.object_metadata.is_sdc,
        keepdims=False,
    )
    sdc_xy_end = datatypes.select_by_onehot(
        simulator_state.log_trajectory.xy[..., -1, :],
        simulator_state.object_metadata.is_sdc,
        keepdims=False,
    )

    # Shape: (..., num_objects, num_timesteps=1)
    obj_valid_curr = datatypes.dynamic_slice(
        simulator_state.sim_trajectory.valid,
        simulator_state.timestep,
        1,
        axis=-1,
    )
    # Shape: (...)
    sdc_valid_curr = datatypes.select_by_onehot(
        obj_valid_curr[..., 0],
        simulator_state.object_metadata.is_sdc,
        keepdims=False,
    )

    # # Shape: (..., num_paths, num_points_per_path)
    # dist_raw = jnp.linalg.norm(
    #     sdc_paths.xy - jnp.expand_dims(sdc_xy_curr, axis=(-2, -3)),
    #     axis=-1,
    #     keepdims=False,
    # )
    # # Only consider valid on-route paths.
    # dist = jnp.where(sdc_paths.valid & sdc_paths.on_route, dist_raw, jnp.inf)
    # # Only consider valid SDC states.
    # dist = jnp.where(
    #     jnp.expand_dims(sdc_valid_curr, axis=(-1, -2)), dist, jnp.inf
    # )
    # dist_path = jnp.min(dist, axis=-1, keepdims=True)  # (..., num_paths, 1)
    # idx = jnp.argmin(dist_path, axis=-2, keepdims=True)  # (..., 1, 1)
    # min_dist_path = jnp.min(dist, axis=(-1, -2))  # (...)

    # # Shape: (..., max(num_points_per_path))
    # ref_path = jax.tree_util.tree_map(
    #     lambda x: jnp.take_along_axis(x, indices=idx, axis=-2)[..., 0, :],
    #     sdc_paths,
    # )

    # def get_arclength_for_pts(xy: jax.Array, path: datatypes.Paths):
    #   # Shape: (..., max(num_points_per_path))
    #   dist_raw = jnp.linalg.norm(
    #       xy[..., jnp.newaxis, :] - path.xy, axis=-1, keepdims=False
    #   )
    #   dist = jnp.where(path.valid, dist_raw, jnp.inf)
    #   idx = jnp.argmin(dist, axis=-1, keepdims=True)
    #   # (..., )
    #   return jnp.take_along_axis(path.arc_length, indices=idx, axis=-1)[..., 0]

    # start_dist = get_arclength_for_pts(sdc_xy_start, ref_path)
    # end_dist = get_arclength_for_pts(sdc_xy_end, ref_path)
    # curr_dist = get_arclength_for_pts(sdc_xy_curr, ref_path)

    # progress = jnp.where(
    #     end_dist == start_dist,
    #     FULL_PROGRESS_VALUE,
    #     (curr_dist - start_dist) / (end_dist - start_dist),
    # )
    # valid = jnp.isfinite(min_dist_path)
    # progress = jnp.where(valid, progress, 0.0)

    sdc_traj = datatypes.select_by_onehot(
        simulator_state.log_trajectory,
        simulator_state.object_metadata.is_sdc,
        keepdims=False,
    )
    # calculate arch length
    prior_pts = sdc_traj.xy[...,:-1,:]
    next_pts = sdc_traj.xy[...,1:,:]
    # (1,bs,num_timesteps-1)
    dist_raw = jnp.linalg.norm(next_pts - prior_pts, axis=-1,keepdims=False)
    # padding the first point
    # (1,bs,num_timesteps)
    dist_raw = jnp.concatenate([jnp.zeros_like(dist_raw[...,0:1]), dist_raw], axis=-1)
    arc_to_start = jnp.cumsum(dist_raw, axis=-1)
    init_dist = arc_to_start[...,-1]
    arc_to_termin = init_dist[...,jnp.newaxis]-arc_to_start


    # [...,time_stamp,1]
    dist_curr2traj = jnp.linalg.norm(sdc_xy_curr[...,jnp.newaxis,:]-sdc_traj.xy, axis=-1,keepdims=False)
    dist_curr2traj = jnp.where(sdc_traj.valid, dist_curr2traj, jnp.inf)
    idx_min_dist = jnp.argmin(dist_curr2traj, axis=-1)
    # projected_xy = jnp.take_along_axis(sdc_traj.xy, indices=idx_min_dist[...,jnp.newaxis,jnp.newaxis], axis=-2)
    # # (1,bs,1,2) -> (1,bs,2)
    # projected_xy = projected_xy[...,0,:]
    dist2des = jnp.take_along_axis(arc_to_termin, indices=idx_min_dist[...,jnp.newaxis], axis=-1)
    # (1,bs,1) - > (1,bs)
    dist2des = dist2des[...,0]
    def close(x):
      return 1-x
    def far(x):
      return (init_dist-x)/(init_dist)
    def arrive(x):
      return jnp.ones_like(x)
    def not_arrive(x):
      return jnp.zeros_like(x)
    valid = sdc_valid_curr
    # init_dist = round(jnp.sqrt(jnp.sum((sdc_xy_end-sdc_xy_start)**2,axis=-1)),1)
    # dist2des = jnp.sqrt(jnp.sum((projected_xy-sdc_xy_end)**2,axis=-1))
    # dist2start = jnp.sqrt(jnp.sum((sdc_xy_curr-sdc_xy_start)**2,axis=-1))
    # jax.debug.print("{x}", x=init_dist)
    '''init dist lower than 3m or dis2des lower than 3m, then progress = 1'''
    progress = jnp.where(
        jnp.logical_or(init_dist<=3,dist2des<=3),
        FULL_PROGRESS_VALUE,
        (init_dist - dist2des) / (init_dist),
    )
    # progress = jax.lax.cond(init_dist==0,close,far,dist2des)
    # progress = jax.lax.cond(progress>=0.95,arrive,not_arrive,jnp.array(0))
    # if dist2des < 1:
    #     progress = 1
    # else:
    # progress = (init_dist-dist2des)/(init_dist)
    # progress = dist2start/init_dist
    # progress = 0.5* jnp.log(2 * (init_dist / dist)-1)
    '''align the shape for linear combination, from (1,bs) -> (1,bs,num_objs)'''
    valid = jnp.repeat(valid[...,jnp.newaxis],simulator_state.num_objects,axis=-1)
    progress = jnp.repeat(progress[...,jnp.newaxis],simulator_state.num_objects,axis=-1)
    return abstract_metric.MetricResult.create_and_validate(progress, valid)


class OffRouteMetric(abstract_metric.AbstractMetric):
  """Off-route metric for the SDC.

  The SDC is considered off-route either if 1) it is farther than
  MAX_DISTANCE_TO_ROUTE_PATH from the closest on-route path, or 2) it is farther
  from the closest on-route path than the closest off-route path by
  MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH.

  If the SDC is on-route, the SDC trajectory is completely invalid, or there are
  no valid on-route paths, it returns zero.

  If the SDC is off-route, this metric returns the distance to the closest valid
  on-route path. If there are no valid on-route paths, it returns the distance
  to the closest valid off-route path.
  """

  MAX_DISTANCE_TO_ROUTE_PATH = 5  # Meters.
  MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH = 2  # Meters.

  def compute(
      self, simulator_state: datatypes.SimulatorState
  ) -> abstract_metric.MetricResult:
    """Computes the off route metric.

    Args:
      simulator_state: The current simulator state of shape (....).

    Returns:
      A (...) MetricResult containing the metric result described above.

    Raises:
      ValueError: If `simulator_state.sdc_paths` is undefined.
    """
    sdc_paths = simulator_state.sdc_paths
    if sdc_paths is None:
      raise ValueError(
          'SimulatorState.sdc_paths required to compute the off-route metric.'
      )

    # Shape: (..., num_objects, num_timesteps=1, 2)
    obj_xy = datatypes.dynamic_slice(
        simulator_state.sim_trajectory.xy,
        simulator_state.timestep,
        1,
        axis=-2,
    )
    # Shape: (..., 2)
    sdc_xy = datatypes.select_by_onehot(
        obj_xy[..., 0, :],
        simulator_state.object_metadata.is_sdc,
        keepdims=False,
    )
    # Shape: (..., num_objects, num_timesteps=1)
    obj_valid = datatypes.dynamic_slice(
        simulator_state.sim_trajectory.valid,
        simulator_state.timestep,
        1,
        axis=-1,
    )
    # Shape: (...)
    sdc_valid = datatypes.select_by_onehot(
        obj_valid[..., 0],
        simulator_state.object_metadata.is_sdc,
        keepdims=False,
    )

    # Shape: (..., num_paths, num_points_per_path)
    sdc_dist_to_paths = jnp.linalg.norm(
        sdc_xy[..., jnp.newaxis, jnp.newaxis, :] - sdc_paths.xy,
        axis=-1,
        keepdims=False,
    )
    # Set distances to invalid paths to inf.
    sdc_dist_to_valid_paths = jnp.where(
        sdc_paths.valid, sdc_dist_to_paths, jnp.inf
    )
    # Set distances to invalid SDC states to inf.
    sdc_dist_to_valid_paths = jnp.where(
        jnp.expand_dims(sdc_valid, (-1, -2)), sdc_dist_to_valid_paths, jnp.inf
    )
    sdc_dist_to_valid_on_route_paths = jnp.where(
        sdc_paths.on_route, sdc_dist_to_valid_paths, jnp.inf
    )
    sdc_dist_to_valid_off_route_paths = jnp.where(
        ~sdc_paths.on_route, sdc_dist_to_valid_paths, jnp.inf
    )

    # Shape: (...)
    min_sdc_dist_to_valid_on_route_paths = jnp.min(
        sdc_dist_to_valid_on_route_paths, axis=(-1, -2)
    )
    min_sdc_dist_to_valid_off_route_paths = jnp.min(
        sdc_dist_to_valid_off_route_paths, axis=(-1, -2)
    )

    sdc_off_route = (
        min_sdc_dist_to_valid_on_route_paths > self.MAX_DISTANCE_TO_ROUTE_PATH
    ) | (
        min_sdc_dist_to_valid_on_route_paths
        - min_sdc_dist_to_valid_off_route_paths
        > self.MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH
    )

    off_route = jnp.where(
        sdc_off_route, min_sdc_dist_to_valid_on_route_paths, 0.0
    )
    valid = jnp.isfinite(off_route)
    off_route = jnp.where(valid, off_route, 0.0)
    return abstract_metric.MetricResult.create_and_validate(off_route, valid)
