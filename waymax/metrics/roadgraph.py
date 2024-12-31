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

"""Metrics functions relating to roadgraph."""

import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric


class WrongWayMetric(abstract_metric.AbstractMetric):
  """Wrong-way metric for SDC.

  This metric checks if SDC is driving into wrong driving the wrong way or path.
  It first computes the distance to the closest roadgraph point in all valid
  paths that the SDC can drive along from its starting position. If the distance
  is larger than the threhold WRONG_WAY_THRES, it's considered wrong-way and
  returns the distance; otherwise, it's driving on the legal lanes, and returns
  0.0.
  """

  WRONG_WAY_THRES = 3.5  # In meter
  YAW_THRES = 1
  @jax.named_scope('WrongWayMetric.compute')
  def compute(
      self, simulator_state: datatypes.SimulatorState
  ) -> abstract_metric.MetricResult:
    # (..., num_objects, num_timesteps, 2) -->
    # (..., num_objects, num_timesteps=1, 2)
    # obj_xy = datatypes.dynamic_slice(
    #     simulator_state.sim_trajectory.xy,
    #     simulator_state.timestep,
    #     1,
    #     axis=-2,
    # )
    # # sdc_xy has shape: (..., 2)
    # sdc_xy = datatypes.select_by_onehot(
    #     obj_xy[..., 0, :],
    #     simulator_state.object_metadata.is_sdc,
    #     keepdims=False,
    # )
    # sdc_paths = simulator_state.sdc_paths
    # # pytype: disable=attribute-error
    # # (..., num_paths, num_points_per_path)
    # dist_raw = jnp.linalg.norm(
    #     sdc_xy[..., jnp.newaxis, jnp.newaxis, :] - sdc_paths.xy,
    #     axis=-1,
    #     keepdims=False,
    # )
    # dist = jnp.where(sdc_paths.valid, dist_raw, jnp.inf)
    # # pytype: enable=attribute-error
    # min_dist = jnp.min(dist, axis=(-1, -2))
    # valid = jnp.isfinite(min_dist)
    # value = jnp.where((min_dist < self.WRONG_WAY_THRES) | ~valid, 0, min_dist)
    '''debug wrong way'''
    obj_yaw_curr = datatypes.dynamic_slice(
        simulator_state.sim_trajectory.yaw,
        simulator_state.timestep,
        1,
        axis=-1,
    )
    sdc_yaw_curr = datatypes.select_by_onehot(
        obj_yaw_curr[..., 0],
        simulator_state.object_metadata.is_sdc,
        keepdims=False,
    )
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
    sdc_traj = datatypes.select_by_onehot(
    simulator_state.log_trajectory,
    simulator_state.object_metadata.is_sdc,
    keepdims=False,
    )
    dist_curr2traj = jnp.linalg.norm(sdc_xy_curr[...,jnp.newaxis,:]-sdc_traj.xy, axis=-1,keepdims=False)
    dist_curr2traj = jnp.where(sdc_traj.valid, dist_curr2traj, jnp.inf)
    idx_min_dist = jnp.argmin(dist_curr2traj, axis=-1)
    projected_xy = jnp.take_along_axis(sdc_traj.xy, indices=idx_min_dist[...,jnp.newaxis,jnp.newaxis], axis=-2)
    dist_ego2project_xy = jnp.linalg.norm(sdc_xy_curr[...,jnp.newaxis,:]-projected_xy, axis=-1)
    # (1,bs,1) - > (1,bs)
    dist_ego2project_xy = dist_ego2project_xy[...,0]
    # fetch sdc direction
    projected_yaw = jnp.take_along_axis(sdc_traj.yaw, indices=idx_min_dist[...,jnp.newaxis],axis=-1)
    delta_yaw = jnp.abs(projected_yaw[...,0]-sdc_yaw_curr)
    # if the ego is arrived, then no offroute
    dist_ego2project_xy = jnp.where(idx_min_dist==sdc_traj.xy.shape[-2]-1, 0.0, dist_ego2project_xy)
    delta_yaw = jnp.where(delta_yaw==sdc_traj.xy.shape[-2]-1, 0.0, delta_yaw)

    critieria = jnp.logical_or(dist_ego2project_xy > self.WRONG_WAY_THRES, delta_yaw > self.YAW_THRES)
    # jax.debug.print('{x} {y}',x=delta_yaw,y=critieria)
    offroute_mask = jnp.where(critieria, 1.0, 0.0)
    value = offroute_mask
    valid = sdc_valid_curr
    '''align the shape for linear combination, from (1,bs) -> (1,bs,num_objs)'''
    valid = jnp.repeat(valid[...,jnp.newaxis],simulator_state.num_objects,axis=-1)
    value = jnp.repeat(value[...,jnp.newaxis],simulator_state.num_objects,axis=-1)
    '''end'''
    return abstract_metric.MetricResult.create_and_validate(value, valid)


class OffroadMetric(abstract_metric.AbstractMetric):
  """Offroad metric.

  This metric returns 1.0 if the object is offroad.
  """

  @jax.named_scope('OffroadMetric.compute')
  def compute(
      self, simulator_state: datatypes.SimulatorState
  ) -> abstract_metric.MetricResult:
    """Computes the offroad metric.

    Args:
      simulator_state: Updated simulator state to calculate metrics for. Will
        compute the offroad metric for timestep `simulator_state.timestep`.

    Returns:
      An array containing the metric result of the same shape as the input
        trajectories. The shape is (..., num_objects).
    """
    current_object_state = datatypes.dynamic_slice(
        simulator_state.sim_trajectory,
        simulator_state.timestep,
        1,
        -1,
    )
    offroad = is_offroad(current_object_state, simulator_state.roadgraph_points)
    valid = jnp.ones_like(offroad, dtype=jnp.bool_)
    return abstract_metric.MetricResult.create_and_validate(
        offroad.astype(jnp.float32), valid
    )


def is_offroad(
    trajectory: datatypes.Trajectory,
    roadgraph_points: datatypes.RoadgraphPoints,
) -> jax.Array:
  """Checks if the given trajectory is offroad.

  This determines the signed distance between each bounding box corner and the
  closest road edge (median or boundary). If the distance is negative, then the
  trajectory is onroad else offroad.

  Args:
    trajectory: Agent trajectories to test to see if they are on or off road of
      shape (..., num_objects, num_timesteps). The bounding boxes derived from
      center and shape of the trajectory will be used to determine if any point
      in the box is offroad. The num_timesteps dimension size should be 1.
    roadgraph_points: All of the roadgraph points in the run segment of shape
      (..., num_points). Roadgraph points of type `ROAD_EDGE_BOUNDARY` and
      `ROAD_EDGE_MEDIAN` are used to do the check.

  Returns:
    agent_mask: a bool array with the shape (..., num_objects). The value is
    True if the bbox is offroad.
  """
  # Shape: (..., num_objects, num_corners=4, 2).
  bbox_corners = jnp.squeeze(trajectory.bbox_corners, axis=-3)
  # Add in the Z dimension from the current center. This assumption will help
  # disambiguate between different levels of the roadgraph (i.e. under and over
  # passes).
  # Shape: (..., num_objects, 1, 1).
  z = jnp.ones_like(bbox_corners[..., 0:1]) * trajectory.z[..., jnp.newaxis, :]
  # Shape: (..., num_objects, num_corners=4, 3).
  bbox_corners = jnp.concatenate((bbox_corners, z), axis=-1)
  shape_prefix = bbox_corners.shape[:-3]
  num_agents, num_points, dim = bbox_corners.shape[-3:]
  # Shape: (..., num_objects * num_corners=4, 3).
  bbox_corners = jnp.reshape(
      bbox_corners, [*shape_prefix, num_agents * num_points, dim]
  )
  # Here we compute the signed distance between the given trajectory and the
  # roadgraph points. The shape prefix represents a set of batch dimensions
  # denoted above as (...). Here we call a set of nested vmaps for each of the
  # batch dimensions in the shape prefix to allow for more flexible parallelism.
  compute_fn = compute_signed_distance_to_nearest_road_edge_point
  for _ in shape_prefix:
    compute_fn = jax.vmap(compute_fn)

  # Shape: (..., num_objects * num_corners=4).
  distances = compute_fn(bbox_corners, roadgraph_points)
  # Shape: (..., num_objects, num_corners=4).
  distances = jnp.reshape(distances, [*shape_prefix, num_agents, num_points])
  # Shape: (..., num_objects).
  return jnp.any(distances > 0.0, axis=-1)


def compute_signed_distance_to_nearest_road_edge_point(
    query_points: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
    z_stretch: float = 2.0,
) -> jax.Array:
  """Computes the signed distance from a set of queries to roadgraph points.

  Args:
    query_points: A set of query points for the metric of shape
      (..., num_query_points, 3).
    roadgraph_points: A set of roadgraph points of shape (num_points).
    z_stretch: Tolerance in the z dimension which determines how close to
      associate points in the roadgraph. This is used to fix problems with
      overpasses.

  Returns:
    Signed distances of the query points with the closest road edge points of
      shape (num_query_points). If the value is negative, it means that the
      actor is on the correct side of the road, if it is positive, it is
      considered `offroad`.
  """
  # Shape: (..., num_points, 3).
  sampled_points = roadgraph_points.xyz
  # Shape: (..., num_query_points, num_points, 3).
  differences = sampled_points - jnp.expand_dims(query_points, axis=-2)
  # Stretch difference in altitude to avoid over/underpasses.
  # Shape: (..., num_query_points, num_points, 3).
  z_stretched_differences = differences * jnp.array([[[1.0, 1.0, z_stretch]]])
  # Shape: (..., num_query_points, num_points).
  square_distances = jnp.sum(z_stretched_differences**2, axis=-1)
  # Do not consider invalid points.
  # Shape: (num_points).
  is_road_edge = datatypes.is_road_edge(roadgraph_points.types)
  # Shape: (..., num_query_points, num_points).
  square_distances = jnp.where(
      roadgraph_points.valid & is_road_edge, square_distances, float('inf')
  )
  # Shape: (..., num_query_points).
  nearest_indices = jnp.argmin(square_distances, axis=-1)
  # Shape: (..., num_query_points).
  prior_indices = jnp.maximum(
      jnp.zeros_like(nearest_indices), nearest_indices - 1
  )
  # Shape: (..., num_query_points, 2).
  nearest_xys = sampled_points[nearest_indices, :2]
  # Direction of the road edge at the nearest points. Should be normed and
  # tangent to the road edge.
  # Shape: (..., num_query_points, 2).
  nearest_vector_xys = roadgraph_points.dir_xyz[nearest_indices, :2]
  # Direction of the road edge at the points that precede the nearest points.
  # Shape: (..., num_query_points, 2).
  prior_vector_xys = roadgraph_points.dir_xyz[prior_indices, :2]
  # Shape: (..., num_query_points, 2).
  points_to_edge = query_points[..., :2] - nearest_xys
  # Get the signed distance to the half-plane boundary with a cross product.
  cross_product = jnp.cross(points_to_edge, nearest_vector_xys)
  cross_product_prior = jnp.cross(points_to_edge, prior_vector_xys)
  # If the prior point is contiguous, consider both half-plane distances.
  # Shape: (..., num_query_points).
  prior_point_in_same_curve = jnp.equal(
      roadgraph_points.ids[nearest_indices], roadgraph_points.ids[prior_indices]
  )
  # Shape: (..., num_query_points).
  offroad_sign = jnp.sign(
      jnp.where(
          jnp.logical_and(
              prior_point_in_same_curve, cross_product_prior < cross_product
          ),
          cross_product_prior,
          cross_product,
      )
  )
  # Shape: (..., num_query_points).
  return (
      jnp.linalg.norm(nearest_xys - query_points[:, :2], axis=-1) * offroad_sign
  )