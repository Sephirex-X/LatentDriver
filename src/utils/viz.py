from waymax import datatypes
from typing import Any, Optional

import jax
import matplotlib
import numpy as np

from waymax import config as waymax_config
from waymax import datatypes
from waymax.utils import geometry
from waymax.visualization import color
from waymax.visualization import utils
import matplotlib.pyplot as plt
from waymax.visualization.viz import (
    _index_pytree,
    plot_roadgraph_points,
    plot_traffic_light_signals_as_points,
)
def get_color(id):
    np.random.seed(id)  # Use id as seed to always get the same color for the same id
    return np.random.rand(3,)  # Return a tuple of 3 random numbers between 0 and 1
max_time_steps = 91


def _plot_bounding_boxes(
    ax: matplotlib.axes.Axes,
    traj_5dof: np.ndarray,
    time_idx: int,
    is_controlled: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Helper function to plot multiple bounding boxes across time."""

    # Plots bounding boxes (traj_5dof) with shape: (A, T)
    # is_controlled: (A,)
    # valid: (A, T)
    valid_controlled = is_controlled[:, np.newaxis] & valid
    valid_context = ~is_controlled[:, np.newaxis] & valid
    transparency = time_idx / max_time_steps
    num_obj = traj_5dof.shape[0]
    time_indices = np.tile(np.arange(traj_5dof.shape[1])[np.newaxis, :], (num_obj, 1))
    # Shrinks bounding_boxes for non-current steps.
    traj_5dof[time_indices != time_idx, 2:4] /= 10
    utils.plot_numpy_bounding_boxes(
        ax=ax,
        bboxes=traj_5dof[(time_indices == time_idx) & valid_controlled],
        color=color.COLOR_DICT["controlled"],
    )
    # debug for pic

    #  ucommment for video
    utils.plot_numpy_bounding_boxes(
        ax=ax,
        bboxes=traj_5dof[(time_indices < time_idx) & valid_controlled],
        color=color.COLOR_DICT["controlled"],
        as_center_pts=True,
    )
    # debug for pic
    utils.plot_numpy_bounding_boxes(
        ax=ax,
        bboxes=traj_5dof[(time_indices == time_idx) & valid_context],
        color=color.COLOR_DICT["context"],
    )

    # Shows current overlap
    # (A, A)
    overlap_fn = jax.jit(geometry.compute_pairwise_overlaps)
    overlap_mask_matrix = overlap_fn(traj_5dof[:, time_idx])
    # Remove overlap against invalid objects.
    overlap_mask_matrix = np.where(valid[None, :, time_idx], overlap_mask_matrix, False)
    # (A,)
    overlap_mask = np.any(overlap_mask_matrix, axis=1)

    utils.plot_numpy_bounding_boxes(
        ax=ax,
        bboxes=traj_5dof[:, time_idx][overlap_mask & valid[:, time_idx]],
        color=color.COLOR_DICT["overlap"],
    )


def plot_trajectory(
    ax: matplotlib.axes.Axes,
    traj: datatypes.Trajectory,
    is_controlled: np.ndarray,
    time_idx: Optional[int] = None,
    indices: Optional[np.ndarray] = None,
) -> None:
    """Plots a Trajectory with different color for controlled and context.

    Plots the full bounding_boxes only for time_idx step, overlap is
    highlighted.

    Notation: A: number of agents; T: numbe of time steps; 5 degree of freedom:
    center x, center y, length, width, yaw.

    Args:
      ax: matplotlib axes.
      traj: a Trajectory with shape (A, T).
      is_controlled: binary mask for controlled object, shape (A,).
      time_idx: step index to highlight bbox, -1 for last step. Default(None) for
        not showing bbox.
      indices: ids to show for each agents if not None, shape (A,).
    """
    if len(traj.shape) != 2:
        raise ValueError("traj should have shape (A, T)")

    traj_5dof = np.array(
        traj.stack_fields(["x", "y", "length", "width", "yaw"])
    )  # Forces to np from jnp

    num_obj, num_steps, _ = traj_5dof.shape
    if time_idx is not None:
        if time_idx == -1:
            time_idx = num_steps - 1
        if time_idx >= num_steps:
            raise ValueError("time_idx is out of range.")

    # Adds id if needed.
    # if indices is not None and time_idx is not None:
    #     for i in range(num_obj):
    #         if not traj.valid[i, time_idx]:
    #             continue
    #         ax.text(
    #             traj_5dof[i, time_idx, 0] - 2,
    #             traj_5dof[i, time_idx, 1] + 2,
    #             f"{indices[i]}",
    #             zorder=10,
    #         )
    _plot_bounding_boxes(
        ax, traj_5dof, time_idx, is_controlled, traj.valid
    )  # pytype: disable=wrong-arg-types  # jax-ndarray


def plot_image(
    state: datatypes.SimulatorState,
    use_log_traj: bool = True,  # not used for now
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    rews=None,
) -> np.ndarray:
    """Plots np array image for SimulatorState.

    Args:
      state: A SimulatorState instance.
      use_log_traj: Set True to use logged trajectory, o/w uses simulated
        trajectory.
      viz_config: dict for optional config.
      batch_idx: optional batch index.
      highlight_obj: Represents the type of objects that will be highlighted with
        `color.COLOR_DICT['controlled']` color.

    Returns:
      np image.
    """
    if batch_idx > -1:
        if len(state.shape) == 1:
            # raise ValueError(
            #     'Expecting one batch dimension, got %s' % len(state.shape)
            # )
            state = _index_pytree(state, batch_idx)
        elif len(state.shape) == 2:
            device_id = batch_idx // state.shape[1]
            bs = batch_idx % state.shape[1]
            state = _index_pytree(state, device_id)
            state = _index_pytree(state, bs)
    if state.shape:
        raise ValueError("Expecting 0 batch dimension, got %s" % len(state.shape))

    viz_config = (
        utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
    )
    fig, ax = utils.init_fig_ax(viz_config)

    # 1. Plots trajectory.
    # traj = state.log_trajectory if use_log_traj else state.sim_trajectory
    slice_size = state.remaining_timesteps - 1
    traj = datatypes.update_by_slice_in_dim(
        inputs=state.sim_trajectory,
        updates=state.log_trajectory,
        inputs_start_idx=state.timestep + 1,
        slice_size=max(0, slice_size),
        axis=-1,
    )

    indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None
    is_controlled = datatypes.get_control_mask(state.object_metadata, highlight_obj)
    # 2. Plots road graph elements.
    plot_roadgraph_points(ax, state.roadgraph_points, verbose=False)
    plot_traffic_light_signals_as_points(
        ax, state.log_traffic_light, state.timestep, verbose=False
    )

    #   target pts for sdc
    #   (1,91,2)
    log_pts = state.log_trajectory.xy[is_controlled][0]
    end_pts = log_pts[-1, :]
    #   plot log trajs
    for pts in log_pts:
        ax.plot(pts[0], pts[1], ".", color="lightsalmon", ms=3, alpha=1)
    plot_trajectory(
        ax, traj, is_controlled, time_idx=state.timestep, indices=indices
    )  # pytype: disable=wrong-arg-types  # jax-ndarray

    # 3. Gets np img, centered on selected agent's current location.
    # [A, 2]
    current_xy = traj.xy[:, state.timestep, :]
    if viz_config.center_agent_idx == -1:
        xy = current_xy[state.object_metadata.is_sdc]
    else:
        xy = current_xy[viz_config.center_agent_idx]
    origin_x, origin_y = xy[0, :2]
    ax.axis(
        (
            origin_x - viz_config.back_x,
            origin_x + viz_config.front_x,
            origin_y - viz_config.back_y,
            origin_y + viz_config.front_y,
        )
    )
    if rews is not None:
        assert isinstance(rews, dict)
        pos_x = origin_x - 50
        pos_y = origin_y + 50
        for key, value in rews.items():
            value = value[batch_idx]
            ax.text(pos_x, pos_y, str(key), ha="center", va="bottom", color="r")
            ax.text(pos_x + 30, pos_y, str(value), ha="center", va="bottom", color="r")
            pos_y -= 5        
    #   plot end point
    ax.plot(end_pts[0], end_pts[1], "*", color="darkorange", ms=10, alpha=1)
    # disable ticks
    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False)
    return utils.img_from_fig(fig)

def plot_rec(batch_idx, rec,info=None, additional_points=None):
    '''
        additional_points [bs, num_pts, 2 or N] (x,y,...)
    '''
    viz_config = (
        utils.VizConfig()
    )
    fig, ax = utils.init_fig_ax(viz_config)
    origin_x =0 
    origin_y =0
    if additional_points is not None:
        ax.scatter(additional_points[batch_idx,:,0], additional_points[batch_idx,:,1],s=5)
    if info is not None:
        assert isinstance(info,dict)
        pos_x = origin_x - viz_config.back_x * 0.66
        pos_y = origin_y + viz_config.back_y * 0.66
        for key, value in info.items():
            value = value[batch_idx]
            ax.text(pos_x, pos_y, str(key), ha='center', va='bottom',color='r')
            ax.text(pos_x+30, pos_y, str(value), ha='center', va='bottom',color='r')
            pos_y -= 5
    ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))
    route_segments = rec[batch_idx]
    from matplotlib.patches import Rectangle
    id_list = []
    for element in route_segments:
        # skip padding
        if element.sum() == 0:
            continue          
        if len(element) == 7:
          element_type, x, y, w, h, yaw, id = element 
        elif len(element) == 6:
          x, y, w, h, yaw, id = element  
        center = np.array([x,y])
        anchor_point = np.array([-h/2,-w/2])
        cos = np.cos(np.radians(yaw))
        sin = np.sin(np.radians(yaw))
        matrix = np.array([
          [cos,sin],
          [-sin,cos]
        ])
        anchor_point = np.dot(anchor_point,matrix) + center
        id = int(id)
        color = (get_color(int(element_type))).tolist()
        rect = Rectangle(anchor_point, h, w, angle=(yaw), edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # ax.plot(x, y, 'o',color=color) 
        # if id not in id_list:
        #   id_list.append(id)
        #   ax.text(anchor_point[0], anchor_point[1], str(int(id)), ha='center', va='center',color='r',zorder=10)
    return utils.img_from_fig(fig)
def plot_seq2image(
    state: datatypes.SimulatorState,
    use_log_traj: bool = True,  # not used for now
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    rectangle_seg = None,
    rews=None,
) -> np.ndarray:
  start_time = 10
  overlap_fn = jax.jit(geometry.compute_pairwise_overlaps)
  if batch_idx > -1:
      if len(state.shape) == 1:
          # raise ValueError(
          #     'Expecting one batch dimension, got %s' % len(state.shape)
          # )
          state = _index_pytree(state, batch_idx)
      elif len(state.shape) == 2:
          device_id = batch_idx // state.shape[1]
          batch_idx = batch_idx % state.shape[1]
          state = _index_pytree(state, device_id)
          state = _index_pytree(state, batch_idx)
  if state.shape:
      raise ValueError("Expecting 0 batch dimension, got %s" % len(state.shape))
   #   
  traj = state.sim_trajectory
  indices = np.arange(traj.num_objects)
  is_controlled = datatypes.get_control_mask(state.object_metadata, highlight_obj)
#   (1,91,2)
  log_pts = state.log_trajectory.xy[is_controlled][0]
  start_pts = log_pts[0, :]
  end_pts = log_pts[-1, :]
  
  dis = np.linalg.norm(start_pts-end_pts)
  if dis > 50:
    dis *= 1.05
  else:
    dis = 75.0
  viz_config = (
      utils.VizConfig(
            front_x = dis,
            back_x = dis,
            front_y = dis,
            back_y = dis,
            px_per_meter=20,
          ) if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots road graph elements and vehs' init states.
  # 1.1 road graph elements
  plot_roadgraph_points(ax, state.roadgraph_points, verbose=False)
  plot_traffic_light_signals_as_points(
      ax, state.log_traffic_light, state.timestep, verbose=False
  )
  # 1.2 vehs' init states and log trajs


  # 1.2.1 log trajs
  #   (1,91,2)
  
  for pts in log_pts:
      ax.plot(pts[0], pts[1], ".", color='lightsalmon', ms=3, alpha=1)
  # 1.2.2 init states
  plot_trajectory(
      ax, traj, is_controlled, time_idx=start_time, indices=indices
  )  # pytype: disable=wrong-arg-types  # jax-ndarray

  # 2. iterates over all sim find the collision and plot the transparency traj
  # TODO: and off-road pts. 
  traj_5dof = np.array(
      traj.stack_fields(["x", "y", "length", "width", "yaw"])
  )  # Forces to np from jnp borrowed from "plot_trajectory"
  num_obj = traj_5dof.shape[0]
  time_indices = np.tile(np.arange(traj_5dof.shape[1])[np.newaxis, :], (num_obj, 1))
  valid = traj.valid
  valid_controlled = is_controlled[:, np.newaxis]
  valid_context = ~is_controlled[:, np.newaxis] & valid
# #   plot init for sdc
#   utils.plot_numpy_bounding_boxes(
#         ax=ax,
#         bboxes=traj_5dof[(time_indices == 0) & valid_controlled],
#         color=color.COLOR_DICT["controlled"],
#         alpha=0.5
#   )   

#   (1,91,2)
  sdc_pos_t = traj.xy[valid_controlled[...,0]]
  others_pos_t = traj.xy[valid_context[...,0]]
#   [::-1] is doing inverse for color
  for pts in others_pos_t:
      ax.scatter(pts[start_time:,0], pts[start_time:,1],s=3,c=np.arange(max_time_steps-start_time)[::-1],cmap='summer',alpha=0.8,zorder=2)
  for pts in sdc_pos_t:
      ax.scatter(pts[start_time:,0], pts[start_time:,1],s=3,c=np.arange(max_time_steps-start_time),cmap='cool',alpha=0.8,zorder=5)
  for t in range(max_time_steps-start_time):
    # transparency = ((t+1)/max_time_steps)
    # num_obj = traj_5dof.shape[0]
    # time_indices = np.tile(np.arange(traj_5dof.shape[1])[np.newaxis, :], (num_obj, 1))

    # utils.plot_numpy_bounding_boxes(
    #     ax=ax,
    #     bboxes=traj_5dof[(time_indices == t) & valid_controlled],
    #     color=color.COLOR_DICT["controlled"],
    #     as_center_pts=True,
    #     alpha=transparency,
    # )
    # utils.plot_numpy_bounding_boxes(
    #     ax=ax,
    #     bboxes=traj_5dof[(time_indices == t) & valid_context],
    #     color=np.array([0.28, 0.57, 0.54]),
    #     as_center_pts=True,
    #     alpha=transparency,
    # )
    # find and draw 5dof collision pts.
    overlap_mask_matrix = overlap_fn(traj_5dof[:, t])
    # Remove overlap against invalid objects.
    overlap_mask_matrix = np.where(valid[None, :, t], overlap_mask_matrix, False)
    overlap_mask = np.any(overlap_mask_matrix, axis=1)
    collision_pts = traj_5dof[:, t][overlap_mask & valid_controlled[:, t]]
    if len(collision_pts) != 0:
        # only draw once
        ax.plot(collision_pts[:,0], collision_pts[:,1], "X", color=color.COLOR_DICT["overlap"], linewidth=5,ms=10, alpha=1,zorder=6)
        break
    # print(collision_pts.shape)
  
  # 3. Gets np img, centered on selected agent's init location.
  # [A, 2]
  current_xy = traj.xy[:, 0, :]
  if viz_config.center_agent_idx == -1:
      xy = current_xy[state.object_metadata.is_sdc]
  else:
      xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis(
      (
          origin_x - viz_config.back_x,
          origin_x + viz_config.front_x,
          origin_y - viz_config.back_y,
          origin_y + viz_config.front_y,
      )
  )
  if rews is not None:
      assert isinstance(rews, dict)
      pos_x = origin_x - 50
      pos_y = origin_y + 50
      for key, value in rews.items():
          value = value[device_id, batch_idx]
          ax.text(pos_x, pos_y, str(key), ha="center", va="bottom", color="r")
          ax.text(pos_x + 30, pos_y, str(value), ha="center", va="bottom", color="r")
          pos_y -= 5
  #   plot end point
  ax.plot(end_pts[0], end_pts[1], "*", color='darkorange', ms=15, alpha=1,zorder=4)

  return fig

