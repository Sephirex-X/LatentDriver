import jax
import jax.numpy as jnp
from waymax import datatypes
import numpy as np
from waymax.datatypes import observation
import chex

def drop_zero_on_roadobs(roadgraph_obs:np.array,
                         max_roadgraph_segments:int = 40):
    num_device,B,_,attribute = roadgraph_obs.shape

    vali_mask = np.where(roadgraph_obs.sum(-1)!=0,True,False)
    # (num_device,B,1000) -> (num_device*B,1000)
    vali_mask = vali_mask.reshape(num_device*B,-1)
    roadgraph_obs = roadgraph_obs.reshape(num_device*B,-1,attribute)
    def flip_N(A,N):
        '''
            if segments larger than max, then drop, else pad
        '''
        assert A.ndim == 1
        if N > 0:
            # Find the indices of `False` values in `A`
            false_indices = np.where(A == False)[0]
            # Check if the number of `False` values is greater than or equal to N
            if len(false_indices) >= N:
                # Change the first N `False` values to `True`
                A[false_indices[:N]] = True 
        elif N<0:
            # Find the indices of `True` values in `A`
            true_indices = np.where(A == True)[0]
            # Check if the number of `True` values is greater than or equal to -N
            if len(true_indices) >= -N:
                # Change the first -N `True` values to `False`
                A[true_indices[:-N]] = False
        else:
            pass
    # (bs,)
    valid_nums = vali_mask.sum(-1)
    for bs_id in range(B*num_device):
        # if valid_nums[bs_id] < cfg.max_roadgraph_segments:
        # modified in place
        flip_N(vali_mask[bs_id],max_roadgraph_segments-valid_nums[bs_id])
        # else:
        #     raise ValueError('max_roadgraph_segments should be larger than {}'.format(valid_nums[bs_id]))
    roadgraph_obs = roadgraph_obs[vali_mask].reshape(num_device,B,-1,attribute)    
    return roadgraph_obs
def preprocess_data_dist_jnp(data: dict[jnp.array],
                             max_roadgraph_segments: int = 40):
    '''
        The data dict is updated in place here to numpy
        TODO: write max_roadgraph_segments to config
    '''
    type_route_seg = np.array(data['route_segments'])
    type_vehicles = np.array(data['vehicle_segments'])
    type_roadobs = np.array(data['roadgraph_obs'])
    type_roadobs = drop_zero_on_roadobs(type_roadobs,max_roadgraph_segments)
    # update the data
    data['roadgraph_obs'] = type_roadobs
    data['route_segments'] = type_route_seg
    data['vehicle_segments'] = type_vehicles
    obs = np.concatenate([type_route_seg, type_vehicles, type_roadobs], axis=2,dtype=np.float32)
    data['obs'] = obs
    # obs [num_devices, collected bs, numbers of types, 7]
    return obs
def get_padding_mask(array):
    return array.sum(axis=-1) == 0
def get_vehicle_obs(sdc_obs, timestep):
    # modified for time-step
    # sdc_obs.trajectory.xy.shape [num_gpus,bs,objs,timesteps,2]
    valid_mask = sdc_obs.trajectory.valid[...,:,:timestep,jnp.newaxis] 
    xy = sdc_obs.trajectory.xy[...,:,:timestep,:] * valid_mask
    # speed [objs,1]
    speed = sdc_obs.trajectory.speed[...,:,:timestep,jnp.newaxis] * valid_mask
    # yaw [objs,1]
    yaw = sdc_obs.trajectory.yaw[...,:,:timestep,jnp.newaxis] * 180 / np.pi * valid_mask
    width = sdc_obs.trajectory.width[...,:,:timestep,jnp.newaxis] * valid_mask
    length = sdc_obs.trajectory.length[...,:,:timestep,jnp.newaxis] * valid_mask
    # jax.debug.print("xy = {x}",x=xy)
    # jax.debug.print("valid = {x}",x=valid_mask)
    # jax.debug.print("yaw_orin={x}",x=sdc_obs.trajectory.yaw.sum())
    # jax.debug.print("xy={x},speed={y},yaw={z},width={w},length={v}",x=xy.sum(),y=speed.sum(),z=yaw.sum(),w=width.sum(),v=length.sum())
    vehicle_obs = jnp.concatenate([xy,width,length,yaw,speed],axis=-1)
    return vehicle_obs
def downsampled_elements_transformation(elements,
                                        pose_global2ego,
                                        sdc_yaw,):
    elements = jnp.array(elements)
    # (bs, max_roadgraph_segments, 6)
    elements_shape = elements.shape
    # (bs, max_roadgraph_segments, 6) -> (1,bs,1,max_roadgraph_segments,6)
    elements = elements[:,jnp.newaxis,...]
    unpad_mask = jnp.where(elements.sum(axis=-1) != 0, True, False)
    # transform xy into ego frame (1,bs,1,max_roadgraph_segments,2)
    transformed_xy = observation.geometry.transform_points(
      pts=elements[...,0:2],
      pose_matrix=pose_global2ego.matrix)
    # (1,bs,1) -> (1,bs,1,max_roadgraph_segments)
    sdc_yaw_ = jnp.repeat(sdc_yaw[...,jnp.newaxis], elements.shape[-2], axis=-1)
    '''debug'''
    # !!! sdc is in radian the data we store is in degree
    sdc_yaw_ = sdc_yaw_ * 180 / jnp.pi
    transformed_yaw = observation.geometry.transform_yaw(-sdc_yaw_,elements[...,-2])
    # (1,bs,1,max_roadgraph_segments) -> (1,bs,1,max_roadgraph_segments,1)
    transformed_yaw = transformed_yaw[...,jnp.newaxis]
    # (1,bs,1,max_roadgraph_segments,6)
    new_elements = jnp.concatenate([transformed_xy,
                                    elements[...,2:4],
                                    transformed_yaw,
                                    elements[...,5:6]],axis=-1)
    return new_elements,unpad_mask
@jax.pmap
def get_obs_from_routeandmap_saved(state:datatypes.SimulatorState,
                           whole_map:np.array,
                           route:np.array,
                           vis_distance:list=[80, 20],): #for width and height
    '''
        TODO: write vis_distance into config
    '''
    
    assert len(state.shape)==1
    def padding_exceed(array,dis):
        x_exceed_mask = jnp.abs(array[...,0])>dis[0]//2
        y_exceed_mask = jnp.abs(array[...,1])>dis[1]//2
        exceed_mask = jnp.logical_or(x_exceed_mask,y_exceed_mask)
        # array[exceed_mask] *= 0
        array = jnp.where(exceed_mask[...,jnp.newaxis], 0, array)
        return array,exceed_mask

    def add_type_and_reset_padding(array, type_id):
        padding_mask = get_padding_mask(array)
        type_array = jnp.concatenate([jnp.ones(array.shape[:2])[...,jnp.newaxis] * type_id, array], axis=-1)
        # type_array[padding_mask] *= 0
        type_array = jnp.where(padding_mask[...,jnp.newaxis], 0, type_array)
        return type_array
    # whole_map (bs, max_segs, 6)
    B,P = state.roadgraph_points.shape
    # Select the XY position at the current timestep.
    # Shape: (..., num_agents, 2)
    obj_xy = state.current_sim_trajectory.xy[..., 0, :]
    obj_yaw = state.current_sim_trajectory.yaw[..., 0]
    obj_valid = state.current_sim_trajectory.valid[..., 0]

    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    # (1,bs,2)
    sdc_xy = jnp.take_along_axis(obj_xy, sdc_idx[..., jnp.newaxis], axis=-2)
    sdc_yaw = jnp.take_along_axis(obj_yaw, sdc_idx, axis=-1)
    # jax.debug.print("{x}", x=sdc_yaw)
    sdc_valid = jnp.take_along_axis(obj_valid, sdc_idx, axis=-1)
    # jax.debug.print("object_yaw = {x}, sdc_yaw = {y}",x=obj_yaw.sum(),y=sdc_yaw.sum())
    # jax.debug.print("{x}", x=sdc_xy.sum())
    # The num_obj is 1 because the it is computing the observation for SDC, and
    # there is only 1 SDC per scene.
    num_obj = 1
    time_step = 10
    global_obs = observation.global_observation_from_state(
        state, time_step, num_obj=num_obj
    )
    is_ego = state.object_metadata.is_sdc[..., jnp.newaxis, :]
    global_obs_filter = global_obs.replace(
        is_ego=is_ego,
    )


    pose2d = observation.ObjectPose2D.from_center_and_yaw(
        xy=sdc_xy, yaw=sdc_yaw, valid=sdc_valid
    )
    # jax.debug.print("pose2d = {x}", x=pose2d.matrix.sum())
    chex.assert_equal(pose2d.shape, state.shape + (1,))
    sdc_obs = observation.transform_observation(global_obs_filter, pose2d)
    pose_global2ego = observation.combine_two_object_pose_2d(src_pose=global_obs_filter.pose2d, dst_pose=pose2d)
    # for roadgraph
    whole_map_shape = whole_map.shape
    new_whole_map,unpad_mask_map = downsampled_elements_transformation(whole_map,pose_global2ego,sdc_yaw)
    # ROI_wh = [-vis_distance, vis_distance]
    # ROI_wh = jnp.array(ROI_wh)
    ROI_wh = jnp.array(vis_distance)
    # x
    mask_x = jnp.logical_and(new_whole_map[...,0] >= -ROI_wh[0]//2,
                             new_whole_map[...,0] <= ROI_wh[0]//2)
    # y
    mask_y = jnp.logical_and(new_whole_map[...,1] >= -ROI_wh[1]//2,
                             new_whole_map[...,1] <= ROI_wh[1]//2)
    mask_roi = jnp.logical_and(mask_x, mask_y)
    
    whole_map_roi = new_whole_map*mask_roi[...,jnp.newaxis]
    whole_map_roi = whole_map_roi * unpad_mask_map[...,jnp.newaxis]

    roadgraph_obs = whole_map_roi.reshape(whole_map_shape)
    # vali_mask = jnp.where(roadgraph_obs.sum(-1)!=0,True,False)
    # # (bs,)
    # valid_nums = vali_mask.sum(-1)
    # for bs_id in range(B):
    #     # if valid_nums[bs_id] < cfg.max_roadgraph_segments:
    #     # modified in place
    #     flip_N(vali_mask[bs_id],max_roadgraph_segments-valid_nums[bs_id])
    #     # else:
    #     #     raise ValueError('max_roadgraph_segments should be larger than {}'.format(valid_nums[bs_id]))
    # roadgraph_obs = roadgraph_obs[vali_mask].reshape(B,-1,6)
    type_roadobs = add_type_and_reset_padding(roadgraph_obs, 3)    


    # for vehicle
    # jax.debug.print("sdc_obs = {x}",x=sdc_obs.trajectory.yaw.sum())
    # jax.debug.print("sdc_validate_mask = {x} shape = {y}",x=sdc_obs.trajectory.valid[...,:,-1:].sum(),y=sdc_obs.trajectory.valid.shape)
    vehicle_sgements = get_vehicle_obs(sdc_obs,time_step)
    cur_vehicle_sgements = vehicle_sgements[...,-1,:]
    
    # jax.debug.print("veh_cat = {x}",x=vehicle_sgements.sum())
    veh_segs, vehicle_exceed_masks = padding_exceed(cur_vehicle_sgements,dis=ROI_wh)
    veh_segs = veh_segs.reshape(B,cur_vehicle_sgements.shape[-2],cur_vehicle_sgements.shape[-1])
    # for other agents trajs
    # (bs,num_objs,time_step-1,6)
    his_veh_trajs = vehicle_sgements[...,:-1,:]
    his_types = jnp.ones(his_veh_trajs.shape[:-1])[...,jnp.newaxis] * 2
    his_veh_trajs = jnp.concatenate([his_types, his_veh_trajs], axis=-1)
    # set sdc to false
    vehicle_exceed_masks.at[jnp.linspace(0,state.shape[0]-1,state.shape[0]).astype(int),
                            jnp.linspace(0,B-1,B).astype(int),
                            sdc_idx.reshape(-1)].set(False)
    his_veh_trajs = jnp.where(vehicle_exceed_masks[...,jnp.newaxis,jnp.newaxis],0,his_veh_trajs).reshape((-1,)+his_veh_trajs.shape[2:])
    
    # jax.debug.print("padding_exceed = {x}",x=veh_segs.sum())
    # type_vehicles [bs,7]
    type_vehicles = add_type_and_reset_padding(veh_segs, 2)
    # jax.debug.print("add_type_and_reset_padding = {x}",x=type_vehicles.sum())
    # set sdc type on type_vehicles into 4
    type_vehicles = type_vehicles.at[jnp.linspace(0,B-1,B).astype(int),sdc_idx.reshape(-1),0].set(4)
    # jax.debug.print("type_vehicles into 4 = {x}",x=type_vehicles.sum())
    # for route
    route_shape = route.shape
    new_route,unpad_mask = downsampled_elements_transformation(route,pose_global2ego,sdc_yaw)
    new_route = new_route * unpad_mask[...,jnp.newaxis]
    route_obs = new_route.reshape(route_shape)
    type_route_seg = add_type_and_reset_padding(route_obs, 1)
    # for vis sdc_obs
    # sdc_obs = jax.tree_util.tree_map(lambda x: x[0,:,:], sdc_obs)
    return dict(
        route_segments=type_route_seg,
        vehicle_segments=type_vehicles,
        roadgraph_obs=type_roadobs,
        his_veh_trajs = his_veh_trajs
        # traj_obs=traj_obs,
        # traj_next_stamp=traj_next_stamp,
    ), sdc_obs    