import numpy as np
import jax
import jax.numpy as jnp
from waymax.datatypes.roadgraph import MapElementIds
import math
from src.utils.utils import saving_data
import os
from src.ops.crdp import crdp
def worker_route(route,max_route_segments,ego_car_width,path):
    route = rdp_downsample_route(np.array(route),float(ego_car_width))
    # padding
    route = padding(route, max_route_segments)
    saving_data(route,name=path,mode='np')
    
def worker_roadgraph(roadgraph,ids,max_roadgraph_segments,path):
    roadgraph = get_roadgraph_obs_(roadgraph,ids)
    # padding
    roadgraph = padding(roadgraph, max_roadgraph_segments)
    saving_data(roadgraph,name=path,mode='np')
    
from src.preprocess.identify_curve import get_length_curvature
# PARAMETERS SETTING
THRES = dict(
    straight = 0.03,
    turning = 0.18
)
YAW_THRES = dict(
    straight = 0.2,
)
def identifiy(max_curvature, sign, diff):
    if max_curvature == -1:
        intention = 'stationary'
    # elif max_curvature <= THRES['straight'] and diff < YAW_THRES['straight']:
    #     intention = 'straight'
    elif (max_curvature > THRES['straight'] and max_curvature < THRES['turning'] and diff > YAW_THRES['straight']) or \
        (max_curvature > 0.1 and max_curvature < THRES['turning']):
        intention = 'turning'
    elif max_curvature >= THRES['turning']:
        intention = 'U-turn'
    else:
        intention = 'straight'
        
    if intention in ['turning', 'U-turn']:
        if sign == 1:
            direction = 'left'
        elif sign == -1 :
            direction = 'right'
            # U-turn is impossible
            if intention == 'U-turn':
                intention = 'turning'
    else:
        direction = ''
    return intention, direction

def intention_label_worker(sdc_xy, yaw, scenario_id, path):
    max_curvature, sign,length, diff,_ = get_length_curvature(sdc_xy, yaw)
    name = str(scenario_id)
    max_curvature = round(max_curvature,2)
    diff = round(diff,2)
    intention, direction = identifiy(max_curvature, sign, diff)
    with open(os.path.join(path,f'{name}.txt'), 'w') as f:
        context = intention+'_'+direction
        f.write(context)
    
def workers(# roadgraph
            roadgraph,
            ids,
            max_roadgraph_segments,
            road_path,
            # route
            route,
            max_route_segments,
            ego_car_width,
            route_path,
            # intention label
            sdc_xy,
            yaw,
            scenario_id,
            intention_lable_path):
    worker_roadgraph(roadgraph,ids,max_roadgraph_segments,road_path)
    worker_route(route,max_route_segments,ego_car_width,route_path)
    intention_label_worker(sdc_xy, yaw, scenario_id, intention_lable_path)

def padding(data,max_len,pad_value=0):
    padding_ = np.ones((1,6))* pad_value
    if len(data) < max_len:
        data += [padding_ for _ in range(max_len - len(data))]
    else:
        data = data[:max_len]
    data = np.array(data).reshape(-1,6) 
    return data

def split_large_BB(route, start_id):
    x = route[0]
    y = route[1]
    angle = route[4]
    extent_x = route[2] / 2
    extent_y = route[3] / 2

    y1 = y + extent_y * math.sin(math.radians(angle))
    y0 = y - extent_y * math.sin(math.radians(angle))

    x1 = x + extent_y * math.cos(math.radians(angle))
    x0 = x - extent_y * math.cos(math.radians(angle))

    number_of_points = (
        math.ceil(extent_y * 2 / 10) - 1
    )  # 5 is the minimum distance between two points, we want to have math.ceil(extent_y / 5) and that minus 1 points
    xs = np.linspace(
        x0, x1, number_of_points + 2
    )  # +2 because we want to have the first and last point
    ys = np.linspace(y0, y1, number_of_points + 2)
    # splitted_routes = [[x0,y0,extent_x,extent_y,route[4],0],
    #                    [x1,y1,extent_x,extent_y,route[4],0]]
    splitted_routes = []
    for i in range(len(xs) - 1):
        route_new = route.copy()
        route_new[0] = (xs[i] + xs[i + 1]) / 2
        route_new[1] = (ys[i] + ys[i + 1]) / 2
        route_new[5] = float(start_id + i)
        route_new[2] = extent_x * 2
        route_new[3] = route[3] / (
            number_of_points + 1
        )
        splitted_routes.append(np.array(route_new).reshape(-1,6))

    return splitted_routes


def rdp_downsample_route(routes,ego_car_width,max_length_per_seg=10):
    # route should be like [N,2], there're only one route and N is the points or timestampes on the route
    # this is operated under ego car coordination
    max_route_distance = 50
    # ori_polyline = obs[...,:2]
    # bs_routes = []
    assert routes.shape[0]==1
    for bs,route in enumerate(routes):
        try:
            route = np.round(route,2)
            shortened_route = np.array(crdp.rdp(route,0.5))
        except Exception as e:
            print(route,route.shape)
            print(e)
        # convert points to vectors
        vectors = shortened_route[1:] - shortened_route[:-1]
        midpoints = shortened_route[:-1] + vectors/2.
        norms = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:,1], vectors[:,0])
        # x y w h yaw id
        route_segments = []
        for i, midpoint in enumerate(midpoints):
            # distance is the distance between ego car to the midpoint of a segment
            distance = np.linalg.norm(midpoint)
            # st_distance is the distance between ego car to the start point of a segment
            st_distance = np.linalg.norm(shortened_route[i])
            # only store route boxes that are near the ego vehicle
            # if st_distance > max_route_distance:
            #     continue
            x,y = midpoint[0],midpoint[1]
            w,h = ego_car_width, norms[i]
            yaw = angles[i] * 180 / np.pi
            id = i
            segments = [x,y,w,h,yaw,id]
            # 10 is the length of segment splitted into several segments
            if h > max_length_per_seg:
                # splitted = []
                splitted = split_large_BB(segments,id)
                route_segments.extend(splitted)
            else:
                route_segments.append(np.array(segments).reshape(-1,6))
        # bs_routes.append(route_segments)
    return route_segments

def get_roadgraph_obs_(road_observation,ids):
    TYPES = [MapElementIds.ROAD_EDGE_BOUNDARY,MapElementIds.ROAD_EDGE_MEDIAN]
    unique_ids = np.unique(ids)
    road_observation = np.array(road_observation)
    ids = np.array(ids)
    splitted_array = []
    for i in range(len(unique_ids)-1):
       road_ob = road_observation[ids.reshape(-1) == unique_ids[i]]
       if len(road_ob) > 2 and road_ob[0,-1] in TYPES:
              splitted_array.extend(rdp_downsample_route(np.array(road_ob[np.newaxis,:,0:2]),float(0.5),max_length_per_seg=10))   
    return splitted_array


@jax.jit
def get_whole_map(state):
    # print(state.roadgraph_points.shape)
    _,B,P = state.roadgraph_points.shape
    road_observation = jnp.concatenate(
                        (state.roadgraph_points.xy.reshape(B,P,-1),
                            # state.roadgraph_points.dir_xy[valid][mask],
                            state.roadgraph_points.types.reshape(B,P,1),
                            # state.roadgraph_points.ids.reshape(-1,1)
                            ),axis=-1)
    ids = state.roadgraph_points.ids.reshape(B,P,-1)
    return road_observation,ids

@jax.jit
def get_route_global(state):
    _,B,P = state.roadgraph_points.shape
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    routes = state.log_trajectory.xy[:,jnp.linspace(0,B-1,B).astype(jnp.int32),sdc_idx.reshape(-1),...]
    ego_car_width = state.log_trajectory.width[:,jnp.linspace(0,B-1,B).astype(jnp.int32),sdc_idx.reshape(-1),...][-1,-1,-1]
    return routes.reshape(B,routes.shape[-2],routes.shape[-1]),\
           ego_car_width