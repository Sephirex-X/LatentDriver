import os
import torch
import pickle
import numpy as np
import random
import pytorch_lightning as pl

def is_shorter(start, end , thre):
    dis = np.linalg.norm(end-start)
    # skip if the distance is too short
    if dis <= thre:
        return True
    else:
        return False
                        
def update_waymax_config(config):
    from waymax import config as _config
    env_conf = dict(
        max_num_objects=config.waymax_conf.max_num_objects,
        controlled_object=_config.ObjectType.SDC if config.ego_control_setting.npc_policy_type=='expert' else _config.ObjectType.VALID,
        compute_reward = True,
        rewards = _config.LinearCombinationRewardConfig(rewards=config.pop('rewards')))
    config.update(dict(env_conf=env_conf))
    config.waymax_conf.update({'batch_dims': config.batch_dims})
    return config

def save_ckpt(obj,path,name):
    os.makedirs(os.path.join(path,'ckpt'),exist_ok=True)
    save_name = os.path.join(path,'ckpt',name)+'.pth.tar'
    torch.save(obj,save_name)

def saving_data(data,name:str,mode='pkl'):
    if mode == 'pkl':
        with open(name+'.pkl', 'wb') as f:
            pickle.dump(data, f)
            # print('Saved: {}'.format(name+'.pkl'))
    elif mode == 'np': 
        with open(name+'.npy', "wb") as f:
            np.save(f, arr=data)
    else:
        raise NotImplementedError

def loading_data(name:str,mode='pkl'):
    if mode == 'pkl':
        if name.endswith('.pkl'):
            name = name
        else:
            name = name+'.pkl'
        with open(name, 'rb') as f:
            # print('Loading: {}'.format(name+'.pkl'))
            return pickle.load(f)
    elif mode == 'np': 
        if name.endswith('.npy'):
            name = name
        else:
            name = name+'.npy'
        with open(name, "rb") as f:
            # print('Loading: {}'.format(name))
            return np.load(f,allow_pickle=True)
    else:
        raise NotImplementedError

def set_seed(seed_value=42):
    """
    Set seed for reproducibility in PyTorch Lightning based training.

    Args:
    seed_value (int): The seed value to be set for random number generators.
    """
    # Set the random seed for PyTorch
    torch.manual_seed(seed_value)

    # If using CUDA (PyTorch with GPU)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU

    # Set the random seed for numpy (if using numpy in the project)
    np.random.seed(seed_value)

    # Set the random seed for Python's `random`
    random.seed(seed_value)

    # Set the seed for PyTorch Lightning's internal operations
    pl.seed_everything(seed_value, workers=True)