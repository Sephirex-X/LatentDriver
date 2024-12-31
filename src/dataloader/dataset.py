from torch.utils.data import Dataset
from src.utils.utils import loading_data
import os
import torch
import random
import numpy as np
from src.dataloader.normalizer import Normalizer
MAX_EPISODE_LEN = 80
def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret
class TransformSamplingSubTraj:
    def __init__(
        self,
        max_len,
        act_key,
        reward_scale,
        action_range,
    ):
        super().__init__()
        self.max_len = max_len
        self.state_dim = 7
        self.reward_scale = reward_scale
        self.normalizer = Normalizer(action_range)
        # the user defined action range.
        self.action_range = action_range
        if act_key == 'bicycle':
            self.act_key = 'bicycle_actions'
            self.act_dim = 2
        elif act_key == 'waypoint':
            self.act_key = 'waypoints_actions'
            self.act_dim = 3
    def __call__(self, traj, si):
        # si = random.randint(0, traj["rewards"].shape[0] - 1 - self.max_len)
        # si = 1
        # get sequences from dataset
        ss = traj["obs"][si : si + self.max_len].reshape(self.max_len, -1, self.state_dim)
        aa = traj[self.act_key][si : si + self.max_len].reshape(self.max_len, self.act_dim)
        try:
            aa_gt = traj[self.act_key][si+1 : si+ 1 + self.max_len].reshape(self.max_len, self.act_dim).astype(np.float32)
        except:
            print(traj[self.act_key][si+1 : si+ 1 + self.max_len].shape, si)
            raise ValueError
        
        rr = traj["rewards"][si : si + self.max_len].reshape(self.max_len, 1)

        if "terminals" in traj:
            dd = traj["terminals"][si : si + self.max_len]  # .reshape(-1)
        else:
            dd = traj["dones"][si : si + self.max_len]  # .reshape(-1)

        # get the total length of a trajectory
        tlen = ss.shape[0]

        timesteps = np.arange(si, si + tlen)  # .reshape(-1)
        ordering = np.arange(tlen)
        ordering[timesteps >= MAX_EPISODE_LEN] = -1
        ordering[ordering == -1] = ordering.max()
        timesteps[timesteps >= MAX_EPISODE_LEN] = MAX_EPISODE_LEN - 1  # padding cutoff
        # reward to go, not used in this case
        rtg = discount_cumsum(traj["rewards"][si:], gamma=1.0)[: tlen + 1].reshape(
            -1, 1
        )
        if rtg.shape[0] <= tlen:
            rtg = np.concatenate([rtg, np.zeros((1, 1))])
            
        # padding and state + reward normalization
        act_len = aa.shape[0]
        if tlen != act_len:
            print(ss.shape, aa.shape)
            raise ValueError

        ss = np.concatenate([np.zeros((self.max_len - tlen, ss.shape[1],self.state_dim)), ss], dtype=np.float32)

        aa = np.concatenate([np.zeros((self.max_len - tlen, self.act_dim)), aa], dtype=np.float32)
        rr = np.concatenate([np.zeros((self.max_len - tlen, 1)), rr], dtype=np.float32)
        dd = np.concatenate([np.ones((self.max_len - tlen)) * 2, dd])
        rtg = (
            np.concatenate([np.zeros((self.max_len - tlen, 1)), rtg], dtype=np.float32)
            * self.reward_scale
        )
        timesteps = np.concatenate([np.zeros((self.max_len - tlen)), timesteps]).astype(np.int32)
        ordering = np.concatenate([np.zeros((self.max_len - tlen)), ordering])
        padding_mask = np.concatenate([np.zeros(self.max_len - tlen), np.ones(tlen)])
        aa_gt_normal = torch.from_numpy(self.normalizer.normalize(aa_gt)).clamp(min=-1.0, max=1.0)
        
        return ss, aa, aa_gt, aa_gt_normal,rr, dd, rtg, timesteps, ordering, padding_mask
        # return dict(
        #     states=ss,
        #     actions=aa,
        #     actions_gt = aa_gt,
        #     rewards=rr,
        #     dones=dd,
        #     rtg=rtg,
        #     timesteps=timesteps,
        #     ordering=ordering,
        #     padding_mask=padding_mask,
        # )
    
class WaymoDataLoader(Dataset):
    def __init__(self,config) -> None:
        self.dir = config.data_path
        self.full_name_list = []
        with open(os.path.join(self.dir,'name.txt')) as f:
            for name in f.readlines():
                self.full_name_list.append(name.strip())
        if config.mini == True:
            random.shuffle(self.full_name_list)
            self.full_name_list = self.full_name_list[:int(len(self.full_name_list)*0.1)]
            print("Using random 0.1 for trainig")
        start_t_idx = np.arange(0, MAX_EPISODE_LEN - config.max_len + 1).tolist()
        if config.overlap_sample==True:
            raise NotImplementedError('Not implemented yet for overlap_sample')
        else:
            # e.g. max_len is 10, MAX_EPISODE_LEN is 80
            # start_t_idx = [0 ... 70] -> [0, 10, 20, 30, 40, 50, 60, 70], every data has no overlap
            start_t_idx = start_t_idx[::config.max_len]
        aug_list = []
        print('Preparing split...')
        for name in self.full_name_list:
            for start_idx in start_t_idx:
                aug_list.append(f"{name}-{start_idx}")
        self.full_name_list = aug_list.copy()
        del aug_list
        self.transform = TransformSamplingSubTraj(
            max_len=config.max_len,
            act_key=config.action_space.dynamic_type,
            reward_scale=1,
            action_range=config.action_space.action_ranges
        )
    def __len__(self):
        return len(self.full_name_list)
    def __getitem__(self, index):
        name, si = self.full_name_list[index].split('-')[0],self.full_name_list[index].split('-')[1]
        traj = loading_data(os.path.join(self.dir,'data',name))
        return self.transform(traj,int(si))
        # return state,action

import hydra
@hydra.main(version_base=None, config_path="../configs", config_name="train")
def debug(cfg):
    from omegaconf import OmegaConf
    from tqdm import tqdm
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    loader = WaymoDataLoader(config=cfg)   
    for idx in tqdm(range(len(loader))):
        # if loader.full_name_list[idx] == '618217707':
        ss, aa, aa_gt, rr, dd, rtg, timesteps, ordering, padding_mask = loader.__getitem__(idx)

if __name__ == '__main__':
    debug()