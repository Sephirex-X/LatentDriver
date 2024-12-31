import torch
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import os
def tree_stack(trees):
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)

def tree_unstack(tree):
    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]

class WaymoScenLoader(object):
    def __init__(self, config):
        batch_size, dir = config.batch_dims, config.path
        distributed=True
        self.dir = dir
        if isinstance(dir,list):   
            self.scen = [torch.load(dir[i]) for i in range(len(dir))]
        elif isinstance(dir,str):
            self.scen = [torch.load(os.path.join(dir, name)) for name in os.listdir(dir)]
        else:
            raise NotImplementedError('The dir should be a list or str')  
        self.distributed = distributed 
        if distributed:
            assert len(batch_size) == 2
            scen_ = []
            self.num_device = batch_size[0]
            self.batch_size = batch_size[1]
            assert len(self.scen) % self.num_device == 0
            # The number of scenario should be divisible by num_device
            #  [1,2,3,4,5,6,7,8] -> [[1,2], [3,4], [5,6], [7,8]] num device is 4
            split_index = np.linspace(0,len(self.scen),self.num_device+1).astype(int)
            for i in range(len(split_index)-1):
                scen_.append(self.scen[split_index[i]:split_index[i+1]])
            
            self.scen = scen_

        else:
            self.batch_size = batch_size

    def __next__(self):
        if self.distributed:
            stacked = []
            # for per device
            # if scen for per device is less than batch_size, then repeat it
            for scen in self.scen:
                stacked.append(tree_stack(scen))
            # stacked: [scen_split_1, scen_split_2, ...]
            stacked = tree_stack(stacked)
        else:
            # from size () into (bs,)
            stacked = tree_stack(self.scen * (self.batch_size // len(self.scen)) )
            # print(stacked._scenario_id)
            # from (bs,) into (1, bs) than same as waymo dataloader
            stacked = tree_stack([stacked])
        return stacked
    def __len__(self):
        return self.batch_size
    @property
    def dir_name(self):
        return self.dir

if __name__ == "__main__":
    scen_dir = 'scenerio_data_temp_2'
    loader = WaymoScenLoader(55, scen_dir)
    a = next(loader)
    print(a._scenario_id.reshape(-1))
    print(a.shape, type(a))