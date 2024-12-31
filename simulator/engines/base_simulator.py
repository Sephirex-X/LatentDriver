import time
from simulator.waymo_env import WaymoEnv
import torch
import os
from src.utils.viz import plot_seq2image, plot_image
import matplotlib.pyplot as plt
import tqdm
import mediapy
class BaseSimulator():
    def __init__(self,
                 model,
                 config,
                 batch_dims,
                 ):
        self.env = WaymoEnv(
            waymax_conf=config.waymax_conf,
            env_conf=config.env_conf,
            batch_dims=batch_dims,
            ego_control_setting=config.ego_control_setting,
            metric_conf=config.metric_conf,
            data_conf=config.data_conf,
        )
        self.batch_dims = batch_dims
        self.size = batch_dims[0] * batch_dims[1]
        self.model = model
        self.model.eval()
        # put the policy model on the last available gpu
        if torch.cuda.is_available():
            self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        self.model.to(self.device)        
        
    def run(self):
        self.idx = 0
        while True:
            try:
                episode_list = []
                preds = []
                info_list = []
                obs, obs_dict = self.env.reset()
                obs = obs.reshape(self.env.num_envs,-1,7)
                # format_data(obs_dict)
                done_ = False
                self.T = 0
                a = time.time()
                diff = 0
                while not done_:
                    actions = self.model.forward(obs, deterministic=True, clip_action=True)
                    obs, obs_dict,rew, done, info = self.env.step(actions,show_global=True)
                    obs = obs.reshape(self.env.num_envs,-1,7)

                    self.T+=1
                    done_ =done[-1]

                self.idx += 1
                print('Processed: ', self.idx, 'th batch, Time: ', time.time()-a, 's')
                    
            except StopIteration:
                print("StopIteration")
                break
            
            
    def render(self, vis, model_name):
        def save_image():
            img = plot_seq2image(state=self.env.states[-1],batch_idx=j)
            img.savefig(name+'.pdf',dpi=300,format='pdf')
            plt.close(img)
            print('Saved Image: ', name)
        def save_video():
            # for full video
            imgs = []
            margin = 30
            vis_config = dict(
                front_x=margin,
                back_x=margin,
                front_y=margin,
                back_y=margin,
                px_per_meter=20,
                show_agent_id=True,
                center_agent_idx=-1,
                verbose=False
            )
            for state in tqdm(self.env.states):
                imgs.append(plot_image(state = state,batch_idx=j,viz_config=vis_config))
                mediapy.write_video(name+'.mp4',imgs , fps=10)
            print('Saved Video: ', name)
            
        if vis==False:
            return
        assert vis in ['image','video'], "vis must be either 'image' or 'video'"
        root_folder = f'vis_results/{model_name}/{vis}/'
        os.makedirs(root_folder, exist_ok=True)
        os.makedirs(root_folder+'straight_', exist_ok=True)
        os.makedirs(root_folder+'turning_left', exist_ok=True)
        os.makedirs(root_folder+'turning_right', exist_ok=True)
        os.makedirs(root_folder+'U-turn_left', exist_ok=True)
        for j in range(self.size):
            intention = self.env.intention_label[j]
            if intention == 'straight_':
                # sub_folder = root_folder+'straight_'
                sub_folder = None
            elif intention == 'turning_left':
                sub_folder = root_folder+'turning_left'
            elif intention == 'turning_right':
                sub_folder = root_folder+'turning_right'
            elif intention == 'U-turn_left':
                sub_folder = root_folder+'U-turn_left'
            else: sub_folder = None
            if sub_folder is not None:
                scen_id = self.env.get_env_idx(j)
                is_offroad = self.env.metric.info_hack['metric/offroad_rate'].reshape(-1)[j]
                is_collision = self.env.metric.info_hack['metric/collision_rate'].reshape(-1)[j]
                is_ar90 = self.env.metric.info_hack['metric/arrival_rate90'].reshape(-1)[j]
                name = f'{sub_folder}/AR{is_ar90}_OR{is_offroad}_CR{is_collision}_{scen_id}'
                if vis == 'image':
                    save_image()
                elif vis == 'video':
                    save_video()