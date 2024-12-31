import src.utils.init_default_jax
import hydra
from omegaconf import OmegaConf
from simulator.engines.ltd_simulator import LTDSimulator
from src.policy import build_model
from src.utils.utils import update_waymax_config
import torch
@hydra.main(version_base=None, config_path="configs", config_name="simulate")
def simulate(cfg):
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    model = build_model(cfg)
    cfg = update_waymax_config(cfg)
    if cfg.ckpt_path is not None:
        model.load_state_dict(torch.load(cfg.ckpt_path,map_location='cuda:{}'.format(torch.cuda.device_count()-1))['state_dict'])
        print(f'Loaded {cfg.ckpt_path}')
    else:
        print('No ckpt provided')
    runner = LTDSimulator(
        model = model,
        config=cfg,
        batch_dims=cfg.batch_dims,
    )
    eval_rtg = 0
    ep_return = [eval_rtg] * runner.env.num_envs
    runner.run(ep_return=ep_return, vis =cfg.vis)
if __name__ == '__main__':
    simulate()