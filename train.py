from pytorch_lightning import Trainer
import torch
torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from src.policy import build_model
from src.dataloader.dataset import WaymoDataLoader
from src.utils.utils import set_seed
from pytorch_lightning.callbacks import ModelCheckpoint  
import hydra
from omegaconf import OmegaConf
import random

@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg):
    if cfg.seed is not None:
        print(f'Setting seed to {cfg.seed}')
    elif cfg.seed is None:
        cfg.seed = random.randint(0, 100000)
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    model = build_model(cfg)

    train_set = WaymoDataLoader(cfg)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices), 1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        # monitor='val/brier_fde',  # Replace with your validation metric
        filename='latest',
        save_top_k=1,
        # mode='min',  # 'min' for loss/error, 'max' for accuracy
    )
    # checkpoint_callback.FILE_EXTENSION = ".pth.tar"
    call_backs.append(checkpoint_callback)
    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,shuffle=True, pin_memory=True)

    trainer = Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None if cfg.debug else TensorBoardLogger(save_dir="planner", name=cfg.exp_name, version=cfg.version),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        # strategy="auto" if cfg.debug else "ddp_find_unused_parameters_true",
        strategy = cfg.method.strategy,
        callbacks=call_backs,
        precision=16
        )
    trainer.logger.log_hyperparams(cfg,{'metric': 1.0})
    if cfg.find_lr == True:
        from pytorch_lightning.tuner import Tuner
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model,train_loader,min_lr=1e-8,max_lr=1e-2)
        print(lr_finder.results)
        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.savefig('find_lr_plot.png')
        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print(f'suggestion_lr is {new_lr}')
    else:
        trainer.fit(model=model, train_dataloaders=train_loader,ckpt_path=cfg.ckpt_path)
    

if __name__ == '__main__':
    train()