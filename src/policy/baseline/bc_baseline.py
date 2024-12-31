from torch import nn
from src.policy.baseline.mlp import MLP
from src.utils.discretizer import Discretizer
import pytorch_lightning as pl
from src.policy.commons.optimzier import build_optimizer
from src.policy.commons.scheduler import build_scheduler
from ..commons.enc import build_model as build_enc
class Simple_driver(pl.LightningModule):
    def __init__(self,
                 action_space,
                 hidden_channels:list[int] = [64,64],
                 control_type:str = 'bicycle',
                 discretizer:Discretizer = None,
                 optimizer = None,
                 encoder = None,
                 scheduler = None,
                 **kwarg,
                 ):
        super().__init__()
        self.control_type = action_space.dynamic_type
        self.discretizer = discretizer
        if control_type == 'bicycle':
            out_dim = 2 if discretizer is None else discretizer._max_discrete_idx+1
        elif control_type == 'waypoint':
            out_dim = 3 if discretizer is None else discretizer._max_discrete_idx+1
        self.bert = build_enc(encoder)
        self.optim_conf = optimizer
        self.sched_conf = scheduler
        self.lr = kwarg['learning_rate']
        
        self.fc_head =MLP( 
            in_channels=self.bert.hidden_size,
            hidden_channels=hidden_channels[0],
            out_channels=out_dim,
            layer_num=len(hidden_channels),
            activation=nn.ReLU(),
            norm_type='BN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=True,) 
            
        self.out_dim = out_dim
    def forward(
        self,
        states, # bs , seq_len, state_attributes, state_dim
    ):
        batch_size, seq_length, state_elements, state_dims = states.shape[0], states.shape[1], states.shape[2], states.shape[3]
        x = states.reshape(batch_size*seq_length,state_elements,state_dims)
        fea = self.bert(x)

        if self.discretizer is not None:
            out_logist = self.fc_head(fea)
            out_softmax = nn.functional.softmax(out_logist, dim=1)
            index = out_softmax.argmax(dim=1).cpu().numpy()
            out = self.discretizer.make_continuous(index.reshape(-1, 1))
        else:
            out = self.fc_head(fea)

        out = out.reshape(batch_size,seq_length,self.out_dim)
        return out
    
    def configure_optimizers(self):
        self.optim_conf.update(dict(lr = self.lr))
        optimizer = build_optimizer(self.optim_conf, self)
        all_steps = self.trainer.estimated_stepping_batches
        print(f'All step is {all_steps}')
        if self.sched_conf.type == 'CosineAnnealingLR':
            self.sched_conf.update(dict(T_max=all_steps))
        elif self.sched_conf.type == 'LinearLR':
            self.sched_conf.update(dict(total_iters=all_steps))
        elif self.sched_conf.type == 'OneCycleLR':
            self.sched_conf.update(dict(total_steps=all_steps)) 
        elif self.sched_conf.type == 'ConstantLR':
            self.sched_conf.update(dict(total_iters=all_steps)) 
            
        scheduler = build_scheduler(self.sched_conf,optimizer)
        scheduler = {
          'scheduler': scheduler, # The LR scheduler instance (required)
          'interval': 'step', # The unit of the scheduler's step size
          'frequency': 1, # The frequency of the scheduler
        }
        return [optimizer], [scheduler]
    def training_step(self, batch, batch_idx):
        (
            states,
            actions,
            action_target,
            aa_gt_normal,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = batch
        # states,action_target = batch
        action_preds = self(states)
        loss_dx = nn.L1Loss()(action_preds[...,0].reshape(-1),action_target[...,0].reshape(-1))
        loss_dy = nn.L1Loss()(action_preds[...,1].reshape(-1),action_target[...,1].reshape(-1))
        loss_dyaw = nn.L1Loss()(action_preds[...,2].reshape(-1),action_target[...,2].reshape(-1))
        loss = 1 * loss_dx + 50 * loss_dy + 50* loss_dyaw #the weight here is alike easychauffeur
        logs = {
                'Lr/lr': self.optimizers().param_groups[0]['lr'],
                'loss/all': loss,
                'loss/loss_dx':loss_dx,
                'loss/loss_dy':loss_dy,
                'loss/loss_dyaw':loss_dyaw}
        self.log_dict(logs,on_step=True)
        return loss
    
    def get_predictions(self, states, actions, timesteps, num_envs=1, **kwargs):
        state = states[:,-1:]
        out = self.forward(state)
        return out[:,-1]

    