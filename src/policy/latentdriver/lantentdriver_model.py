import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.policy.commons.optimzier import build_optimizer
from src.policy.commons.scheduler import build_scheduler
from .loss.GMMloss import GMMloss, get_nearest_mode_idxs
from ..commons.enc import build_model as build_enc
from .world import build_model as build_world
from .world.latent_world_model import sample_from_distribution
from .loss.KLloss import KLLoss
from .transformers.mpa_decoder import TransformerDecoderLayer
from .transformers.utils import TrainableQueryProvider
import math
def unpack_action(action, B, T):
    # action: B*T, K, 7
    prob, out_model, yaw = action[...,0:1], action[...,1:6], action[...,6:]
    bs_slice = torch.arange(B * T)
    mode = prob.reshape(prob.shape[0],-1).argmax(dim=-1)
    action_ = out_model[bs_slice,mode[bs_slice],:2]
    action_ = torch.cat([action_,yaw[bs_slice,mode[bs_slice],...]],dim=-1)
    return prob, out_model, yaw, action_

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """
    Generate sine embeddings for positions.

    Parameters:
    pos_tensor (torch.Tensor): A tensor containing positions with shape (n_query, bs, 2) where the last dimension contains x, y.
    d_model (int): The dimensionality of the output embeddings (default is 256).

    Returns:
    torch.Tensor: A tensor containing the sine embeddings with shape (n_query, bs, d_model).
    """
    assert pos_tensor.size(-1) in [2, 4]
    scale = 2 * math.pi
    dim_t = torch.arange(hidden_dim // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (hidden_dim // 2))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def apply_cross_attention(kv_feature, kv_mask, kv_pos, query_content, query_embed, attention_layer,
                              dynamic_query_center=None, layer_idx=0, use_local_attn=False, query_index_pair=None,
                              query_content_pre_mlp=None, query_embed_pre_mlp=None):
        """
        Args:
            kv_feature (B, N, C):
            kv_mask (B, N):
            kv_pos (B, N, 3):
            query_tgt (M, B, C):
            query_embed (M, B, C):
            dynamic_query_center (M, B, 2): . Defaults to None.
            attention_layer (layer):

            query_index_pair (B, M, K)

        Returns:
            attended_features: (B, M, C)
            attn_weights:
        """
        if query_content_pre_mlp is not None:
            query_content = query_content_pre_mlp(query_content)
        if query_embed_pre_mlp is not None:
            query_embed = query_embed_pre_mlp(query_embed)

        num_q, batch_size, d_model = query_content.shape
        searching_query = dynamic_query_center
        # searching_query = gen_sineembed_for_position(dynamic_query_center, hidden_dim=d_model)
        kv_pos = kv_pos.permute(1, 0, 2)[:, :, 0:2]
        kv_pos_embed = gen_sineembed_for_position(kv_pos, hidden_dim=d_model)


        query_feature = attention_layer(
            tgt=query_content,
            query_pos=query_embed,
            query_sine_embed=searching_query,
            memory=kv_feature.permute(1, 0, 2),
            memory_key_padding_mask=~kv_mask,
            pos=kv_pos_embed,
            is_first=(layer_idx == 0)
        )  # (M, B, C)

        return query_feature

class GMMHead(nn.Module):
    def __init__(
        self,
        d_model,
        **kwargs
    ):
        super().__init__()
        self.prob_predictor = nn.Linear(d_model, 1)
        self.output_model = nn.Linear(d_model, 5)
        # # yaw is depended on xy
        self.out_yaw = nn.Linear(d_model, 1)
    def forward(self, x):
        '''
            return torch[prob, out_model, yaw]
        '''
        # prob: bs, mode, 1
        prob = self.prob_predictor(x)
        # model: bs, mode, 5
        out_model = self.output_model(x)
        # yaw bs, mode, 1
        yaw = self.out_yaw(x)
        return torch.concat([prob, out_model, yaw], dim=-1)

class MPA_blocks(nn.Module):
    def __init__(self, 
                 hidden_size,
                 num_cross_attention_heads,
                 **kwargs):
        super().__init__()
        self.decoder_layer = TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_cross_attention_heads, dim_feedforward=hidden_size * 4, dropout=0.1, normalize_before=False, keep_query_pos=False,
            rm_self_attn_decoder=False, use_local_attn=False
        )
        self.gmmhead = GMMHead(hidden_size)
    def forward(self, latent, q_content, q_emb, layer_idx, action_embedding):
        B,T,M,_ = latent.shape
        latent = latent.view(B*T,M,-1)
        query_content = apply_cross_attention(
                kv_feature=latent,
                kv_mask=torch.ones((B*T,M), device=latent.device, dtype=torch.bool),
                kv_pos=torch.zeros((B*T,M,3),device=latent.device, dtype=torch.float32),
                query_content=q_content,
                query_embed=q_emb, #only vaild for first layer
                attention_layer=self.decoder_layer,
                dynamic_query_center = action_embedding.permute(1,0,2),
                layer_idx=layer_idx,
            )
        action_dis = self.gmmhead(query_content.permute(1,0,2))
        return action_dis, query_content

class LantentDriver(pl.LightningModule):
    def __init__(
        self,
        max_length=None,
        eval_context_length=None,
        pretrain_enc = None,
        freeze_enc = False,
        encoder = None,
        pretrain_world = None,
        freeze_world = False,
        world = None,
        function = 'enc-LWM-MPP',
        mode = 6,
        num_of_decoder = 3,
        num_cross_attention_heads=4,
        est_layer = 0,
        **kwargs,
    ):
        super().__init__()
        self.est_layer = est_layer
        self.function = function
        self.act_dim = world.act_dim
        self.max_length = max_length
        self.hidden_size = world.hidden_size
        self.eval_context_length = eval_context_length
        self.ordering = world.ordering
        self.init_enc(encoder, pretrain_enc, freeze_enc)
        assert self.function in ['enc-LWM-MPP', 'enc-MPP']
        if self.function == 'enc-LWM-MPP':
            print("Initating world")
            self.init_world(world,pretrain_world, freeze_world)
        # MPAD
        hidden_size = world.hidden_size
        self.action_prob_emb = nn.Linear(7, hidden_size)
        self.query_pe = TrainableQueryProvider(num_queries=mode, num_query_channels=hidden_size, init_scale=0.01)
        self.action_distribution_queries = TrainableQueryProvider(num_queries=mode, num_query_channels=hidden_size, init_scale=0.01)
        self.mpad_blocks = nn.ModuleList([MPA_blocks(hidden_size=hidden_size, num_cross_attention_heads=num_cross_attention_heads) for _ in range(num_of_decoder)])
        
        #         self.imagine_query = None
        self.optim_conf = kwargs['optimizer']
        self.sched_conf = kwargs['scheduler']
        self.lr = kwargs['learning_rate']
        
    def init_world(self, world_conf, pretrain_world, freeze_world):
        self.world_model = build_world(world_conf)
        if pretrain_world is not None:
            self.world_model.load_state_dict(torch.load(pretrain_world))
            print(f'Loaded pretained world from {pretrain_world}')
        if freeze_world:
            for para in self.world_model.parameters():
                para.requires_grad = False
    
    def init_enc(self, bert_conf, pretrain_enc, freeze_enc):
        self.bert = build_enc(bert_conf)
        if pretrain_enc is not None:
            self.bert.load_state_dict(torch.load(pretrain_enc))
            print(f'Loaded pretained enc from {pretrain_enc}')
        if freeze_enc:
            for para in self.bert.parameters():
                para.requires_grad = False
    
    def forward(
        self,
        states, # bs , seq_len, state_attributes, state_dim
        actions = None,# bs , seq_len, act_dim
        timesteps = None, # bs, seq_len e.g. [1,2,3 ... 20]
        padding_mask=None, # bs , seq_len
    ):
        batch_size, seq_length, state_elements, state_dims = states.shape[0], states.shape[1], states.shape[2], states.shape[3]
        bert_embeddings = self.bert(states.reshape(batch_size*seq_length,state_elements,state_dims),return_full_length=True).reshape(batch_size,seq_length,-1,self.bert.hidden_size)
        
        actions_layers = []
        cur_latent_token = bert_embeddings[:,:,0:1,:]
        B,T,_,_ = bert_embeddings.shape
        query_pe = self.query_pe(None).repeat(B*T,1,1).permute(1,0,2)
        query_content = torch.zeros_like(query_pe)
        action_dis = None
        fut_latent_dis = None
        latent_dist = None
        rep_dist = None
        for i in range(len(self.mpad_blocks)):
            if fut_latent_dis is not None:
                # cat imagine goal and current observation
                latent = torch.concat([fut_latent_dis, cur_latent_token],dim=2)
            else:
                latent = cur_latent_token
            # if no action_dis
            if action_dis == None:
                action_embedding = self.action_distribution_queries(None).repeat(B*T,1,1)
            else:
                action_embedding = self.action_prob_emb(action_dis.reshape(B*T,-1,7))
            
            action_dis, query_content = self.mpad_blocks[i](latent = latent,
                                                            q_content = query_content,
                                                            q_emb = query_pe,
                                                            layer_idx = i,
                                                            action_embedding = action_embedding
                                                            )
            if i == self.est_layer:
                # forward with world
                _,_,_,action_layer_0 = unpack_action(action_dis, B=B, T=T)
                action_layer_0 = action_layer_0.reshape(B,T,-1)
                with torch.no_grad():
                    if self.training:
                        his_actions = action_layer_0.detach()
                    else:
                        # replace the first action with the predicted action
                        action_latest = action_layer_0.reshape(B, T,-1)[:,-1:,:]
                        his_actions = torch.concatenate([actions, action_latest],dim=1)
                        his_actions = his_actions[:,1:,:]
                if self.function == 'enc-LWM-MPP':
                    rep_dist, latent_dist  = self.world_model(bert_embeddings, his_actions, timesteps, padding_mask)
                    fut_latent_dis = sample_from_distribution(latent_dist[0],latent_dist[1],not self.training)
                else:
                    latent_dist = None
                    rep_dist = None
                    fut_latent_dis = None
                    
            actions_layers.append(action_dis.reshape(B,T,-1,7).clone())
            
        return actions_layers,latent_dist, rep_dist
        
    def get_predictions(
        self, states, actions, timesteps, num_envs=1, **kwargs
    ):
        states_elements = states.shape[2]

        # max_length is the context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            actions = actions[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            states_elements,
                            states.shape[-1],
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)

            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

            ordering = torch.cat(
                [
                    torch.zeros(
                        (ordering.shape[0], self.max_length - ordering.shape[1]),
                        device=ordering.device,
                    ),
                    ordering,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            padding_mask = None
        action, _, _ = self.forward(
            states,
            actions,
            timesteps,
            padding_mask=padding_mask,
        )
        B, T = states.shape[0], states.shape[1]
        if self.function in ['enc-LWM-MPP','enc-MPP']:
            _, _, _, action_ = unpack_action(action[-1].reshape(B*T,-1,7), B, T)
            action = action_.reshape(B, T, -1)
        return action[:,-1,:]

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
        B, T, _, _ = states.shape

        action_preds,rep_dist, latent_dist = self.forward(
            states,
            actions, #no used here for training
            timesteps,
            padding_mask=padding_mask,
        )
        # loss for lwm
        if self.function == 'enc-LWM-MPP':
            klloss= KLLoss()
            posterior = {'mu': rep_dist[0][:,1:,...],'sigma':rep_dist[1][:,1:,...]}
            prior = {'mu': latent_dist[0][:,:-1,...], 'sigma':latent_dist[1][:,:-1,...] }
            loss_rec = klloss.forward(prior,posterior) * 1e-3
        else:
            loss_rec = 0
        
        loss_all = 0
        logs = {}
        # loss for mpp
        for i in range(len(action_preds)):
            prob, out_model, yaw, action_ = unpack_action(action_preds[i].reshape(B*T,-1,7), B, T)
            with torch.no_grad():
                nearest_mode_idx, iou = get_nearest_mode_idxs(out_model.unsqueeze(2),action_target.reshape(B*T,1,-1),yaw,type='IoU')
                        # reweight
            w_dy = 1.0
            action_target_gmm = action_target.clone()
            action_target_gmm[...,1] *= w_dy
            # # mu_y, sigma_y
            out_model[...,1] *= w_dy
            out_model[...,3] *= w_dy
            loss_dict = GMMloss(out_model.unsqueeze(2), 
                                prob.squeeze(-1),
                                action_target_gmm.reshape(B*T,1,-1), 
                                yaw, 
                                nearest_mode_idx,
                                label=iou)
            loss = (loss_dict['loss_gmm'] + loss_dict['loss_cls'] + loss_dict['loss_yaw']).mean()
            loss_all += loss
            with torch.no_grad():
                loss_dx = nn.L1Loss()(action_[...,0].reshape(-1),action_target[...,0].reshape(-1))
                loss_dy = nn.L1Loss()(action_[...,1].reshape(-1),action_target[...,1].reshape(-1))
                loss_dyaw = nn.L1Loss()(action_[...,2].reshape(-1),action_target[...,2].reshape(-1))
                
            logs.update({
                    f'loss_{i}/all': loss,
                    f'loss_{i}/loss_gmm': loss_dict['loss_gmm'],
                    f'loss_{i}/loss_cls': loss_dict['loss_cls'],
                    f'loss_{i}/loss_gmm_yaw': loss_dict['loss_yaw'],
                    f'loss_{i}/loss_dx':loss_dx,
                    f'loss_{i}/loss_dy':loss_dy,
                    f'loss_{i}/loss_dyaw':loss_dyaw
                    })
            
        loss_all /= len(action_preds)
        loss_all += loss_rec
        logs.update({
            'Lr/lr': self.optimizers().param_groups[0]['lr'],
            'loss/all': loss_all,
            'loss/loss_gmm': loss_dict['loss_gmm'],
            'loss/loss_cls': loss_dict['loss_cls'],
            'loss/loss_gmm_yaw': loss_dict['loss_yaw'],
            'loss/loss_dx':loss_dx,
            'loss/loss_dy':loss_dy,
            'loss/loss_dyaw':loss_dyaw,
            'loss/loss_rec':loss_rec
            })
        self.log_dict(logs,on_step=True)
        return loss_all
        
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