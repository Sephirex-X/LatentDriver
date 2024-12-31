import torch.nn as nn
import torch
import os
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModel,
)

def randomize_model(model):
    for module_ in model.named_modules(): 
        if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model
class BertEncoder(nn.Module):
    def __init__(self,
                 type_name='bert-mini',
                 online=False,
                 attributes = 6,
                 embedding_type = 'bicycle',
                 **kwargs,):
        super(BertEncoder, self).__init__()
        # debug: split bicycle or wpts by embedding_type, 5 if bicycle, 4 is wpts
        # we think bicycle needs sdc_mask but not padding, but wpts needs padding not sdc_mask
        # 2.1 changed for bicycle
        self.control_type = embedding_type
        # self.control_type = 'bicycle'
        embedding_type = 5
        # end debug
        # debug add sdc_mask
        self.embedding_type = embedding_type # routes, vehicles road and sdc, dont need to add paddings
        if online:
            config_bert = AutoConfig.from_pretrained(os.path.join('prajjwal1',type_name))  # load config from hugging face model
            self.model = AutoModel.from_config(config=config_bert)   
            
        else:
            self.model = AutoModel.from_pretrained(type_name)
            # model
            config_bert = AutoConfig.from_pretrained(os.path.join(type_name,'config.json'))
            # self.model = randomize_model(self.model)
            # self.model = AutoModel.from_config(config=config_bert)
        self.hidden_size =  config_bert.hidden_size 
        n_embd = config_bert.hidden_size        
        self.cls_emb = nn.Parameter(
            torch.randn(1, attributes+1)
        )# +1 beacuse of the type
        # token embedding
        self.tok_emb = nn.Linear(attributes, n_embd)
        # object type embedding
        self.obj_token = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, attributes))
                for _ in range(self.embedding_type)
            ]
        )
        self.obj_emb = nn.ModuleList(
            [nn.Linear(attributes, n_embd) for _ in range(self.embedding_type)]
        )
        self.drop = nn.Dropout(0.5)
    
    def forward(self, x, return_full_length=False):
        # if x.dim() == 4:
        #     x = x.reshape(-1,x.shape[-2],x.shape[-1])
        # obs.shape [bs,max_len_routes+max_num_vehicles, 1+6]
        # typeis for 1:routes, 2:vehicles 3:road_graph 4:trajs
        B,_,_ = x.shape 
        x = torch.cat([self.cls_emb.repeat(B,1,1), x],dim=1)
        input_batch_type = x[:, :, 0]  # car or map
        input_batch_data = x[:, :, 1:]
        car_mask = (input_batch_type == 2).unsqueeze(-1)
        road_graph_mask = (input_batch_type == 3).unsqueeze(-1)
        route_mask = (input_batch_type == 1).unsqueeze(-1)
        sdc_mask = (input_batch_type == 4).unsqueeze(-1)
        padding_mask = (input_batch_type == 0).unsqueeze(-1)
        # get other mask
        other_mask = torch.logical_not(torch.logical_or(torch.logical_or(torch.logical_or(torch.logical_or(route_mask, car_mask), road_graph_mask), sdc_mask), padding_mask))
        # other_mask = torch.logical_and(route_mask.logical_not(), car_mask.logical_not(), road_graph_mask.logical_not(),sdc_mask.logical_not(),padding_mask.logical_not())
        if self.control_type == 'bicycle':
            masks = [car_mask, route_mask, road_graph_mask,other_mask,sdc_mask]
        elif self.control_type == 'waypoint':
            masks = [car_mask, route_mask, road_graph_mask,other_mask,padding_mask]

        # get size of input
        (B, O, A) = (input_batch_data.shape)  # batch size, number of objects, number of attributes

        # embed tokens object wise (one object -> one token embedding)
        input_batch_data = rearrange(
            input_batch_data, "b objects attributes -> (b objects) attributes"
        )
        embedding = self.tok_emb(input_batch_data)
        embedding = rearrange(embedding, "(b o) features -> b o features", b=B, o=O)

        # create object type embedding
        obj_embeddings = [
            self.obj_emb[i](self.obj_token[i]) for i in range(self.embedding_type)
        ]  # list of a tensors of size 1 x features

        # add object type embedding to embedding (mask needed to only add to the correct tokens)
        embedding = [
            (embedding + obj_embeddings[i]) * masks[i] for i in range(self.embedding_type)
        ]
        # debug dropout sdc_obj_emb
        # embedding[-1] = self.drop(embedding[-1])
        embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)

        # embedding dropout
        embedding = self.drop(embedding)
        # Transformer Encoder; use embedding for hugging face model and get output states and attention map
        output = self.model(**{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions
        # fea = x[sdc_mask.squeeze(-1)]
        if not return_full_length:
            fea = x[:, 0, :]
            return fea
        elif return_full_length:
            return x