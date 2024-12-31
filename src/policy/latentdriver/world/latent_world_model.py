import torch
import torch.nn as nn
import transformers

from ..gpt2_model import GPT2Model

from ..transformers.utils import PerceiverDecoder,TrainableQueryProvider

def sample_from_distribution(mu, sigma, deterministic):
    sample = mu
    if not deterministic:
        noise = torch.randn_like(sample)
        sample = sample + sigma * noise
    return sample
class RepresentationModel(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_std = 0.1

        self.module = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(True),
            nn.Linear(in_channels, 2*self.latent_dim),
        )

    def forward(self, x):
        def sigmoid2(tensor: torch.Tensor, min_value: float) -> torch.Tensor:
            return 2 * torch.sigmoid(tensor / 2) + min_value

        mu_log_sigma = self.module(x)
        mu, log_sigma = torch.split(mu_log_sigma, self.latent_dim, dim=-1)

        sigma = sigmoid2(log_sigma, self.min_std)
        return mu, sigma
class LatentWorldModel(nn.Module):
    def __init__(        
        self,
        act_dim,
        enc_hidden_size,
        hidden_size,
        ordering=1,
        max_ep_len=4096,
        decoder_layer = 4,
        decoder_head = 4,
        representation_query_num = 32,
        reconstruction_query_num = 189,
        GPT_conf = None,
        **kwargs,):
        super().__init__()
        self.act_dim = act_dim
        self.enc_hidden_size = enc_hidden_size
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **GPT_conf
        )
        self.transformer = GPT2Model(config)
        representation_query = TrainableQueryProvider(
            num_queries=representation_query_num,
            num_query_channels=enc_hidden_size,
            init_scale=0.1,
        )
        self.representation_decoder = PerceiverDecoder(representation_query,enc_hidden_size,decoder_head,decoder_layer)  
        self.pe_rep = nn.Parameter(torch.zeros((1, 1, reconstruction_query_num, enc_hidden_size)),requires_grad=True)
        self.pe_gpt_instance = nn.Parameter(torch.zeros((1, 1, representation_query_num+self.act_dim, hidden_size)),requires_grad=True)
        self.rep_dis_net = RepresentationModel(hidden_size, hidden_size)
  
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.act_dim*hidden_size)
        self.latent_net = RepresentationModel(hidden_size, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.ordering = ordering
    def forward(self,
                bert_embeddings, # bs T N+1 D
                actions, # bs T D
                timesteps, # bs, T
                padding_mask, # bs, T
                ):
        
        if padding_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        batch_size, seq_length = bert_embeddings.shape[0], bert_embeddings.shape[1]

        r_decoder_input = (bert_embeddings+self.pe_rep).reshape(batch_size*seq_length, -1, self.enc_hidden_size)
        representation_embedding = self.representation_decoder(r_decoder_input).reshape(batch_size, seq_length, -1, self.enc_hidden_size)

        representation_dist_mu, representation_dist_sigma = self.rep_dis_net(representation_embedding)
        representation_embedding = sample_from_distribution(representation_dist_mu, representation_dist_sigma, not self.training)

        if actions is not None:
            action_embeddings = self.embed_action(actions).reshape(batch_size,seq_length,3,self.hidden_size)
        else:
            action_embeddings = torch.zeros((batch_size, seq_length, 3, self.hidden_size), device=bert_embeddings.device)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps).unsqueeze(2)
        else:
            order_embeddings = 0.0
        state_embeddings = representation_embedding + order_embeddings
        action_embeddings = action_embeddings + order_embeddings
        # this makes the sequence look like (x1 y1 z1 s1_1 s1_2 s1_3... x2 y2 z2 s2_1 s2_2 ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.concat([action_embeddings, state_embeddings],dim=-2) + self.pe_gpt_instance
        sentence_length = stacked_inputs.shape[-2] 
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_padding_mask = padding_mask.repeat(1, 1, sentence_length)
        if actions is not None:
            pass
        else: 
            stacked_padding_mask[:,:,:3] = 0
        stacked_padding_mask = stacked_padding_mask.reshape(batch_size, -1)
            
        stacked_inputs = stacked_inputs.reshape(batch_size, -1, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs, attention_mask=stacked_padding_mask)
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # states (0), or actions (1); i.e. x[:,0,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, -1, self.hidden_size)
        latent_representation_token = x[:,:,3:,:] # bs, len, M, D
        #  mu:  bs, len, M, D
        # sigma: bs, len, M, D
        latent_state_mu, latent_state_sigma = self.latent_net(latent_representation_token)
        # latent_space: bs, len, M, D
        # latent_space = sample_from_distribution(latent_state_mu,latent_state_sigma,not self.training)
        
        # c_decoder_input = latent_space.reshape(batch_size * seq_length, -1, self.hidden_size)
        # reconstructed_token: bs, len, 1+N, D
        # reconstructed_token = self.reconstruction_decoder(c_decoder_input).reshape(batch_size, seq_length, -1, self.hidden_size)
        return (representation_dist_mu, representation_dist_sigma), (latent_state_mu,latent_state_sigma)