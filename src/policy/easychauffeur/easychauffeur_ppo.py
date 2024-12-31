import gym
import torch
import torch.nn as nn
import numpy as np
from ..commons.enc import build_model as build_enc
from src.policy.easychauffeur.distributions import BetaDistribution_waymo


class EasychauffeurPolicy(nn.Module):

    def __init__(self,
                 action_space: dict,
                 policy_head_arch=[256, 256],
                 value_head_arch=[256, 256],
                 pretrain_enc=None,
                 encoder={},
                 distribution_entry_point=None,
                 distribution_kwargs={},
                 **kwargs
                 ):

        super(EasychauffeurPolicy, self).__init__()
        # self.observation_space = observation_space
        # formulate action space for steer and acc
        shape = (2,)
        ranges = action_space.action_ranges
        low = np.array([r[0] for r in ranges])
        high = np.array([r[1] for r in ranges])
        self.action_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
        # end formulate
        self.pretrain_enc = pretrain_enc
        self.encoder = encoder
        self.distribution_entry_point = distribution_entry_point
        self.distribution_kwargs = distribution_kwargs

        if torch.cuda.is_available():
            self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        else:
            self.device = 'cpu'
        self.features_extractor = build_enc(encoder)
        if pretrain_enc is not None:
            self.features_extractor.load_state_dict(torch.load(pretrain_enc))
            print(f'Loaded pretained enc from {pretrain_enc}')

        self.action_dist = BetaDistribution_waymo(**distribution_kwargs)

        # best_so_far
        self.policy_head_arch = list(policy_head_arch)
        self.value_head_arch = list(value_head_arch)
        self.activation_fn = nn.ReLU

        self._build()

    def reset_noise(self, n_envs: int = 1) -> None:
        assert self.use_sde, 'reset_noise() is only available when using gSDE'
        self.action_dist.sample_weights(self.dist_sigma, batch_size=n_envs)

    def _build(self) -> None:
        last_layer_dim_pi = self.features_extractor.hidden_size
        policy_net = []
        for layer_size in self.policy_head_arch:
            policy_net.append(nn.Linear(last_layer_dim_pi, layer_size))
            policy_net.append(self.activation_fn())
            last_layer_dim_pi = layer_size
        # policy_net.append(nn.Linear(last_layer_dim_vf, self.action_space.shape[0]))
        self.policy_head = nn.Sequential(*policy_net).to(self.device)
        # # mu->alpha/mean, sigma->beta/log_std (nn.Module, nn.Parameter)
        self.dist_mu, self.dist_sigma = self.action_dist.proba_distribution_net(last_layer_dim_pi)

        last_layer_dim_vf = self.features_extractor.hidden_size
        value_net = []
        for layer_size in self.value_head_arch:
            value_net.append(nn.Linear(last_layer_dim_vf, layer_size))
            value_net.append(self.activation_fn())
            last_layer_dim_vf = layer_size

        value_net.append(nn.Linear(last_layer_dim_vf, 1))
        self.value_head = nn.Sequential(*value_net).to(self.device)

    def _get_features(self, obs:torch.Tensor) -> torch.Tensor:
        """
        :param birdview: torch.Tensor (num_envs, frame_stack*channel, height, width)
        :param state: torch.Tensor (num_envs, state_dim)
        """
        features = self.features_extractor(obs)
        return features

    def _get_action_dist_from_features(self, features: torch.Tensor):
        latent_pi = self.policy_head(features)
        mu = self.dist_mu(latent_pi)
        if isinstance(self.dist_sigma, nn.Parameter):
            sigma = self.dist_sigma
        else:
            sigma = self.dist_sigma(latent_pi)
        return self.action_dist.proba_distribution(mu, sigma), mu.detach().cpu().numpy(), sigma.detach().cpu().numpy()

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         detach_values=False):
        features = self._get_features(obs)

        if detach_values:
            detached_features = features.detach()
            values = self.value_head(detached_features)
        else:
            values = self.value_head(features)

        distribution, mu, sigma = self._get_action_dist_from_features(features)
        actions = self.scale_action(actions)
        log_prob = distribution.log_prob(actions)
        return values.flatten(), log_prob, distribution.entropy_loss(), distribution.distribution

    def evaluate_values(self, obs: torch.Tensor):
        features = self._get_features(obs)
        values = self.value_head(features)
        distribution, mu, sigma = self._get_action_dist_from_features(features)
        return values.flatten(), distribution.distribution

    def forward(self, obs: np.ndarray, deterministic: bool = False, clip_action: bool = False):
        '''
        used in collect_rollouts(), do not clamp actions
        '''
        with torch.no_grad():
            # obs_tensor_dict = dict([(k, torch.as_tensor(v).to(self.device)) for k, v in obs_dict.items()])
            obs_tensor = torch.as_tensor(obs).to(self.device)
            features = self._get_features(obs_tensor)
            values = self.value_head(features)
            distribution, mu, sigma = self._get_action_dist_from_features(features)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)

        if isinstance(actions,torch.Tensor):
            actions = actions.cpu().numpy()
        actions = self.unscale_action(actions)
        if clip_action:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        values = values.cpu().numpy().flatten()
        log_prob = log_prob.cpu().numpy()
        features = features.cpu().numpy()
        return actions, values, log_prob, mu, sigma, features

    def forward_value(self, obs:np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).to(self.device)
            features = self._get_features(obs_tensor)
            values = self.value_head(features)
        values = values.cpu().numpy().flatten()
        return values

    def forward_policy(self, obs:np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).to(self.device)
            features = self._get_features(obs_tensor)
            distribution, mu, sigma = self._get_action_dist_from_features(features)
        return mu, sigma

    def scale_action(self, action: torch.Tensor, eps=1e-7) -> torch.Tensor:
        # input action \in [a_low, a_high]
        # output action \in [d_low+eps, d_high-eps]
        d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

        if d_low is not None and d_high is not None:
            a_low, a_high = self.action_space.low, self.action_space.high
            # brodcast a_low and a_high to [batch_size, action_dim]
            a_low = np.repeat(a_low.reshape(1,-1), action.shape[0], axis=0)
            a_high = np.repeat(a_high.reshape(1,-1), action.shape[0], axis=0)
            # to tensor
            a_low = torch.as_tensor(a_low.astype(np.float32)).to(action.device)
            a_high = torch.as_tensor(a_high.astype(np.float32)).to(action.device)
            # same shape as action [batch_size, action_dim]
            action = (action-a_low)/(a_high-a_low) * (d_high-d_low) + d_low
            action = torch.clamp(action, d_low+eps, d_high-eps)
        return action

    def unscale_action(self, action: np.ndarray, eps=0.0) -> np.ndarray:
        # input action \in [d_low, d_high]
        # output action \in [a_low+eps, a_high-eps]
        d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

        if d_low is not None and d_high is not None:
            # batch_size = action.shape[0]
            a_low, a_high = self.action_space.low, self.action_space.high
            # brodcast a_low and a_high to [batch_size, action_dim]
            a_low = np.repeat(a_low.reshape(1,-1), action.shape[0], axis=0)
            a_high = np.repeat(a_high.reshape(1,-1), action.shape[0], axis=0)
            # same shape as action [batch_size, action_dim]
            # a_high = np.tile(self.action_space.high, [batch_size, 1])
            action = (action-d_low)/(d_high-d_low) * (a_high-a_low) + a_low
            # action = np.clip(action, a_low+eps, a_high-eps)

        return action

    def get_predictions(self, states, actions, timesteps, num_envs=1, **kwargs):
        state = states[:,-1]
        action, values, log_probs, mu, sigma, _ = self.forward(state, deterministic=True, clip_action=True)
        return action

    @classmethod
    def load(cls, path):
        if torch.cuda.is_available():
            device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        else:
            device = 'cpu'
        saved_variables = torch.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables['policy_init_kwargs'])
        # Load weights
        model.load_state_dict(saved_variables['policy_state_dict'])
        model.to(device)
        print(f"Loaded policy from {path}")
        return model, saved_variables['train_init_kwargs']

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)