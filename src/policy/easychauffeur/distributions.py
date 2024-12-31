from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions import Beta, Normal

def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

class DiagGaussianDistribution_waymo():
    def __init__(self, control_type='waypoint', dist_init=None, action_dependent_std=False):
        self.distribution = None
        if control_type == 'waypoint':
            self.action_dim = 3
        elif control_type == 'bicycle':
            self.action_dim = 2
        self.dist_init = dist_init
        self.action_dependent_std = action_dependent_std

        self.low = None
        self.high = None
        self.log_std_max = 2
        self.log_std_min = -20

        # [mu, log_std], [0, 1]
        self.acc_exploration_dist = {
            'go': torch.FloatTensor([0.66, -3]),
            'stop': torch.FloatTensor([-0.66, -3])
        }
        self.steer_exploration_dist = {
            'turn': torch.FloatTensor([0.0, -1]),
            'straight': torch.FloatTensor([3.0, 3.0])
        }

        if torch.cuda.is_available():
            self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        else:
            self.device = 'cpu'

    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Parameter]:
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        if self.action_dependent_std:
            log_std = nn.Linear(latent_dim, self.action_dim)
        else:
            log_std = nn.Parameter(-2.0*torch.ones(self.action_dim), requires_grad=True)

        if self.dist_init is not None:
            # log_std.weight.data.fill_(0.01)
            # mean_actions.weight.data.fill_(0.01)
            # acc/steer
            mean_actions.bias.data[0] = self.dist_init[0][0]
            mean_actions.bias.data[1] = self.dist_init[1][0]
            if self.control_type == 'waypoint':
                # dyaw
                mean_actions.bias.data[2] = self.dist_init[2][0]

            if self.action_dependent_std:
                log_std.bias.data[0] = self.dist_init[0][1]
                log_std.bias.data[1] = self.dist_init[1][1]
            else:
                init_tensor = torch.FloatTensor([self.dist_init[0][1], self.dist_init[1][1]])
                log_std = nn.Parameter(init_tensor, requires_grad=True)

        return mean_actions, log_std

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor):
        if self.action_dependent_std:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy_loss(self) -> torch.Tensor:
        entropy_loss = -1.0 * self.distribution.entropy()
        return torch.mean(entropy_loss)

    def exploration_loss(self, exploration_suggests) -> torch.Tensor:
        # [('stop'/'go'/None, 'turn'/'straight'/None)]
        # (batch_size, action_dim)
        mu = self.distribution.loc.detach().clone()
        sigma = self.distribution.scale.detach().clone()

        for i, (acc_suggest, steer_suggest) in enumerate(exploration_suggests):
            if acc_suggest != '':
                mu[i, 0] = self.acc_exploration_dist[acc_suggest][0]
                sigma[i, 0] = self.acc_exploration_dist[acc_suggest][1]
            if steer_suggest != '':
                mu[i, 1] = self.steer_exploration_dist[steer_suggest][0]
                sigma[i, 1] = self.steer_exploration_dist[steer_suggest][1]

        dist_ent = Normal(mu, sigma)

        exploration_loss = torch.distributions.kl_divergence(dist_ent, self.distribution)
        return torch.mean(exploration_loss)

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()


class GaussianDistribution_waymo():
    '''
        This class aims to output an continous action as mean of the distrubtion
    '''
    def __init__(self, control_type='waypoint', dist_init=None):
        self.control_type = control_type

        self.distribution = None
        if control_type == 'waypoint':
            self.action_dim = 3
        elif control_type == 'bicycle':
            self.action_dim = 2
        # not used for now
        self.dist_init = dist_init
        self.low = 0.0
        self.high = 1.0

        if torch.cuda.is_available():
            self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        else:
            self.device = 'cpu'

    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Module]:
        '''
            build the network, the mean is from pretained BC, 
            the sigma is learnable parameter
        '''
        linear_mu = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64, self.action_dim)).to(self.device)
        # cov_var = nn.Parameter(torch.full(size=(self.action_dim,), fill_value=0.5).to(self.device),requires_grad=False)
        log_std = nn.Parameter(-2.0*torch.ones(self.action_dim), requires_grad=True)

        return linear_mu, log_std

    def proba_distribution(self, mean_actions, log_std):
        # cov_mat = torch.diag(cov_var)
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        # self.distribution = MultivariateNormal(mean, cov_mat)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy_loss(self) -> torch.Tensor:
        entropy_loss = -1.0 * self.distribution.entropy()
        return torch.mean(entropy_loss)

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        x = self.distribution.mean
        return x

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

class BetaDistribution_waymo():
    def __init__(self, control_type='waypoint', dist_init=None):
        self.control_type = control_type

        self.distribution = None
        if control_type == 'waypoint':
            self.action_dim = 3
        elif control_type == 'bicycle':
            self.action_dim = 2
        self.dist_init = dist_init
        self.low = 0.0
        self.high = 1.0
        
        # [beta, alpha], [0, 1]
        '''not used for waymo for now'''
        self.acc_exploration_dist = {
            # [1, 2.5]
            # [1.5, 1.0]
            'go': torch.FloatTensor([1.0, 2.5]),
            'stop': torch.FloatTensor([1.5, 1.0])
        }
        self.steer_exploration_dist = {
            'turn': torch.FloatTensor([1.0, 1.0]),
            'straight': torch.FloatTensor([3.0, 3.0])
        }

        if torch.cuda.is_available():
            self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        else:
            self.device = 'cpu'

    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Module]:

        linear_alpha = nn.Linear(latent_dim, self.action_dim)
        linear_beta = nn.Linear(latent_dim, self.action_dim)

        if self.dist_init is not None:
            # linear_alpha.weight.data.fill_(0.01)
            # linear_beta.weight.data.fill_(0.01)
            # dx/ acc
            linear_alpha.bias.data[0] = self.dist_init[0][1]
            linear_beta.bias.data[0] = self.dist_init[0][0]
            # dy/ steer
            linear_alpha.bias.data[1] = self.dist_init[1][1]
            linear_beta.bias.data[1] = self.dist_init[1][0]
            if self.control_type == 'waypoint':
                # dyaw
                linear_alpha.bias.data[2] = self.dist_init[2][1]
                linear_beta.bias.data[2] = self.dist_init[2][0]

        alpha = nn.Sequential(linear_alpha, nn.Softplus())
        beta = nn.Sequential(linear_beta, nn.Softplus())
        return alpha, beta

    def proba_distribution(self, alpha, beta):
        self.distribution = Beta(alpha, beta)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy_loss(self) -> torch.Tensor:
        entropy_loss = -1.0 * self.distribution.entropy()
        return torch.mean(entropy_loss)
    '''not used for waymo for now'''
    def exploration_loss(self, exploration_suggests) -> torch.Tensor:
        # [('stop'/'go'/None, 'turn'/'straight'/None)]
        # (batch_size, action_dim)
        alpha = self.distribution.concentration1.detach().clone()
        beta = self.distribution.concentration0.detach().clone()

        for i, (acc_suggest, steer_suggest) in enumerate(exploration_suggests):
            if acc_suggest != '':
                beta[i, 0] = self.acc_exploration_dist[acc_suggest][0]
                alpha[i, 0] = self.acc_exploration_dist[acc_suggest][1]
            if steer_suggest != '':
                beta[i, 1] = self.steer_exploration_dist[steer_suggest][0]
                alpha[i, 1] = self.steer_exploration_dist[steer_suggest][1]

        dist_ent = Beta(alpha, beta)

        exploration_loss = torch.distributions.kl_divergence(self.distribution, dist_ent)
        return torch.mean(exploration_loss)

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:

        x = self.distribution.mean
        return x

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

