import torch
from torch import nn
class ProbabilisticLoss(nn.Module):
    """ Given a prior distribution and a posterior distribution, this module computes KL(posterior, prior)"""
    def __init__(self, remove_first_timestamp=True):
        super().__init__()
        self.remove_first_timestamp = remove_first_timestamp

    def forward(self, prior_mu, prior_sigma, posterior_mu, posterior_sigma):
        posterior_var = posterior_sigma ** 2
        prior_var = prior_sigma ** 2

        posterior_log_sigma = torch.log(posterior_sigma)
        prior_log_sigma = torch.log(prior_sigma)

        kl_div = (
                prior_log_sigma - posterior_log_sigma - 0.5
                + (posterior_var + (posterior_mu - prior_mu) ** 2) / (2 * prior_var)
        )
        first_kl = - posterior_log_sigma[:, :1] - 0.5 + (posterior_var[:, :1] + posterior_mu[:, :1] ** 2) / 2
        kl_div = torch.cat([first_kl, kl_div], dim=1)

        # Sum across channel dimension
        # Average across batch dimension, keep time dimension for monitoring
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss
class KLLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.loss = ProbabilisticLoss(remove_first_timestamp=False)

    def forward(self, prior, posterior):
        prior_mu, prior_sigma = prior['mu'], prior['sigma']
        posterior_mu, posterior_sigma = posterior['mu'], posterior['sigma']
        prior_loss = self.loss(prior_mu, prior_sigma, posterior_mu.detach(), posterior_sigma.detach())
        posterior_loss = self.loss(prior_mu.detach(), prior_sigma.detach(), posterior_mu, posterior_sigma)
        # return prior_loss
        return self.alpha * prior_loss + (1 - self.alpha) * posterior_loss