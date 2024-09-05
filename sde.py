import torch
import torchsde


class SDE(torch.nn.Module):
    noise_type = 'general'
    sde_type = 'ito'
    state_size = 1
    brownian_size = 1

    def __init__(self, sample_size, times):
        super().__init__()
        self.sample_size = sample_size
        self.times = times
    
    def forward(self, initial_states):
        sample_paths = torchsde.sdeint(self, initial_states, self.times)
        terminal_states = sample_paths[-1]
        return torch.log(terminal_states / initial_states)
    
    def solution(self, initial_states):
        """
        Parameters:
            initial_states: of shape (sample_size, state_size)
            times: 1-dimensional tensor
        """
        with torch.no_grad():
            initial_states = torch.tensor(initial_states)
            return torchsde.sdeint(self, initial_states, self.times).numpy().squeeze()


class BlackScholes(SDE):

    def __init__(self, mu, sigma, **kwargs):
        super().__init__(**kwargs)
        self.mu = torch.nn.Parameter(mu * torch.ones(self.state_size, self.state_size))
        self.sigma = torch.nn.Parameter(sigma * torch.ones(self.state_size, self.state_size * self.brownian_size))

    def f(self, t, y):
        return (self.mu * y).view(self.sample_size, self.state_size)

    def g(self, t, y):
        return (self.sigma * y).view(self.sample_size, self.state_size, self.brownian_size)
