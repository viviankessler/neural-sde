import torch
import torchsde


class SDE(torch.nn.Module):
    noise_type = 'general'
    sde_type = 'ito'
    state_size = 1
    brownian_size = 1

    def __init__(self, sample_size, times, desired_log_returns):
        super().__init__()
        self.sample_size = sample_size
        self.times = times
        self.desired_log_returns = desired_log_returns
    
    def forward(self, initial_states):
        sample_paths = torchsde.sdeint(self, initial_states, self.times)
        terminal_states = sample_paths[-1]
        estimated_log_returns = torch.log(terminal_states / initial_states)
        #estimated_log_returns = torch.nan_to_num(estimated_log_returns, nan=1.0)
        return estimated_log_returns
    
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


class DeepSDE(SDE):

    def __init__(self, mu, sigma, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma
        self.activation = torch.nn.Tanh()
        self.f1 = torch.nn.Linear(self.state_size, hidden_size)
        self.f2 = torch.nn.Linear(hidden_size, self.state_size)
        with torch.no_grad():
            self.f1.weight = torch.nn.Parameter(mu * torch.eye(self.state_size, hidden_size))
            self.f2.weight = torch.nn.Parameter(torch.eye(hidden_size, self.state_size))
            self.f1.bias = torch.nn.Parameter(torch.zeros(hidden_size))
            self.f2.bias = torch.nn.Parameter(torch.zeros(self.state_size))
        self.g1 = torch.nn.Linear(self.state_size, hidden_size)
        self.g2 = torch.nn.Linear(hidden_size, self.state_size)
        with torch.no_grad():
            self.g1.weight = torch.nn.Parameter(sigma * torch.eye(self.state_size, hidden_size))
            self.g2.weight = torch.nn.Parameter(torch.eye(hidden_size, self.state_size))
            self.g1.bias = torch.nn.Parameter(torch.zeros(hidden_size))
            self.g2.bias = torch.nn.Parameter(torch.zeros(self.state_size))

    def f(self, t, y):
        y = y.to(torch.float32)
        y = self.f1(y)
        y = self.activation(y)
        y = self.f2(y)
        y = self.activation(y)
        return y.view(self.sample_size, self.state_size)

    def g(self, t, y):
        y = y.to(torch.float32)
        y = self.g1(y)
        y = self.activation(y)
        y = self.g2(y)
        y = self.activation(y)
        return y.view(self.sample_size, self.state_size, self.brownian_size)
