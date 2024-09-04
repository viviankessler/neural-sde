import torch
import torchsde
import matplotlib.pyplot as plt

batch_size, state_size, brownian_size = 32, 1, 1
t_size = 100

class SDE(torch.nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Parameter(.1 * torch.ones(state_size, state_size))
        self.sigma = torch.nn.Parameter(.5 * torch.ones(state_size, state_size * brownian_size))

    # Drift
    def f(self, t, y):
        return (self.mu * y).view(batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        return (self.sigma * y).view(batch_size, state_size, brownian_size)

sde = SDE()
y0 = torch.abs(torch.randn((batch_size, state_size))) + 100
ts = torch.linspace(0, 1, t_size)
ys = torchsde.sdeint(sde, y0, ts).squeeze()
plt.plot(ys.detach().numpy())
plt.show()