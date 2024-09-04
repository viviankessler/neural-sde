import torch
import torchsde
import matplotlib.pyplot as plt

batch_size, state_size, brownian_size = 32, 1, 1
t_size = 20

class SDE(torch.nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.ones(state_size, state_size))
        self.sigma = torch.nn.Parameter(torch.ones(state_size, state_size * brownian_size))

    # Drift
    def f(self, t, y):
        return self.mu * y  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        return (self.sigma * y).view(batch_size, state_size, brownian_size)

sde = SDE()
y0 = torch.full((batch_size, state_size), 0.1)
ts = torch.linspace(0, 1, t_size)
# Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
# ys will have shape (t_size, batch_size, state_size)
ys = torchsde.sdeint(sde, y0, ts).squeeze()
plt.plot(ys.detach().numpy())
plt.show()