import torch
import torchsde


class Heston(torch.nn.Module):

    noise_type = "diagonal"
    sde_type = "ito"
    state_size = 2

    def __init__(self, theta, eta, xi, r, q=.0, rho=.0):
        super().__init__()
        if eta < 0:
            raise AssertionError("eta must be positive")
        self.r = torch.nn.Parameter(torch.tensor(r), requires_grad=False)
        self.q = torch.nn.Parameter(torch.tensor(q), requires_grad=False)
        self.eta = torch.nn.Parameter(torch.tensor(eta))
        self.theta = torch.nn.Parameter(torch.tensor(theta))
        self.xi = torch.nn.Parameter(torch.tensor(xi))
        self.rho = torch.nn.Parameter(torch.tensor(rho), requires_grad=False)  # TODO

    def f(self, t, y):
        y = torch.maximum(1e-9 * torch.ones_like(y), y)
        S, nu = y[..., 0], y[..., 1]
        drift_S = (self.r - self.q) * S
        drift_nu = self.eta * (self.theta - nu)
        drift = torch.stack([drift_S, drift_nu], dim=-1)
        return drift

    def g(self, t, y):
        y = torch.maximum(1e-9 * torch.ones_like(y), y)
        S, nu = y[..., 0], y[..., 1]
        diffusion_S = torch.sqrt(nu) * S
        diffusion_nu = torch.sqrt(nu) * self.xi
        return torch.stack([diffusion_S, diffusion_nu], dim=-1)
    
    def predict(self, trajectory, num_simulations, dt):
        times = torch.tensor(trajectory["t"], requires_grad=False)
        initial_states = torch.tensor(trajectory[["S", "ν"]].iloc[0], requires_grad=False).repeat(num_simulations, 1)
        solution = torchsde.sdeint(self, initial_states, times, dt=dt, method="euler")
        return solution
