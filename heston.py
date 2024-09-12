import numpy as np
import torch
import torchsde


class Heston(torch.nn.Module):

    noise_type = "diagonal"
    sde_type = "ito"
    state_size = 2

    def __init__(self, theta, eta, xi, r, rho=.0):
        super().__init__()
        
        if eta < 0:
            raise AssertionError("eta must be positive")
        
        self.r = torch.nn.Parameter(torch.tensor(r))
        self.eta = torch.nn.Parameter(torch.tensor(eta))
        self.theta = torch.nn.Parameter(torch.tensor(theta))
        self.xi = torch.nn.Parameter(torch.tensor(xi))
        self.rho = torch.nn.Parameter(torch.tensor(rho))  # TODO

    def f(self, t, y):
        S, nu = self.get_S_nu(y)
        drift_S = self.r  * S
        drift_nu = self.eta * (self.theta - nu)
        drift = torch.stack([drift_S, drift_nu], dim=-1)
        return drift

    def g(self, t, y):
        S, nu = self.get_S_nu(y)
        diffusion_S = torch.sqrt(nu) * S
        diffusion_nu = torch.sqrt(nu) * self.xi
        return torch.stack([diffusion_S, diffusion_nu], dim=-1)
    
    def get_S_nu(self, y):
        y = torch.maximum(1e-9 * torch.ones_like(y), y)
        S, nu = y[..., 0], y[..., 1]
        return S, nu
    
    def predict(self, trajectory, num_simulations, dt):
        with torch.no_grad():
            t = torch.tensor(trajectory["t"])
            Y0 = torch.tensor(trajectory.iloc[0][["S", "ν"]]).repeat(num_simulations, 1)
            Y = torchsde.sdeint(self, Y0, t, dt=dt, method="euler")
            return Y


def infer_heston_parameters(observed_process):
    r = get_r(observed_process["r"])
    theta = observed_process["theta"].mean()
    eta = get_eta(theta, observed_process["_loc_vol_deltas"], observed_process["ν"], observed_process["_time_deltas"])
    xi = get_xi(theta, eta, observed_process["_loc_vol_deltas"], observed_process["_time_deltas"], observed_process["ν"])
    return dict(r=r, theta=theta, eta=eta, xi=xi)


def get_r(rates):
    return rates.mean()


def get_eta(theta, loc_vol_deltas, local_volatilities, time_deltas):
    eta_process = loc_vol_deltas / (theta - local_volatilities) / time_deltas
    eta_process = np.maximum(eta_process, np.zeros_like(eta_process))
    return eta_process.mean()


def get_xi(theta, eta, loc_vol_deltas, time_deltas, local_volatilities):
    numerator = loc_vol_deltas - eta * (theta - local_volatilities) * time_deltas
    denominator = np.sqrt(local_volatilities * time_deltas)
    return (numerator / denominator).std()
