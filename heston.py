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
        
        self.r = torch.nn.Parameter(torch.tensor(r), requires_grad=False)
        self.eta = torch.nn.Parameter(torch.tensor(eta), requires_grad=False)
        self.theta = torch.nn.Parameter(torch.tensor(theta), requires_grad=False)
        self.xi = torch.nn.Parameter(torch.tensor(xi), requires_grad=False)
        self.rho = torch.nn.Parameter(torch.tensor(rho), requires_grad=False)  # TODO

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
    
    @staticmethod
    def get_S_nu(y):
        y = torch.maximum(1e-9 * torch.ones_like(y), y)
        S, nu = y[..., 0], y[..., 1]
        return S, nu
    
    def predict(self, trajectory, num_simulations, dt):
        t = pandas_to_tensor(trajectory["t"])
        Y0 = pandas_to_tensor(trajectory.iloc[0][["S", "ν"]]).repeat(num_simulations, 1)
        Y = torchsde.sdeint(self, Y0, t, dt=dt, method="euler")
        return Y


def infer_heston_parameters(observed_process):
    r = get_r(observed_process["_price_deltas"], observed_process["S"], observed_process["_time_deltas"])
    theta = get_theta(observed_process["ν"])
    eta = get_eta(theta, observed_process["_loc_vol_deltas"], observed_process["ν"], observed_process["_time_deltas"])
    xi = get_xi(theta, eta, observed_process["_loc_vol_deltas"], observed_process["_time_deltas"], observed_process["ν"])
    return dict(r=r, theta=theta, eta=eta, xi=xi)


def get_r(price_deltas, observed_prices, time_deltas):
    return (price_deltas / observed_prices / time_deltas).mean()


def get_theta(local_volatilities):
    return local_volatilities.mean()


def get_eta(theta, loc_vol_deltas, local_volatilities, time_deltas):
    eta_process = loc_vol_deltas / (theta - local_volatilities) / time_deltas
    eta_process = np.maximum(eta_process, np.zeros_like(eta_process))
    return eta_process.mean()


def get_xi(theta, eta, loc_vol_deltas, time_deltas, local_volatilities):
    numerator = loc_vol_deltas - eta * (theta - local_volatilities) * time_deltas
    denominator = np.sqrt(local_volatilities * time_deltas)
    return (numerator / denominator).std()


def pandas_to_tensor(pandas_object):
    return torch.tensor(pandas_object, requires_grad=False)
