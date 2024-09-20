"""Neural SDEs."""

import numpy as np
import torch
import torchsde


class GeometricBrownianMotion(torch.nn.Module):

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, mu, sigma, lambdas):
        super().__init__()
        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)
        self.lambdas = torch.tensor(lambdas)
        self.brownian_size = len(self.mu)

        for i in range(self.brownian_size):
            assert np.allclose(torch.square(self.lambdas[i]).sum(), 1.0), torch.square(self.lambdas[i]).sum()
    
    def f(self, t, y):
        return self.mu * y
    
    def g(self, t, y):
        diffusion = torch.zeros_like(y).unsqueeze(-1).repeat((1, 1, self.brownian_size))
        for i in range(self.brownian_size):
            for j in range(self.brownian_size):
                diffusion[..., i, j] = self.lambdas[i, j] * self.sigma[i] * y[..., i]
        return diffusion
    
    def predict(self, observed_processes, num_simulations, dt):
        with torch.no_grad():
            t = torch.tensor(observed_processes[0]["t"])
            Y0 = torch.concat([torch.tensor(op.iloc[0][["S"]]) for op in observed_processes]).repeat(num_simulations, 1)
            Y = torchsde.sdeint(self, Y0, t, dt=dt, method="euler")
            return Y


class Heston(torch.nn.Module):

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, thetas, etas, xis, rs, lambdas):
        super().__init__()
        
        self.rs = torch.tensor(rs)
        self.etas = torch.tensor(etas)
        self.thetas = torch.tensor(thetas)
        self.xis = torch.tensor(xis)
        self.lambdas = torch.tensor(lambdas)
        self.num_stocks = len(rs)
        self.brownian_size = 2 * self.num_stocks

        if torch.any(self.etas) < 0:
            raise AssertionError("eta must be positive")
        for i in range(self.num_stocks):
            assert np.allclose(torch.square(self.lambdas[i]).sum(), 1.0), torch.square(self.lambdas[i]).sum()

    def f(self, t, y):
        drift = torch.zeros_like(y)
        for i in range(self.num_stocks):
            S, nu = self.get_S_nu(y[..., 2 * i : 2 * (i + 1)])
            drift_S = self.rs[i]  * S
            drift_nu = self.etas[i] * (self.thetas[i] - nu)
            drift[..., 2 * i : 2 * (i + 1)] = torch.stack([drift_S, drift_nu], dim=-1)
        return drift

    def g(self, t, y):
        diffusion = torch.zeros_like(y).unsqueeze(-1).repeat((1, 1, self.brownian_size))
        for i in range(self.num_stocks):
            S, nu = self.get_S_nu(y[..., 2 * i : 2 * (i + 1)])
            diffusion_S = torch.sqrt(nu) * S
            for j in range(self.num_stocks):
                diffusion[..., 2 * i, 2 * j] = self.lambdas[i, j] * diffusion_S
            diffusion_nu = torch.sqrt(nu) * self.xis[i]
            diffusion[..., 2 * i + 1, 2 * i + 1] = diffusion_nu
        return diffusion
    
    def get_S_nu(self, y):
        y = torch.maximum(1e-6 * torch.ones_like(y), y)
        S, nu = y[..., 0], y[..., 1]
        return S, nu
    
    def predict(self, observed_processes, num_simulations, dt):
        with torch.no_grad():
            t = torch.tensor(observed_processes[0]["t"])
            Y0 = torch.concat([torch.tensor(op.iloc[0][["S", "ν"]]) for op in observed_processes]).repeat(num_simulations, 1)
            Y = torchsde.sdeint(self, Y0, t, dt=dt, method="euler")
            Y = torch.maximum(1e-6 * torch.ones_like(Y), Y)
            return Y


def infer_heston_parameters(observed_processes):
    thetas, etas, xis, rs = [], [], [], []
    for observed_process in observed_processes:
        r = get_r(observed_process["r"])
        theta = get_theta(observed_process["ν"])
        eta = get_eta(theta, observed_process["_loc_vol_deltas"], observed_process["ν"], observed_process["_time_deltas"])
        xi = get_xi(theta, eta, observed_process["_loc_vol_deltas"], observed_process["_time_deltas"], observed_process["ν"])
        thetas.append(theta), etas.append(eta), xis.append(xi), rs.append(r)
    lambdas = get_lambdas(observed_processes)
    parameters = dict(rs=np.array(rs), thetas=np.array(thetas), xis=np.array(xis), etas=np.array(etas), lambdas=np.array(lambdas))
    return parameters


def get_r(rates):
    return rates.mean()


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


def get_lambdas(observed_processes):
    corrcoef = np.corrcoef([get_noise_S(observed_process) for observed_process in observed_processes])
    lambdas = np.linalg.cholesky(corrcoef)
    return lambdas


def get_noise_S(observed_process):
    return (observed_process["_price_deltas"] - observed_process["r"] * observed_process["S"] * observed_process["_time_deltas"]) / np.sqrt(observed_process["ν"] * observed_process["_time_deltas"]) / observed_process["S"]
