import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, date
import torch
import torchsde
import matplotlib.pyplot as plt

MODE = "close"


def simulate(
        stock_name,
        num_simulations=10,
        start=date.today() - timedelta(days=1),
        end=date.today(),
        interval="1m",
        local_vol_window_size=15,
        dt=1 if MODE == "close" else 60,
        plot=False,
        title="",
        parameters="infer",
        ):
    
    observed_process = get_observed_process(stock_name, start, end, interval, local_vol_window_size)
    times = torch.tensor(observed_process["t"], requires_grad=False)
    initial_states = torch.tensor(observed_process[["S", "ν"]].iloc[0], requires_grad=False).repeat(num_simulations, 1)
    if parameters == "infer":
        parameters = infer_heston_parameters(observed_process)
    sde = Heston(**parameters)
    solution = torchsde.sdeint(sde, initial_states, times, method="euler", dt=dt)

    if plot:
        plot_simulation(solution, observed_process, title)

    return solution, parameters, observed_process


def plot_simulation(simulation, observed_process, title):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    """for i, trajectory in enumerate(simulation[:, :20, 1].transpose(0, 1)):
        ax2.scatter(observed_process.index, trajectory, c="navy", alpha=.2, label="simulation ($\\nu$)" if i == 0 else None)"""
    for i, trajectory in enumerate(simulation[..., 0].transpose(0, 1)):
        ax1.plot(observed_process.index, trajectory, c="red", alpha=.5, label="simulation ($S$)" if i == 0 else None)
    ax1.plot(observed_process.index, list(observed_process["S"]), c="purple", linewidth=3, label="market data ($S$)")
    # ax2.scatter(observed_process.index, list(observed_process["ν"]), c="blue", linewidth=3, label="market data ($\\nu$)", marker="x", alpha=.2)
    ax1.legend(), ax1.set_ylabel("$S$", color="red")
    ax2.legend(), ax2.set_ylabel("$\\nu$", color="grey")
    plt.title(title)
    plt.show()


class Heston(torch.nn.Module):

    noise_type = "diagonal"
    sde_type = "ito"
    state_size = 2

    def __init__(self, theta, eta, xi, r=.0, q=.0, rho=.0):
        super().__init__()
        if eta < 0:
            raise AssertionError("eta must be positive")
        self.r = torch.nn.Parameter(torch.tensor(r), requires_grad=False)
        self.q = torch.nn.Parameter(torch.tensor(q), requires_grad=False)
        self.eta = torch.nn.Parameter(torch.tensor(eta), requires_grad=False)
        self.theta = torch.nn.Parameter(torch.tensor(theta), requires_grad=False)
        self.xi = torch.nn.Parameter(torch.tensor(xi), requires_grad=False)
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


def infer_heston_parameters(observed_process):
    r = (observed_process["S"].diff() / observed_process["S"] / observed_process["time_deltas"]).mean()
    theta = observed_process["ν"].mean()
    local_volatility_deltas = observed_process["ν"].diff()
    eta = get_eta(local_volatility_deltas, observed_process, theta)
    xi = (local_volatility_deltas / np.sqrt(observed_process["ν"] * observed_process["time_deltas"])).std()
    return dict(r=r, theta=theta, eta=eta, xi=xi)


def get_eta(local_volatility_deltas, observed_process, theta):
    eta_quantity = local_volatility_deltas.shift(1) / (theta - observed_process["ν"]) / observed_process["time_deltas"]
    eta_quantity = np.maximum(np.zeros_like(eta_quantity), eta_quantity)
    return eta_quantity.mean()


def get_observed_process(stock_name, start, end, interval, local_vol_window_size):
    
    observed_prices = get_S(stock_name, start, end, interval)
    observation_times = get_t(observed_prices)
    time_deltas = get_time_deltas(observation_times, observed_prices)
    local_volatilities = get_nu(time_deltas, observed_prices, local_vol_window_size)

    observed_process = pd.DataFrame({
        "S": observed_prices,
        "t": observation_times,
        "time_deltas": time_deltas,
        "ν": local_volatilities,
        }, index=observed_prices.index
    ).dropna()

    return observed_process


def get_S(stock_name, start, end, interval):
    return yf.download(stock_name, start=start, end=end, interval=interval)["Adj Close"]


def get_t(observed_prices):
    observation_times = observed_prices.index.astype("int64")  # datetime to number of 1e-9 seconds since 1/1/1970
    observation_times = (observation_times - min(observation_times)) / 1e9
    if MODE == "close":
        observation_times /= 24 * 60 * 60
    return observation_times


def get_time_deltas(observation_times, observed_prices):
    return pd.Series(observation_times.diff(), index=observed_prices.index)


def get_nu(time_deltas, observed_prices, local_vol_window_size):
    price_quotients = observed_prices.diff() / observed_prices
    local_volatilities = price_quotients.rolling(local_vol_window_size).var().div(time_deltas).shift(periods=local_vol_window_size)
    return local_volatilities
