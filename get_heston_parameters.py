import numpy as np


def infer_heston_parameters(observed_process):
    r = observed_process["r"].mean()
    theta = observed_process["ν"].mean()
    local_volatility_deltas = observed_process["ν"].diff()
    eta = get_eta(local_volatility_deltas, observed_process, theta)
    xi = (local_volatility_deltas / np.sqrt(observed_process["ν"] * observed_process["time_deltas"])).std()
    return dict(r=r, theta=theta, eta=eta, xi=xi)


def get_eta(local_volatility_deltas, observed_process, theta):
    eta_quantity = local_volatility_deltas.shift(1) / (theta - observed_process["ν"]) / observed_process["time_deltas"]
    eta_quantity = np.maximum(np.zeros_like(eta_quantity), eta_quantity)
    return eta_quantity.mean()
