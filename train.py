import torch

def loss(actual_predict):
    error = None
    for actual, predictions in actual_predict:
        actual = torch.tensor(actual["S"].iloc[-1], requires_grad=False)
        predictions = predictions[-1, :, 0]
        error = torch.abs(actual - predictions) if error is None else error + torch.abs(actual - predictions)
    return error.mean()