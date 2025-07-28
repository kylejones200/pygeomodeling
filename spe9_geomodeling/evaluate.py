import torch
import gpytorch
import joblib
import numpy as np
from model_gp import GPModel


def predict_all(
    X,
    y,
    model_path="gpr_model_gpytorch.pth",
    likelihood_path="gpr_likelihood_gpytorch.pth",
):
    scaler = joblib.load("x_scaler.save")
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(X_tensor, y_tensor, likelihood)
    model.load_state_dict(torch.load(model_path))
    likelihood.load_state_dict(torch.load(likelihood_path))

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_tensor))
        return preds.mean.numpy(), preds.stddev.numpy()
