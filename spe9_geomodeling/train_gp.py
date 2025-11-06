import gpytorch
import joblib
import torch
from model_gp import GPModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model(X, y, train_size=3000, seed=42, n_iter=100):
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=train_size, random_state=seed
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(X_tensor, y_tensor, likelihood)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = -mll(output, y_tensor)
        loss.backward()
        if i % 10 == 0:
            print(f"Iter {i}/{n_iter} - Loss: {loss.item():.3f}")
        optimizer.step()

    torch.save(model.state_dict(), "gpr_model_gpytorch.pth")
    torch.save(likelihood.state_dict(), "gpr_likelihood_gpytorch.pth")
    joblib.dump(scaler, "x_scaler.save")
