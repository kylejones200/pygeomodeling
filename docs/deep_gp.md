# Deep Gaussian Process Guide

This guide covers advanced Deep Gaussian Process modeling techniques available in the SPE9 Geomodeling Toolkit.

## 🧠 What are Deep Gaussian Processes?

Deep Gaussian Processes (Deep GPs) combine the uncertainty quantification of Gaussian Processes with the representational power of deep neural networks. They use neural networks as feature extractors before applying a final GP layer.

### Architecture Overview

```
Input (x, y, z) → Neural Network → Learned Features → Gaussian Process → Output + Uncertainty
```

## 🚀 Quick Start with Deep GP

### Basic Deep GP Model

```python
from spe9_geomodeling import DeepGPExperiment

# Run the complete experiment
experiment = DeepGPExperiment()
results = experiment.run_comparison_experiment()

# View results
print("Model Performance Comparison:")
for model_name, result in results.items():
    metrics = result['metrics']
    print(f"{model_name}: R² = {metrics['r2_score']:.4f}, "
          f"RMSE = {metrics['rmse']:.2f}, "
          f"Time = {result['training_time']:.1f}s")
```

### Custom Deep GP Architecture

```python
from spe9_geomodeling import create_gp_model
import torch
import gpytorch

# Create custom Deep GP
model = create_gp_model(
    model_type='deep',
    hidden_dims=[64, 32, 16],  # 3-layer network
    kernel_type='rbf',
    input_dim=3,
    activation='relu'
)

# Custom likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
```

## 🏗️ Architecture Design

### Network Size Guidelines

#### Small Networks (16-32 neurons)

```python
model = create_gp_model('deep', hidden_dims=[32, 16])
```

- **Best for**: Small datasets (<1000 points)
- **Training time**: ~1-2 seconds
- **Memory usage**: Low
- **Risk**: May underfit complex patterns

#### Medium Networks (32-64 neurons)

```python
model = create_gp_model('deep', hidden_dims=[64, 32])
```

- **Best for**: Medium datasets (1000-5000 points)
- **Training time**: ~2-4 seconds
- **Memory usage**: Medium
- **Balance**: Good capacity vs. efficiency

#### Large Networks (64-128 neurons)

```python
model = create_gp_model('deep', hidden_dims=[128, 64, 32])
```

- **Best for**: Large datasets (>5000 points)
- **Training time**: ~4-8 seconds
- **Memory usage**: High
- **Risk**: May overfit small datasets

### Activation Functions

#### ReLU (Default)

```python
model = create_gp_model('deep', activation='relu')
```

- **Pros**: Fast computation, good gradients
- **Cons**: Can cause dead neurons
- **Best for**: General purpose applications

#### Tanh

```python
model = create_gp_model('deep', activation='tanh')
```

- **Pros**: Smooth, bounded output
- **Cons**: Vanishing gradient problems
- **Best for**: Normalized input data

#### GELU

```python
model = create_gp_model('deep', activation='gelu')
```

- **Pros**: Smooth, probabilistic interpretation
- **Cons**: Slightly slower computation
- **Best for**: Modern architectures

## 🎯 Training Deep GP Models

### Basic Training Loop

```python
import torch
import gpytorch
from torch.optim import Adam

# Set up model and likelihood
model = create_gp_model('deep', hidden_dims=[64, 32])
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Training mode
model.train()
likelihood.train()

# Optimizer
optimizer = Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# Loss function
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(y_train))

# Training loop
for i in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        print(f'Iter {i}, Loss: {loss.item():.3f}')
```

### Advanced Training Techniques

#### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR

optimizer = Adam(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

for epoch in range(100):
    # Training step...
    scheduler.step()
```

#### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

early_stopping = EarlyStopping(patience=15)

for epoch in range(200):
    # Training...
    val_loss = validate_model(model, X_val, y_val)
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

#### Batch Training

```python
from torch.utils.data import DataLoader, TensorDataset

# Create data loader
dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Batch training
for epoch in range(100):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = -mll(output, batch_y)
        loss.backward()
        optimizer.step()
```

## 🔧 Hyperparameter Optimization

### Manual Grid Search

```python
def train_and_evaluate(hidden_dims, lr, num_epochs):
    model = create_gp_model('deep', hidden_dims=hidden_dims)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Training...
    # Evaluation...
    return r2_score

# Grid search
best_score = -float('inf')
best_params = None

for hidden_dims in [[32, 16], [64, 32], [128, 64]]:
    for lr in [0.01, 0.1, 0.2]:
        for epochs in [50, 100, 150]:
            score = train_and_evaluate(hidden_dims, lr, epochs)
            if score > best_score:
                best_score = score
                best_params = (hidden_dims, lr, epochs)

print(f"Best parameters: {best_params}, Score: {best_score:.4f}")
```

### Optuna Optimization

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    hidden_dim1 = trial.suggest_int('hidden_dim1', 16, 128)
    hidden_dim2 = trial.suggest_int('hidden_dim2', 8, 64)
    lr = trial.suggest_float('lr', 0.01, 0.3, log=True)
    epochs = trial.suggest_int('epochs', 50, 200)

    # Train model
    model = create_gp_model('deep', hidden_dims=[hidden_dim1, hidden_dim2])
    score = train_and_evaluate_model(model, lr, epochs)

    return score

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best trial: {study.best_trial.params}")
print(f"Best score: {study.best_value:.4f}")
```

## 📊 Model Interpretation

### Feature Importance Analysis

```python
import torch.nn.functional as F

def analyze_feature_importance(model, X_test):
    model.eval()

    # Get feature activations
    with torch.no_grad():
        # Forward pass through network layers
        x = torch.tensor(X_test, dtype=torch.float32)
        activations = []

        for layer in model.feature_extractor:
            x = layer(x)
            if isinstance(layer, torch.nn.Linear):
                activations.append(x.clone())

    # Analyze activation patterns
    for i, activation in enumerate(activations):
        importance = torch.mean(torch.abs(activation), dim=0)
        print(f"Layer {i+1} feature importance: {importance.numpy()}")
```

### Uncertainty Decomposition

```python
def decompose_uncertainty(model, likelihood, X_test):
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Get predictive distribution
        pred_dist = likelihood(model(X_test))

        # Aleatoric uncertainty (data noise)
        aleatoric = likelihood.noise.item()

        # Epistemic uncertainty (model uncertainty)
        epistemic = pred_dist.variance.mean().item() - aleatoric

        return {
            'total_uncertainty': pred_dist.variance.mean().item(),
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic
        }

uncertainty_breakdown = decompose_uncertainty(model, likelihood, X_test)
print("Uncertainty Decomposition:")
for key, value in uncertainty_breakdown.items():
    print(f"  {key}: {value:.4f}")
```

## 🎨 Visualization

### Training Progress

```python
import matplotlib.pyplot as plt

def plot_training_progress(losses, val_losses=None):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Deep GP Training Progress')
    plt.show()

# During training, collect losses
training_losses = []
validation_losses = []

# Plot after training
plot_training_progress(training_losses, validation_losses)
```

### Feature Space Visualization

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_learned_features(model, X_data, y_data):
    model.eval()

    with torch.no_grad():
        # Extract features from the network
        features = model.feature_extractor(torch.tensor(X_data, dtype=torch.float32))
        features_np = features.numpy()

    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_np)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y_data, cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Learned Feature Space (2D Projection)')
    plt.show()

visualize_learned_features(model, X_test, y_test)
```

## 🔬 Advanced Techniques

### Variational Sparse GP

```python
# For large datasets, use sparse approximations
model = create_gp_model(
    'deep',
    hidden_dims=[64, 32],
    inducing_points=100,  # Sparse approximation
    kernel_type='rbf'
)
```

### Multi-Output Deep GP

```python
# For multiple properties simultaneously
class MultiOutputDeepGP(gpytorch.models.ApproximateGP):
    def __init__(self, num_outputs=2):
        # Implementation for multiple outputs
        pass

# Example: Predict both PERMX and PORO
multi_model = MultiOutputDeepGP(num_outputs=2)
```

### Hierarchical Deep GP

```python
# Multiple GP layers
class HierarchicalDeepGP(gpytorch.models.ApproximateGP):
    def __init__(self, hidden_dims_list):
        # Multiple Deep GP layers
        self.gp_layers = torch.nn.ModuleList([
            create_gp_model('deep', hidden_dims=dims)
            for dims in hidden_dims_list
        ])
```

## 📈 Performance Optimization

### GPU Acceleration

```python
# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
likelihood = likelihood.to(device)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
```

### Memory Optimization

```python
# Use gradient checkpointing for large models
import torch.utils.checkpoint as checkpoint

class MemoryEfficientDeepGP(torch.nn.Module):
    def forward(self, x):
        # Use checkpointing to save memory
        return checkpoint.checkpoint(self.feature_extractor, x)
```

### Numerical Stability

```python
# Add jitter for numerical stability
model.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
model.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)

# Use double precision if needed
model = model.double()
likelihood = likelihood.double()
```

## 🐛 Troubleshooting

### Common Issues

#### Poor Convergence

```python
# Try different learning rates
optimizer = Adam(model.parameters(), lr=0.01)  # Lower LR

# Add learning rate scheduling
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
```

#### Overfitting

```python
# Add dropout
class RegularizedDeepGP(torch.nn.Module):
    def __init__(self, hidden_dims, dropout_rate=0.2):
        super().__init__()
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.extend([
                torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate)
            ])
        self.feature_extractor = torch.nn.Sequential(*layers)
```

#### Memory Issues

```python
# Use smaller batch sizes
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use gradient accumulation
accumulation_steps = 4
for i, (batch_x, batch_y) in enumerate(dataloader):
    loss = compute_loss(batch_x, batch_y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 💡 Best Practices

1. **Start Simple**: Begin with small networks and increase complexity gradually
2. **Monitor Training**: Track both training and validation losses
3. **Use Early Stopping**: Prevent overfitting with patience-based stopping
4. **Regularize**: Apply dropout or weight decay for better generalization
5. **Validate Properly**: Use spatial cross-validation for geostatistical data
6. **Save Checkpoints**: Save model states during long training runs
7. **Experiment**: Try different architectures and hyperparameters

## 🔄 Integration with Toolkit

### Using with UnifiedSPE9Toolkit

```python
from spe9_geomodeling import UnifiedSPE9Toolkit

# Create toolkit with GPyTorch backend
toolkit = UnifiedSPE9Toolkit(backend='gpytorch')

# Create and train Deep GP
model = toolkit.create_gpytorch_model('deep', hidden_dims=[64, 32])
likelihood = gpytorch.likelihoods.GaussianLikelihood()

toolkit.train_gpytorch_model(model, likelihood, 'deep_gp', training_iter=100)

# Evaluate
results = toolkit.evaluate_model('deep_gp', X_test, y_test)
print(f"Deep GP R² Score: {results.r2:.4f}")
```

---

**Next Steps**:

- Use the built-in plotting utilities in the `SPE9Plotter` class for visualizing Deep GP results
- For large datasets, consider using batch processing and GPU acceleration with GPyTorch
- See [API Reference](api.md) for complete function documentation
