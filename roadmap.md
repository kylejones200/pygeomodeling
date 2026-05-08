
---

## Roadmap for *pygeomodeling*

### Expand Model Types

~~Add **multi-output GP** to model multiple correlated spatial fields simultaneously.~~ ✅ Implemented
~~Add **non-stationary kernels** and **spectral mixture kernels**.~~ ✅ Implemented
~~Add **sparse GP** approximations to handle very large grids (>1M cells).~~ ✅ Implemented
~~Add **neural network Gaussian processes** or hybrid deep architectures.~~ ✅ Implemented
~~Add **Bayesian neural network** surrogate models for comparison.~~ ✅ Implemented

---

## Performance and Scaling

Implement **memory-mapping** for extremely large GRDECL files that don't fit in RAM.
Benchmark model training time and accuracy on standard synthetic spatial data.
Add comprehensive performance profiling tools.

---

## Data Input and Format Support

~~Add support for **additional reservoir formats** like Petrel, RESQML, and ASCII grids.~~ ✅ Implemented
~~Standardize coordinate reference system handling with **pyproj** for all spatial operations.~~ ✅ Implemented

---

## Validation and Error Metrics

Add **spatial residual maps** and **uncertainty surfaces** diagnostics.
Provide **Monte Carlo cross-validation** and error decomposition.
Add **log-likelihood and AIC/BIC scoring** for model comparison.

---

## Visualization

Add **interactive 3D plots** with Plotly or PyVista.
Add **field slice animation** utilities.
Add **uncertainty contour plotting** for spatial prediction.

---

## High-Level APIs

Provide a **pipeline utility** to chain parsing, splitting, training, evaluating.
Add a **single CLI tool** to train and predict from a config file.

---

## Tutorials and Examples

Add **colab notebooks** that load real public reservoir datasets.
Add **comparison notebooks** between GP, deep GP, and kriging.

---

## Documentation and Guides

Add a **model comparison guide** that describes when to use each algorithm.

---

## Testing and Quality

Increase **unit coverage** for covariance functions and optimizer routines.
Benchmark training on CPU vs GPU for regressions.

---

## Interoperability

Add export to **standard geostat formats** like GSLIB.
Provide **PyTorch Lightning** integration.

---

## Feature Flags and Packaging

Add package to **conda-forge**.

---

## Real Use Cases

Add **uncertainty propagation** workflows for reservoir decisions.
Document **field case studies** showing value for engineering teams.

