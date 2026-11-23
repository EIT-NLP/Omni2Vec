from __future__ import annotations
import numpy as np

def fit_directions(features, labels, shrinkage=.1):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(y) != len(x) or set(y.tolist()) != set(range(4)):
        raise ValueError('Training candidates must cover four modalities')
    means = np.stack([x[y == m].mean(0) for m in range(4)])
    center = means.mean(0)
    residual = x - means[y]
    covariance = residual.T @ residual / len(x)
    ridge = shrinkage * np.trace(covariance) / x.shape[1]
    if ridge <= 0:
        raise ValueError('LDA requires nonzero within-class covariance')
    covariance *= 1 - shrinkage
    covariance.flat[::len(covariance) + 1] += ridge
    directions = np.linalg.solve(covariance, (means - center).T).T
    intercept = -.5 * np.einsum('md,md->m', directions, means + center)
    scores = x @ directions.T
    separation = np.array([scores[y == m, m].mean() - scores[y != m, m].mean() for m in range(4)])
    if (separation <= 0).any():
        raise ValueError('Training-only target-versus-rest separation must be positive')
    return directions / separation[:, None], directions, intercept, separation
