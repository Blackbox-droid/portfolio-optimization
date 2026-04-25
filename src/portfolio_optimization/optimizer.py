"""Constrained mean-variance portfolio optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import cvxpy as cp  # type: ignore

    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False

from .config import (
    ASSETS,
    DEFAULT_WEIGHT_BOUNDS,
    MV_RISK_AVERSION,
    REGIME_WEIGHT_BOUNDS,
)


def _nearest_psd(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eig_values, eig_vectors = np.linalg.eigh(sym)
    eig_values = np.clip(eig_values, epsilon, None)
    return (eig_vectors * eig_values) @ eig_vectors.T


def _bounds_arrays(
    assets: list[str],
    bounds_map: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    lb = np.array([bounds_map[a][0] for a in assets], dtype=float)
    ub = np.array([bounds_map[a][1] for a in assets], dtype=float)
    if lb.sum() > 1.0 + 1e-9:
        raise ValueError(f"Lower bounds sum to {lb.sum():.4f} > 1; infeasible.")
    if ub.sum() < 1.0 - 1e-9:
        raise ValueError(f"Upper bounds sum to {ub.sum():.4f} < 1; infeasible.")
    return lb, ub


def _solve_cvxpy(
    mu: np.ndarray,
    sigma: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    n = mu.shape[0]
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma)))
    constraints = [cp.sum(w) == 1, w >= lb, w <= ub]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        problem.solve(verbose=False)
    if w.value is None:
        raise RuntimeError("cvxpy failed to find a feasible solution")
    weights = np.array(w.value).flatten()
    weights = np.clip(weights, lb, ub)
    weights = weights / weights.sum()
    return weights


def _solve_projected_gradient(
    mu: np.ndarray,
    sigma: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    risk_aversion: float,
    max_iter: int = 5000,
    lr: float = 0.05,
) -> np.ndarray:
    n = mu.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        grad = mu - risk_aversion * sigma @ w
        w = w + lr * grad
        w = np.clip(w, lb, ub)
        total = w.sum()
        if total > 0:
            residual = 1.0 - total
            if abs(residual) > 1e-10:
                if residual > 0:
                    slack = ub - w
                else:
                    slack = w - lb
                slack_sum = slack.sum()
                if slack_sum > 0:
                    w = w + np.sign(residual) * (residual * slack / slack_sum)
            w = np.clip(w, lb, ub)
    return w / w.sum()


def markowitz_weights(
    expected_returns: pd.Series | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
    assets: list[str] | None = None,
    weight_bounds: dict[str, tuple[float, float]] | None = None,
    risk_aversion: float = MV_RISK_AVERSION,
) -> pd.Series:
    assets = assets or ASSETS
    if isinstance(expected_returns, pd.Series):
        mu = expected_returns.reindex(assets).values.astype(float)
    else:
        mu = np.asarray(expected_returns, dtype=float)

    if isinstance(covariance, pd.DataFrame):
        sigma = covariance.reindex(index=assets, columns=assets).values.astype(float)
    else:
        sigma = np.asarray(covariance, dtype=float)

    sigma = _nearest_psd(sigma)
    bounds_map = weight_bounds or DEFAULT_WEIGHT_BOUNDS
    lb, ub = _bounds_arrays(assets, bounds_map)

    if _HAS_CVXPY:
        weights = _solve_cvxpy(mu, sigma, lb, ub, risk_aversion)
    else:
        weights = _solve_projected_gradient(mu, sigma, lb, ub, risk_aversion)

    return pd.Series(weights, index=assets, name="weight")


def equal_weight_portfolio(assets: list[str] | None = None) -> pd.Series:
    assets = assets or ASSETS
    w = 1.0 / len(assets)
    return pd.Series([w] * len(assets), index=assets, name="weight")


def regime_aware_weights(
    regime: str | None,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    assets: list[str] | None = None,
    risk_aversion: float = MV_RISK_AVERSION,
) -> pd.Series:
    bounds = REGIME_WEIGHT_BOUNDS.get(regime, DEFAULT_WEIGHT_BOUNDS)
    return markowitz_weights(
        expected_returns,
        covariance,
        assets=assets,
        weight_bounds=bounds,
        risk_aversion=risk_aversion,
    )
