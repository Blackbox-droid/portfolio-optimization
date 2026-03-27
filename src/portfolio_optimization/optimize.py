import numpy as np

from .config import ASSETS, REGIME_BOUNDS, RISK_AVERSION_GAMMA

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - handled at runtime
    cp = None


def _equal_weight_with_bounds(bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    weights = np.array([1.0 / len(ASSETS)] * len(ASSETS), dtype=float)
    for idx, asset in enumerate(ASSETS):
        lb, ub = bounds[asset]
        weights[idx] = np.clip(weights[idx], lb, ub)
    total = weights.sum()
    return weights / total if total else np.array([1.0 / len(ASSETS)] * len(ASSETS), dtype=float)


def solve_markowitz(
    mu_vec: np.ndarray,
    cov_mat: np.ndarray,
    bounds: dict[str, tuple[float, float]] | None = None,
    gamma: float = RISK_AVERSION_GAMMA,
) -> np.ndarray:
    bounds = bounds or {asset: (0.0, 1.0) for asset in ASSETS}

    if cp is None:
        return _equal_weight_with_bounds(bounds)

    n_assets = len(mu_vec)
    weights = cp.Variable(n_assets)
    objective = cp.Maximize(mu_vec @ weights - (gamma / 2.0) * cp.quad_form(weights, cov_mat))
    constraints = [cp.sum(weights) == 1]

    for idx, asset in enumerate(ASSETS):
        lb, ub = bounds[asset]
        constraints.extend([weights[idx] >= lb, weights[idx] <= ub])

    problem = cp.Problem(objective, constraints)
    solvers = [cp.CLARABEL, cp.ECOS, cp.SCS]
    for solver in solvers:
        try:
            problem.solve(solver=solver, verbose=False)
            if weights.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
                clipped = np.clip(np.array(weights.value).astype(float), 0.0, 1.0)
                return clipped / clipped.sum()
        except Exception:
            continue

    return _equal_weight_with_bounds(bounds)


def static_classical_markowitz_weights(
    train_simple_returns,
    gamma: float = RISK_AVERSION_GAMMA,
) -> np.ndarray:
    mu = train_simple_returns.mean().reindex(ASSETS).values.astype(float)
    cov = train_simple_returns[ASSETS].cov().values.astype(float)
    bounds = {asset: (0.0, 1.0) for asset in ASSETS}
    return solve_markowitz(mu, cov, bounds=bounds, gamma=gamma)


def regime_bounds_for(regime: str) -> dict[str, tuple[float, float]]:
    return REGIME_BOUNDS.get(regime, REGIME_BOUNDS["Growth"])

