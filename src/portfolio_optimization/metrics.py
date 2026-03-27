import numpy as np
import pandas as pd


def log_to_simple(log_returns: pd.DataFrame | pd.Series):
    return np.exp(log_returns) - 1.0


def cumulative_wealth(simple_returns: pd.Series) -> pd.Series:
    return (1.0 + simple_returns).cumprod()


def drawdown_series(simple_returns: pd.Series) -> pd.Series:
    wealth = cumulative_wealth(simple_returns)
    running_max = wealth.cummax()
    return wealth / running_max - 1.0


def annualized_return(simple_returns: pd.Series) -> float:
    simple_returns = simple_returns.dropna()
    if simple_returns.empty:
        return np.nan
    total_wealth = float((1.0 + simple_returns).prod())
    num_months = simple_returns.shape[0]
    return total_wealth ** (12.0 / num_months) - 1.0


def annualized_volatility(simple_returns: pd.Series) -> float:
    return float(simple_returns.std(ddof=1) * np.sqrt(12.0))


def sharpe_ratio(simple_returns: pd.Series) -> float:
    ann_return = annualized_return(simple_returns)
    ann_vol = annualized_volatility(simple_returns)
    if np.isnan(ann_vol) or ann_vol == 0.0:
        return np.nan
    return ann_return / ann_vol


def sortino_ratio(simple_returns: pd.Series) -> float:
    downside = simple_returns[simple_returns < 0.0]
    if downside.empty:
        return np.nan
    downside_vol = float(downside.std(ddof=1) * np.sqrt(12.0))
    ann_return = annualized_return(simple_returns)
    if np.isnan(downside_vol) or downside_vol == 0.0:
        return np.nan
    return ann_return / downside_vol


def max_drawdown(simple_returns: pd.Series) -> float:
    return float(drawdown_series(simple_returns).min())


def summarize_strategy(simple_returns: pd.Series) -> dict[str, float]:
    wealth = cumulative_wealth(simple_returns)
    return {
        "Total_Return": float(wealth.iloc[-1] - 1.0),
        "Annualized_Return": annualized_return(simple_returns),
        "Annualized_Volatility": annualized_volatility(simple_returns),
        "Sharpe_Ratio": sharpe_ratio(simple_returns),
        "Sortino_Ratio": sortino_ratio(simple_returns),
        "Max_Drawdown": max_drawdown(simple_returns),
        "Num_Months": int(simple_returns.dropna().shape[0]),
    }
