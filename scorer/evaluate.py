"""
投资组合优化评分脚本（Judge端）

读取Agent提交的weights.csv，用隐藏的测试集收益率计算样本外夏普比率。
输出格式兼容score_sum parser: TOTAL_SCORE <score>
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

EVAL_DATA = Path(__file__).parent / "eval-data"
CWD = Path("/home/workspace/portfolio_opt")
TRANSACTION_COST = 0.001


def evaluate():
    returns_path = EVAL_DATA / "returns_test.csv"
    weights_path = CWD / "weights.csv"

    if not weights_path.exists():
        print("CASE 0001 FAIL score=0")
        print("TOTAL_SCORE 0")
        print(f"ERROR: weights.csv not found at {weights_path}", file=sys.stderr)
        return

    try:
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    except Exception as e:
        print("CASE 0001 FAIL score=0")
        print("TOTAL_SCORE 0")
        print(f"ERROR: Failed to read files: {e}", file=sys.stderr)
        return

    common_dates = returns.index.intersection(weights.index)
    if len(common_dates) < 10:
        print("CASE 0001 FAIL score=0")
        print("TOTAL_SCORE 0")
        print(f"ERROR: Not enough overlapping dates ({len(common_dates)})", file=sys.stderr)
        return

    returns = returns.loc[common_dates]
    weights = weights.loc[common_dates]

    assets = [c for c in returns.columns if c in weights.columns]
    if len(assets) < 5:
        print("CASE 0001 FAIL score=0")
        print("TOTAL_SCORE 0")
        print(f"ERROR: Not enough overlapping assets ({len(assets)})", file=sys.stderr)
        return

    returns = returns[assets]
    weights = weights[assets]

    # 约束检查
    penalty = 1.0
    weight_sums = weights.sum(axis=1)
    if (weight_sums - 1.0).abs().max() > 0.05:
        penalty *= 0.5

    if (weights < -0.001).any().any():
        penalty *= 0.5

    # 换手率
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()

    # 组合收益
    gross_returns = (returns * weights).sum(axis=1)
    costs = turnover * TRANSACTION_COST
    net_returns = gross_returns - costs

    # 夏普比率
    ann_return = net_returns.mean() * 252
    ann_vol = net_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 1e-8 else 0

    # 最大回撤
    cum = (1 + net_returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    # 评分: sharpe * 10, 上限100
    sharpe_score = min(100, max(0, sharpe * 10)) * penalty
    total = round(sharpe_score, 2)

    # 输出格式：用100个CASE编码分数，pass_rate即为分数/100
    n_passed = int(round(total))
    n_failed = 100 - n_passed
    for i in range(n_passed):
        print(f"CASE {i:04d} OK score=1")
    for i in range(n_passed, 100):
        print(f"CASE {i:04d} WA score=0")
    print(f"TOTAL_SCORE {total}")
    print(f"CASES_OK {n_passed}")
    print(f"CASES_TOTAL 100")

    # 详细信息输出到stderr（Agent可见作为反馈）
    print(f"\n--- Portfolio Metrics ---", file=sys.stderr)
    print(f"Sharpe Ratio: {sharpe:.4f}", file=sys.stderr)
    print(f"Annual Return: {ann_return*100:.2f}%", file=sys.stderr)
    print(f"Annual Volatility: {ann_vol*100:.2f}%", file=sys.stderr)
    print(f"Max Drawdown: {max_dd*100:.2f}%", file=sys.stderr)
    print(f"Avg Daily Turnover: {turnover.mean():.4f}", file=sys.stderr)
    print(f"Total Transaction Cost: {costs.sum()*100:.2f}%", file=sys.stderr)
    print(f"Score: {total}/100", file=sys.stderr)


if __name__ == "__main__":
    evaluate()
