"""
本地评测工具（Agent开发用）
用训练集后200天模拟测试，前800天作为训练。
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TRANSACTION_COST = 0.001


def main():
    weights_path = Path(__file__).parent.parent / "weights.csv"
    if not weights_path.exists():
        print("ERROR: weights.csv not found. Run your solution first.")
        sys.exit(1)

    train = pd.read_csv(DATA_DIR / "returns_train.csv", index_col=0, parse_dates=True)

    # 用后200天做本地验证
    local_test = train.iloc[800:]
    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)

    # 对齐日期
    common = local_test.index.intersection(weights.index)
    if len(common) == 0:
        print("WARNING: No overlapping dates with local test period.")
        print("Local test period:", local_test.index[0], "to", local_test.index[-1])
        print("Using full weights for evaluation against actual test set.")
        return

    ret = local_test.loc[common]
    w = weights.loc[common]
    assets = [c for c in ret.columns if c in w.columns]
    ret = ret[assets]
    w = w[assets]

    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()

    gross = (ret * w).sum(axis=1)
    costs = turnover * TRANSACTION_COST
    net = gross - costs

    ann_ret = net.mean() * 252
    ann_vol = net.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0

    cum = (1 + net).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    score = min(90, max(0, sharpe * 15))
    w_sum_ok = (w.sum(axis=1) - 1.0).abs().max() <= 0.05
    w_pos_ok = not (w < -0.001).any().any()
    score += 5 * w_sum_ok + 5 * w_pos_ok

    print(f"=== Local Evaluation (train[800:]) ===")
    print(f"Days: {len(common)}")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"Annual Return: {ann_ret*100:.2f}%")
    print(f"Annual Vol: {ann_vol*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print(f"Avg Turnover: {turnover.mean():.4f}")
    print(f"Score: {score:.2f}/100")
    print(f"\nNOTE: This is local validation only. Final score uses hidden test data.")


if __name__ == "__main__":
    main()
