"""等权组合Baseline"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def main():
    train = pd.read_csv(DATA_DIR / "returns_train.csv", index_col=0, parse_dates=True)
    n_assets = train.shape[1]

    # 生成测试期日期（训练集最后一天之后250个交易日）
    last_date = train.index[-1]
    test_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=250)

    # 等权
    weights = pd.DataFrame(
        np.ones((250, n_assets)) / n_assets,
        index=test_dates,
        columns=train.columns,
    )
    weights.index.name = "date"
    weights.to_csv(Path(__file__).parent / "weights.csv")
    print(f"Generated equal-weight portfolio: {weights.shape}")

if __name__ == "__main__":
    main()
