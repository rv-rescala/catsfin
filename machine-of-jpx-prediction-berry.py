# %% [code]
import numpy as np
import pandas as pd
#import jpx_tokyo_market_prediction
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# %% [code]
prices = pd.read_csv("./input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")

# %% [code]
prices.isnull().sum()

# %% [code]
average = pd.DataFrame(prices.groupby("SecuritiesCode").Target.mean())
def get_avg(_id_):
    return average.loc[_id_]
prices["Avg"] = prices["SecuritiesCode"].apply(get_avg)

# %% [code]
prices.Date = pd.to_datetime(prices.Date)
prices['Date'] = prices['Date'].dt.strftime("%Y%m%d").astype(int)
X=prices[["Date","SecuritiesCode","Avg"]]
y=prices[["Target"]]
codes = X.SecuritiesCode.unique()

# %% [code]
model=LGBMRegressor(num_leaves=500, learning_rate=0.05, n_estimators=100)
model.fit(X,y)
model.score(X,y)

# %% [code]
# I/O Func
import os
from pathlib import Path
MODEL_NAME = "BERRY"
BASE_OUTPUT_PATH = Path(f'/kaggle/working')
BASE_INPUT_PATH = Path(f'../input/jpx-tokyo-stock-exchange-prediction')

if os.environ.get("USER") == "rv":
    BASE_OUTPUT_PATH = Path("./output")
    BASE_INPUT_PATH = Path(f'./input/jpx-tokyo-stock-exchange-prediction')

print(BASE_OUTPUT_PATH)
print(BASE_INPUT_PATH)

def load_submission_test():
    # prices, options, financials, trades, secondary_prices, sample_prediction
    # pricesだけはTarget取得のため、train_filesから取得する
    prices = pd.read_csv(BASE_INPUT_PATH / 'supplemental_files/stock_prices.csv')
    options = pd.read_csv(BASE_INPUT_PATH / 'example_test_files/options.csv')
    financials = pd.read_csv(BASE_INPUT_PATH / 'example_test_files/financials.csv')
    trades = pd.read_csv(BASE_INPUT_PATH / 'example_test_files/trades.csv')
    secondary_prices = pd.read_csv(BASE_INPUT_PATH / 'example_test_files/secondary_stock_prices.csv')
    sample_prediction = pd.read_csv(BASE_INPUT_PATH / 'example_test_files/sample_submission.csv')
    dates = options["Date"].unique()
    for d in dates:
        target_prices = prices[prices["Date"] == d]
        target_options = options[options["Date"] == d]
        target_financials = financials[financials["Date"] == d]
        target_trades = trades[trades["Date"] == d]
        target_secondary_prices = secondary_prices[secondary_prices["Date"] == d]
        target_sample_prediction= sample_prediction[sample_prediction["Date"] == d]
        yield target_prices, target_options, target_financials, target_trades, target_secondary_prices, target_sample_prediction
    #print(dates)
    #return prices, options, financials, trades, secondary_prices, sample_prediction



# %% [code]
# Evaluate
# 予測値を降順に並べて順位番号を振る関数
# 言い換えると、目的変数から提出用項目を導出する関数
def add_rank(df, col_name="Prediction"):
    df["Rank"] = df.groupby("Date")[col_name].rank(ascending=False, method="first") - 1 
    df["Rank"] = df["Rank"].astype("int")
    return df

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        #print(f"date: {df['Date'].min()}")
        print(f"min: {df['Rank'].min()}")
        print(f"max: {df['Rank'].max()}")
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        print(f"purchase: {purchase}")
        print(f"short: {short}")
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    print(f"mean: {buf.mean()}")
    print(f"std: {buf.std()}")
    sharpe_ratio = buf.mean() / buf.std()
    print(f"sharpe_ratio: {sharpe_ratio}")
    return sharpe_ratio

# 予測用のデータフレームと、予測結果をもとに、スコアを計算する関数
def evaluator(df):
    df = add_rank(df)
    score = calc_spread_return_sharpe(df)
    return score

# Test
tests = load_submission_test()
for (prices, options, financials, trades, secondary_prices, sample_prediction) in tests:
    date = prices["Date"].unique()
    print(f"target_date: {date}")
    df["Avg"] = sample_prediction["SecuritiesCode"].apply(get_avg)
    df = df[["Date","SecuritiesCode","Avg"]]
    df.Date = pd.to_datetime(df.Date)
    df['Date'] = df['Date'].dt.strftime("%Y%m%d").astype(int)
    df["Prediction"] = model.predict(df)
    
    # for test adding Target
    target_df = prices[["Target", "SecuritiesCode"]]
    result_df = df.merge(target_df, on='SecuritiesCode', how="left")
    score = evaluator(df=result_df)
    print(f"score {date}: {score}")

    #sample_prediction = sample_prediction.sort_values(by = "Prediction", ascending=False)
    #sample_prediction.Rank = np.arange(0,2000)
    #sample_prediction = sample_prediction.sort_values(by = "SecuritiesCode", ascending=True)
    #sample_prediction.drop(["Prediction"],axis=1)
    #submission = df[["Date","SecuritiesCode","Rank"]]



# %%

# %%

# %%

# %%

# %%
