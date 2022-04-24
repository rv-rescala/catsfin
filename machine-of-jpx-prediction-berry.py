# %% [code]
import joblib
from pathlib import Path
import os
import numpy as np
import pandas as pd
#import jpx_tokyo_market_prediction
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T10:21:38.211350Z","iopub.execute_input":"2022-04-22T10:21:38.211752Z","iopub.status.idle":"2022-04-22T10:21:38.234707Z","shell.execute_reply.started":"2022-04-22T10:21:38.211720Z","shell.execute_reply":"2022-04-22T10:21:38.233675Z"}}
# I/O Functinons
MODEL_NAME = "Berry"
BASE_OUTPUT_PATH = Path(f'/kaggle/working')
BASE_INPUT_PATH = Path(f'../input/jpx-tokyo-stock-exchange-prediction')

if os.environ.get("USER") == "rv":
    print("local dev mode")
    BASE_OUTPUT_PATH = Path("./output")
    BASE_INPUT_PATH = Path(f'./input/jpx-tokyo-stock-exchange-prediction')

print(BASE_OUTPUT_PATH)
print(BASE_INPUT_PATH)

def adjusting_price(df_raw, key: str):
    """[Adjusting Price/価格は単元の変化で大きく変化することがある、そのためVolumeとPriceから連続性のある値に変換する]
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """

    def generate_adjusted(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, f"CumulativeAdjustmentFactor{key}"] = df["AdjustmentFactor"].cumprod(
        )
        # generate AdjustedClose
        df.loc[:, f"Adjusted{key}"] = (
            df[f"CumulativeAdjustmentFactor{key}"] * df[key]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df[f"Adjusted{key}"] == 0, f"Adjusted{key}"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, f"Adjusted{key}"] = df.loc[:, f"Adjusted{key}"].ffill()
        return df

    # generate AdjustedClose
    df_raw = df_raw.sort_values(["SecuritiesCode", "Date"])
    df_raw = df_raw.groupby("SecuritiesCode").apply(
        generate_adjusted).reset_index(drop=True)

    # price.set_index("Date", inplace=True)
    return df_raw


def adjusting_volume(df_raw, key="Volume"):
    """[Adjusting Close Price/単元の変化を吸収する]
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """

    def generate_adjusted(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, f"CumulativeAdjustmentFactor{key}"] = df["AdjustmentFactor"].cumprod(
        )
        # generate AdjustedClose
        df.loc[:, f"Adjusted{key}"] = (
            df[key] / df[f"CumulativeAdjustmentFactor{key}"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df[f"Adjusted{key}"] == 0, f"Adjusted{key}"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, f"Adjusted{key}"] = df.loc[:, f"Adjusted{key}"].ffill()
        return df

    # generate AdjustedClose
    df_raw = df_raw.sort_values(["SecuritiesCode", "Date"])
    df_raw = df_raw.groupby("SecuritiesCode").apply(
        generate_adjusted).reset_index(drop=True)

    # price.set_index("Date", inplace=True)
    return price


def read_prices(dir_name: str, securities_code: int = None):
    """[Important: the dateset of 2020/10/1 is lost because of system failer in JPX, see: https://www.jpx.co.jp/corporate/news/news-releases/0060/20201019-01.html]

    """
    path = BASE_INPUT_PATH / f"{dir_name}/stock_prices.csv"
    print(f"read_prices: {path}")
    df = pd.read_csv(path)
    df = df[df['Open'].notna()]
    if securities_code:
        df = df[df["SecuritiesCode"] == securities_code]
    return df

def read_stock_list(securities_code: int = None, only_universe: bool = True):
    """_summary_

    Args:
        securities_code (int, optional): _description_. Defaults to None.
        only_universe (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    path = BASE_INPUT_PATH / 'stock_list.csv'
    print("read_stock_list: {path}")
    df = pd.read_csv(path)
    if only_universe:
        df = df[df['Universe0']]
    if securities_code:
        df = df[df["SecuritiesCode"] == securities_code]
    return df

def merge_data_by_price(prices, stock_list):
    """[処理ベースとなるデータセットを作成]

    Args:
        prices (_type_): _description_
        stock_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    # stock_prices がベース
    base_df = prices.copy()

    # stock_listと結合
    _stock_list = stock_list.copy()
    _stock_list.rename(columns={'Close': 'Close_x'}, inplace=True)
    base_df = base_df.merge(_stock_list, on='SecuritiesCode', how="left")
    # format
    base_df.loc[:, "Date"] = pd.to_datetime(
        base_df.loc[:, "Date"], format="%Y-%m-%d")
    base_df.loc[:, "EffectiveDate"] = pd.to_datetime(
        base_df.loc[:, "EffectiveDate"], format="%Y%m%d")
    return base_df

def read_submission_test():
    """
    api iteratorは一度しか実行されないため、submissionからデータを読み取る
    # prices, options, financials, trades, secondary_prices, sample_prediction
    # Targetはtrain/pricesにあるため、train_filesから取得する
    """
    prices = pd.read_csv(
        BASE_INPUT_PATH / 'supplemental_files/stock_prices.csv')
    options = pd.read_csv(BASE_INPUT_PATH / 'example_test_files/options.csv')
    financials = pd.read_csv(
        BASE_INPUT_PATH / 'example_test_files/financials.csv')
    trades = pd.read_csv(BASE_INPUT_PATH / 'example_test_files/trades.csv')
    secondary_prices = pd.read_csv(
        BASE_INPUT_PATH / 'example_test_files/secondary_stock_prices.csv')
    sample_prediction = pd.read_csv(
        BASE_INPUT_PATH / 'example_test_files/sample_submission.csv')
    dates = options["Date"].unique()
    print(f"submission target date: {dates}")
    for d in dates:
        target_prices = prices[prices["Date"] == d]
        target_options = options[options["Date"] == d]
        target_financials = financials[financials["Date"] == d]
        target_trades = trades[trades["Date"] == d]
        target_secondary_prices = secondary_prices[secondary_prices["Date"] == d]
        target_sample_prediction = sample_prediction[sample_prediction["Date"] == d]
        yield target_prices, target_options, target_financials, target_trades, target_secondary_prices, target_sample_prediction

def read_train_data(securities_code: int = None, with_supplemental: bool = True):
    """[The train base is price dataset, the other data are joined to prices DF by left join]

    """
    # origin
    df = merge_data_by_price(prices=read_prices(dir_name="train_files", securities_code=securities_code),
                    stock_list=read_stock_list(securities_code=securities_code))

    # supplyment
    if with_supplemental:
        supplemental_df = merge_data_by_price(prices=read_prices(
            dir_name="supplemental_files", securities_code=securities_code), stock_list=read_stock_list(securities_code=securities_code))
        df = pd.concat([df, supplemental_df]).reset_index(drop=True)

    # historical
    df = adjusting_price(df, "Close")
    df = adjusting_price(df, "Open")
    df = adjusting_price(df, "High")
    df = adjusting_price(df, "Low")
    df = adjusting_volume(df)
    return df

def create_inference_data(additional_prices, stock_list):
    """[推論フェーズで使用するデータ]

    Args:
        additional_prices (_type_): 追加されたprice
        stock_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = merge_data(prices, stock_list)
    # AdjustedClose項目の生成
    df = adjusting_price(df, "Close")
    df = adjusting_price(df, "Open")
    df = adjusting_price(df, "High")
    df = adjusting_price(df, "Low")
    df = adjusting_volume(df)
    return df

def write_df(df, filename):
    """_summary_

    Args:
        df (_type_): _description_
        filename (_type_): _description_
    """
    path = '{BASE_OUTPUT_PATH}/{filename}_{MODEL_NAME}.csv'
    print(f"write_df: {path}")
    df.to_csv(path, index=False)
    return path

def write_model(model, name):
    """[write trained model]

    Args:
        model (_type_): _description_
        name (_type_): _description_
    """
    # save model
    path = f'{BASE_OUTPUT_PATH}/{name}_{MODEL_NAME}.pkl'
    print(f"write_model: {path}")
    joblib.dump(model, path)

# load model
def read_model(name):
    """[read trained model]

    Args:
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    path = f'{BASE_OUTPTU_PATH}/{name}_{MODEL_NAME}.pkl'
    print(f"read_model: {read_model}")
    return joblib.load(path)


# %% [code]
# Evaluate
# 予測値を降順に並べて順位番号を振る関数
# 言い換えると、目的変数から提出用項目を導出する関数
def add_rank(df, col_name="Prediction"):
    df["Rank"] = df.groupby("Date")[col_name].rank(
        ascending=False, method="first") - 1
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
        weights = np.linspace(start=toprank_weight_ratio,
                              stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')[
                    'Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)[
                 'Target'][:portfolio_size] * weights).sum() / weights.mean()
        print(f"purchase: {purchase}")
        print(f"short: {short}")
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day,
                                   portfolio_size, toprank_weight_ratio)
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
    df = df[["Date", "SecuritiesCode", "Avg"]]
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
    # sample_prediction.drop(["Prediction"],axis=1)
    #submission = df[["Date","SecuritiesCode","Rank"]]


# %%

# %%

# %%

# %%

# %%

# %%
