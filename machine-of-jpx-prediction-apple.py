# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T10:21:37.020294Z","iopub.execute_input":"2022-04-22T10:21:37.020949Z","iopub.status.idle":"2022-04-22T10:21:38.209725Z","shell.execute_reply.started":"2022-04-22T10:21:37.020839Z","shell.execute_reply":"2022-04-22T10:21:38.208989Z"}}
import os
from pathlib import Path
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T10:21:38.211350Z","iopub.execute_input":"2022-04-22T10:21:38.211752Z","iopub.status.idle":"2022-04-22T10:21:38.234707Z","shell.execute_reply.started":"2022-04-22T10:21:38.211720Z","shell.execute_reply":"2022-04-22T10:21:38.233675Z"}}
# I/O Func
MODEL_NAME = "APPLE"
BASE_PATH = Path(f'/kaggle/working')
if os.environ.get("USER") == "rv":
    BASE_PATH = Path("./output")

def adjusting_price(price, key: str):
    """[Adjusting Close Price]
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
        df.loc[:, f"CumulativeAdjustmentFactor{key}"] = df["AdjustmentFactor"].cumprod()
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
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted).reset_index(drop=True)

    # price.set_index("Date", inplace=True)
    return price

def adjusting_volume(price, key = "Volume"):
    """[Adjusting Close Price]
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
        df.loc[:, f"CumulativeAdjustmentFactor{key}"] = df["AdjustmentFactor"].cumprod()
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
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted).reset_index(drop=True)

    # price.set_index("Date", inplace=True)
    return price

def read_prices(dir_name: str, securities_code: int = None):
    """[Important: the dateset of 2020/10/1 is lost because of system failer in JPX, see: https://www.jpx.co.jp/corporate/news/news-releases/0060/20201019-01.html]
    
    """
    base_path = Path(f'../input/jpx-tokyo-stock-exchange-prediction/{dir_name}')
    df = pd.read_csv(base_path / 'stock_prices.csv')
    df = df[df['Open'].notna()]
    if securities_code:
        df = df[df["SecuritiesCode"] == securities_code]
    return df

def read_stock_list(securities_code: int = None, only_universe: bool = True):
    df = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv')
    if only_universe:
        df = df[df['Universe0']]
    if securities_code:
        df = df[df["SecuritiesCode"] == securities_code]
    return df

def merge_data(prices, stock_list):
    # stock_prices がベース
    base_df = prices.copy()
    
    # stock_listと結合
    _stock_list = stock_list.copy()
    _stock_list.rename(columns={'Close': 'Close_x'}, inplace=True)
    base_df = base_df.merge(_stock_list, on='SecuritiesCode', how="left")
    # format
    base_df.loc[: ,"Date"] = pd.to_datetime(base_df.loc[: ,"Date"], format="%Y-%m-%d")
    base_df.loc[: ,"EffectiveDate"] = pd.to_datetime(base_df.loc[: ,"EffectiveDate"], format="%Y%m%d")
    return base_df

def read_train_data_by_price(securities_code: int = None, with_supplemental: bool = True):
    """[The train base is price dataset, the other data are joined to prices DF by left join]
    
    """
    # origin
    df = merge_data(prices=read_prices(dir_name="train_files", securities_code=securities_code), stock_list=read_stock_list(securities_code=securities_code))
    
    # supplyment
    if with_supplemental:
        supplemental_df = merge_data(prices=read_prices(dir_name="supplemental_files", securities_code=securities_code), stock_list=read_stock_list(securities_code=securities_code))
        df = pd.concat([df, supplemental_df]).reset_index(drop=True)
        
    df = adjusting_price(df, "Close")
    df = adjusting_price(df, "Open")
    df = adjusting_price(df, "High")
    df = adjusting_price(df, "Low")
    df = adjusting_volume(df)
    return df

def collector(prices, options, financials, trades, secondary_prices, stock_list):
    # 読み込んだデータを統合して一つのファイルに纏める
    df = merge_data(prices, stock_list)
    # AdjustedClose項目の生成
    df = adjusting_price(df, "Close")
    df = adjusting_price(df, "Open")
    df = adjusting_price(df, "High")
    df = adjusting_price(df, "Low")
    df = adjusting_volume(df)
    return df

def write_df(df, filename):
    df.to_csv(f'{BASE_PATH}/{filename}_{MODEL_NAME}.csv',index = False)
    
import joblib
def write_model(model, name):
    # save model
    joblib.dump(model, f'{BASE_PATH}/{name}_{MODEL_NAME}.pkl')


# load model
def read_model(name):
    return joblib.load(f'{BASE_PATH}/{name}_{MODEL_NAME}.pkl')

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T10:21:38.235712Z","iopub.execute_input":"2022-04-22T10:21:38.236376Z","iopub.status.idle":"2022-04-22T10:21:38.321380Z","shell.execute_reply.started":"2022-04-22T10:21:38.236339Z","shell.execute_reply":"2022-04-22T10:21:38.319603Z"}}
stock_list = read_stock_list()
stock_list

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T10:21:38.323243Z","iopub.execute_input":"2022-04-22T10:21:38.323923Z","iopub.status.idle":"2022-04-22T10:23:28.492518Z","shell.execute_reply.started":"2022-04-22T10:21:38.323892Z","shell.execute_reply":"2022-04-22T10:23:28.491972Z"}}
train_df = read_train_data_by_price()
train_df

# %% [markdown]
# # Featrue

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T10:23:28.493472Z","iopub.execute_input":"2022-04-22T10:23:28.494148Z","iopub.status.idle":"2022-04-22T10:25:17.099357Z","shell.execute_reply.started":"2022-04-22T10:23:28.494117Z","shell.execute_reply":"2022-04-22T10:25:17.097019Z"}}
def cal_moving_average(key:str, periods):
    def func(df):
        for period in periods:
            col = f"MovingAverage{key}{period}"
            col_gap = f"{col}GapPercent"
            df[col] = df[key].rolling(period, min_periods=1).mean()
            df[col_gap] = (df[key] / df[col]) * 100.0
        return df
    return func

def cal_changing_ration(key:str, periods):
    def func(df):
        for period in periods:
            col = f"ChangingRatio{key}{period}"
            df[col] = df[key].pct_change(period) * 100
        return df
    return func

def cal_historical_vix(key: str, periods):
    def func(df):
        for period in periods:
            col = f"HistoricalVIX{key}{period}"
            df[col] = np.log(df[key]).diff().rolling(period).std()
        return df
    return func

def add_columns_per_code(df, functions):
    def func(df):
        for f in functions:
            df = f(df)
        return df
    df = df.sort_values(["SecuritiesCode", "Date"])
    df = df.groupby("SecuritiesCode").apply(func)
    df = df.reset_index(drop=True)
    return df

def add_columns_per_day(base_df):
    base_df['diff_rate1'] = (base_df['Close'] - base_df['Open']) / base_df['Close']
    base_df['diff_rate2'] = (base_df['High'] - base_df['Low']) / base_df['Close']    
    return base_df

def generate_features(df):
    base_df = df.copy()
    prev_column_names = base_df.columns
    periods = [5, 25, 75]
    functions = [
        cal_moving_average("AdjustedClose", periods),
        cal_moving_average("AdjustedOpen", periods),
        cal_moving_average("AdjustedHigh", periods),
        cal_moving_average("AdjustedLow", periods),
        cal_moving_average("AdjustedVolume", periods),
        cal_changing_ration("AdjustedClose", periods),
        cal_changing_ration("AdjustedOpen", periods),
        cal_changing_ration("AdjustedHigh", periods),
        cal_changing_ration("AdjustedLow", periods),
        cal_changing_ration("AdjustedVolume", periods),
        cal_historical_vix("AdjustedClose", periods),
        cal_historical_vix("AdjustedOpen", periods),
        cal_historical_vix("AdjustedHigh", periods),
        cal_historical_vix("AdjustedLow", periods),
        cal_historical_vix("AdjustedVolume", periods)
    ]
    
    base_df = add_columns_per_code(base_df, functions)
    base_df = add_columns_per_day(base_df)
    
    add_column_names = list(set(base_df.columns) - set(prev_column_names))
    #feats = feats[feats["HistoricalVIXAdjustedClose75"] != 0]
    return base_df, add_column_names

def select_features(feature_df, add_column_names, is_train):
    base_cols = ['RowId', 'Date', 'SecuritiesCode']
    numerical_cols = sorted(add_column_names)
    categorical_cols = ['NewMarketSegment', '33SectorCode', '17SectorCode']
    label_col = ['Target']
    feat_cols = numerical_cols + categorical_cols
    feature_df = feature_df[base_cols + feat_cols + label_col]
    feature_df[categorical_cols] = feature_df[categorical_cols].astype('category')
    if is_train:
        feature_df.dropna(inplace=True)
    else:
        feature_df[numerical_cols] = feature_df[numerical_cols].fillna(0)
        feature_df[numerical_cols] = feature_df[numerical_cols].replace([np.inf, -np.inf], 0)
    return feature_df, feat_cols, label_col

def preprocessor(base_df, is_train=True):
    feature_df = base_df.copy()
    
    ## 特徴量生成
    feature_df, add_column_names = generate_features(feature_df)
    
    ## 特徴量選択
    feature_df, feat_cols, label_col = select_features(feature_df, add_column_names, is_train)

    # 上書き
    feat_cols = ['33SectorCode', 'ChangingRatioAdjustedVolume25', 'diff_rate2', 'MovingAverageAdjustedHigh5GapPercent', 'MovingAverageAdjustedOpen5GapPercent', 'HistoricalVIXAdjustedLow5', 'MovingAverageAdjustedClose5GapPercent', 'HistoricalVIXAdjustedOpen5', 'MovingAverageAdjustedLow25GapPercent', 'ChangingRatioAdjustedVolume5', 'HistoricalVIXAdjustedOpen75', 'HistoricalVIXAdjustedVolume5', 'MovingAverageAdjustedVolume25GapPercent', 'diff_rate1', 'ChangingRatioAdjustedHigh5', 'ChangingRatioAdjustedOpen25', 'HistoricalVIXAdjustedOpen25', 'MovingAverageAdjustedClose25GapPercent', 'MovingAverageAdjustedVolume75GapPercent', 'ChangingRatioAdjustedLow25', 'ChangingRatioAdjustedLow5', 'HistoricalVIXAdjustedHigh75', 'MovingAverageAdjustedLow5GapPercent', 'ChangingRatioAdjustedClose75', 'MovingAverageAdjustedClose75', 'MovingAverageAdjustedClose75GapPercent', 'HistoricalVIXAdjustedVolume75']
    return feature_df, feat_cols, label_col

feature_df, feat_cols, label_col = preprocessor(train_df)
feat_cols

# %% [markdown]
# # Learning

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T10:25:17.100912Z","iopub.execute_input":"2022-04-22T10:25:17.101416Z","iopub.status.idle":"2022-04-22T10:25:18.832539Z","shell.execute_reply.started":"2022-04-22T10:25:17.101367Z","shell.execute_reply":"2022-04-22T10:25:18.831489Z"}}
# 予測値を降順に並べて順位番号を振る関数
# 言い換えると、目的変数から提出用項目を導出する関数
def add_rank(df, col_name="pred"):
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
        #print(f"min: {df['Rank'].min()}")
        #print(f"max: {df['Rank'].max()}")
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        #print(f"short: {short}")
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    print(f"sharpe_ratio: {sharpe_ratio}")
    return sharpe_ratio

# 予測用のデータフレームと、予測結果をもとに、スコアを計算する関数
def evaluator(df, pred):
    df["pred"] = pred
    df = add_rank(df)
    score = calc_spread_return_sharpe(df)
    return score

import lightgbm as lgb
import optuna.integration.lightgbm as lgb

# 学習を実行する関数
def trainer(feature_df, feat_cols, label_col, fold_params, seed=2022, use_cache: bool = False):
    scores = []
    models = []
    params = []
    i = 0
    for param in fold_params:
        if not use_cache:
            ################################
            # データ準備
            ################################
            train = feature_df[(param[0] <= feature_df['Date']) & (feature_df['Date'] < param[1])]
            valid = feature_df[(param[1] <= feature_df['Date']) & (feature_df['Date'] < param[2])]

            X_train = train[feat_cols]
            y_train = train[label_col]
            X_valid = valid[feat_cols]
            y_valid = valid[label_col]

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

            ################################
            # 学習
            ################################
            params = {
                'task': 'train',                   # 学習
                'boosting_type': 'gbdt',           # GBDT
                'objective': 'regression',         # 回帰
                'metric': 'rmse',                  # 損失（誤差）
                'learning_rate': 0.01,             # 学習率
                'lambda_l1': 0.5,                  # L1正則化項の係数
                'lambda_l2': 0.5,                  # L2正則化項の係数
                'num_leaves': 10,                  # 最大葉枚数
                'feature_fraction': 0.5,           # ランダムに抽出される列の割合
                'bagging_fraction': 0.5,           # ランダムに抽出される標本の割合
                'bagging_freq': 5,                 # バギング実施頻度
                'min_child_samples': 10,           # 葉に含まれる最小データ数
                'seed': seed                       # シード値
            } 

            lgb_results = {}                       
            model = lgb.train( 
                params,                            # ハイパーパラメータ
                lgb_train,                         # 訓練データ
                valid_sets=[lgb_train, lgb_valid], # 検証データ
                valid_names=['Train', 'Valid'],    # データセット名前
                num_boost_round=2000,              # 計算回数
                early_stopping_rounds=100,         # 計算打ち切り設定
                evals_result=lgb_results,          # 学習の履歴
                verbose_eval=100,                  # 学習過程の表示サイクル
            )  

            ################################
            # 結果描画
            ################################
            fig = plt.figure(figsize=(10, 4))

            # loss
            plt.subplot(1,2,1)
            loss_train = lgb_results['Train']['rmse']
            loss_test = lgb_results['Valid']['rmse']   
            plt.xlabel('Iteration')
            plt.ylabel('logloss')
            plt.plot(loss_train, label='train loss')
            plt.plot(loss_test, label='valid loss')
            plt.legend()

            # feature importance
            plt.subplot(1,2,2)
            importance = pd.DataFrame({'feature':feat_cols, 'importance':model.feature_importance()})
            write_df(importance, f"importance_{i}")
            sns.barplot(x = 'importance', y = 'feature', data = importance.sort_values('importance', ascending=False))

            plt.tight_layout()
            plt.show()

            ################################
            # 評価
            ################################
            # 推論
            pred =  model.predict(X_valid, num_iteration=model.best_iteration)
            # 評価
            score = evaluator(valid, pred)
            print(f"score {i}: {score}")

            scores.append(score)
            models.append(model)
            # save model
            write_model(model, f'model_{i}')

        else:
            read_model(f'model_{i}')
        i = i + 1
    print("CV_SCORES:", scores)
    print("CV_SCORE:", np.mean(scores))
    
    return models

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T10:25:18.834016Z","iopub.execute_input":"2022-04-22T10:25:18.834822Z","iopub.status.idle":"2022-04-22T11:31:00.468151Z","shell.execute_reply.started":"2022-04-22T10:25:18.834784Z","shell.execute_reply":"2022-04-22T11:31:00.467073Z"}}
# 2020-12-23よりも前のデータは証券コードが2000個すべて揃っていないため、これ以降のデータのみを使う。
# (学習用データの開始日、学習用データの終了日＝検証用データの開始日、検証用データの終了日)
fold_params = [
    ('2020-12-23', '2021-11-01', '2021-12-01'),
    ('2021-01-23', '2021-12-01', '2022-01-01'),
    ('2021-02-23', '2022-01-01', '2022-02-01'),
]
models = trainer(feature_df, feat_cols, label_col, fold_params, use_cache=False)
models

# %% [markdown]
# # Evaluation

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T11:31:00.469987Z","iopub.execute_input":"2022-04-22T11:31:00.470318Z","iopub.status.idle":"2022-04-22T11:31:00.481113Z","shell.execute_reply.started":"2022-04-22T11:31:00.470270Z","shell.execute_reply":"2022-04-22T11:31:00.480124Z"}}
def predictor(feature_df, feat_cols, models, is_train=True):
    X = feature_df[feat_cols]
    
    # 推論
    preds = list(map(lambda model: model.predict(X, num_iteration=model.best_iteration), models))
    print(f"preds: {preds}")
    
    # スコアは学習時のみ計算
    if is_train:
        scores = list(map(lambda pred: evaluator(feature_df, pred), preds))
        print("SCORES:", scores)

    # 推論結果をバギング
    pred = np.array(preds).mean(axis=0)

    # スコアは学習時のみ計算
    if is_train:
        score = evaluator(feature_df, pred)
        print("SCORE:", score)
    
    return pred

def eval_predictor(df, feat_cols, models, target_date=['2021-12-06', '2021-12-07']):
    # 日次で推論・登録
    print(f"target_date: {target_date}")
    target_df = df.copy()
    target_df = target_df[(target_date[0] <= target_df['Date']) & (target_df['Date'] < target_date[1])]

    # 推論20
    target_df["pred"] = predictor(target_df, feat_cols, models, True)

    # 推論結果からRANKを導出し、提出データに反映
    result_df = add_rank(target_df)
    return result_df

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T11:31:00.482339Z","iopub.execute_input":"2022-04-22T11:31:00.482545Z","iopub.status.idle":"2022-04-22T11:31:05.844885Z","shell.execute_reply.started":"2022-04-22T11:31:00.482515Z","shell.execute_reply":"2022-04-22T11:31:05.844003Z"}}
_df = eval_predictor(feature_df, feat_cols, models)
#pred_df = _df[0]
#pred_df = pred_df[["SecuritiesCode", "pred", "Date", "Rank"]]
#target_df = _df[1]
#target_df = target_df[["SecuritiesCode", "Target"]]
#result_df = pred_df.merge(target_df, on='SecuritiesCode', how="left")
#write_df(result_df, "result_df")
_df

# %% [markdown]
# # Submit

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T11:31:05.847052Z","iopub.execute_input":"2022-04-22T11:31:05.847348Z","iopub.status.idle":"2022-04-22T11:31:05.882211Z","shell.execute_reply.started":"2022-04-22T11:31:05.847316Z","shell.execute_reply":"2022-04-22T11:31:05.881157Z"}}
# 時系列APIのロード
import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T11:31:05.884644Z","iopub.execute_input":"2022-04-22T11:31:05.885206Z","iopub.status.idle":"2022-04-22T11:31:08.219292Z","shell.execute_reply.started":"2022-04-22T11:31:05.885154Z","shell.execute_reply":"2022-04-22T11:31:08.218382Z"}}
# supplemental filesを履歴データの初期状態としてセットアップ
past_df = train_df.copy()

# %% [code] {"execution":{"iopub.status.busy":"2022-04-22T11:31:08.220712Z","iopub.execute_input":"2022-04-22T11:31:08.220948Z","iopub.status.idle":"2022-04-22T11:35:41.755885Z","shell.execute_reply.started":"2022-04-22T11:31:08.220921Z","shell.execute_reply":"2022-04-22T11:35:41.755004Z"}}
# 日次で推論・登録
for i, (prices, options, financials, trades, secondary_prices, sample_prediction) in enumerate(iter_test):
    current_date = prices["Date"].iloc[0]
    print(f"count {i}, {current_date}")

    if i == 0:
        # リークを防止するため、時系列APIから受け取ったデータより未来のデータを削除
        past_df = past_df[past_df["Date"] < current_date]

    # リソース確保のため古い履歴を削除
    threshold = (pd.Timestamp(current_date) - pd.offsets.BDay(80))
    past_df = past_df[past_df["Date"] >= threshold]
    
    # 時系列APIから受け取ったデータを履歴データに統合
    base_df = collector(prices, options, financials, trades, secondary_prices, stock_list)
    past_df = pd.concat([past_df, base_df]).reset_index(drop=True)

    # 特徴量エンジニアリング
    feature_df, feat_cols, label_col = preprocessor(past_df, False)

    # 予測対象レコードだけを抽出
    feature_df = feature_df[feature_df['Date'] == current_date]

    # 推論
    feature_df["pred"] = predictor(feature_df, feat_cols, models, False)

    # 推論結果からRANKを導出し、提出データに反映
    feature_df = add_rank(feature_df)
    write_df(feature_df, f"result_{i}")
    feature_map = feature_df.set_index('SecuritiesCode')['Rank'].to_dict()
    sample_prediction['Rank'] = sample_prediction['SecuritiesCode'].map(feature_map)

    # 結果を登録
    env.predict(sample_prediction)