import pandas as pd
import plotly.graph_objs as go
import os
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
import joblib

# trained model paths
STOCK_MODEL_PATH = "models/arima_model.pkl"
VOLUME_MODEL_PATH = "models/volume_model.pkl"

# general data directory
DATA_FOLDER = '../data'

# default file names
FILE_MARKETSTACK_RAW = '../data/marketstack_raw.csv'
FILE_MARKETSTACK_CLEAN = '../data/marketstack_clean.csv'

# Stock to analyze
SYMBOL = ['GT']


def create_folder(folder_name):
    """ Make directory if it doesn't already exist """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print('Folder {} created'.format(folder_name))
    else:
        print('Folder {} already exist'.format(folder_name))


def gather_data(symbol):
    """ get stocks data using marketstack API if data folder is empty"""

    # Make main data directory if it doesn't already exist
    create_folder(DATA_FOLDER)

    # import API credentials from environment variables
    if symbol is None:
        stocks = SYMBOL

    # get API access key
    access_key = os.environ.get('MARKET_STACK_API')

    error_list = []
    df_all = pd.DataFrame()

    # for stock in stocks:
    params = {
        'access_key': access_key,
        'symbols': symbol,
        'date_from': '2014-01-01',
        'date_to': '2024-02-29',
        'limit': 10000
    }

    api_result = requests.get('https://api.marketstack.com/v1/eod', params)
    api_response = api_result.json()
    if api_result.status_code == 200:
        data = api_response['data']
        df = pd.DataFrame.from_dict(data)
        if df_all.shape[0] == 0:
            df_all = df.copy(deep=True)
        else:
            frames = [df, df_all]
            df_all = pd.concat(frames)
            print('#records for {} is {}; total records is {}'.format(stock, df.shape[0], df_all.shape[0]))
    else:
        # construct error message and append to error list
        error_message = "{} Request returned an error: {} {}".format(stock, api_response['error']['code'],
                                                                     api_response['error']['message'])
        print(error_message)
        error_list.append(error_message)

    df_all.to_csv(FILE_MARKETSTACK_RAW, index=False)


def clean_data(from_path, to_path):
    """ Read raw stock data from csv, clean and save as cleaned csv file """

    if not os.path.exists(DATA_FOLDER):
        gather_data(symbol=SYMBOL)

    marketstack = pd.read_csv(from_path)
    marketstack_clean = marketstack.copy(deep=True)

    # convert to date field and drop the time
    marketstack_clean['date'] = pd.to_datetime(marketstack_clean['date'])
    marketstack_clean['date'] = pd.to_datetime(
        marketstack_clean['date'],
        format='%Y/%m/%d',
        errors='raise').dt.date.astype('datetime64[ns]')

    # select only columns needed
    cols = ['date', 'symbol', 'close', 'volume']
    marketstack_clean = marketstack_clean[cols]

    # set date as index for time series analysis
    marketstack_clean.set_index('date', inplace=True)
    marketstack_clean = marketstack_clean.sort_index()

    # reduce size of volume store divide by mil
    marketstack_clean['volume'] = marketstack_clean['volume'] / 1000000

    # get rid of some weekly noice
    marketstack_clean['close'] = marketstack_clean['close'].rolling(7, center=False).mean()
    marketstack_clean['volume'] = marketstack_clean['volume'].rolling(7, center=False).mean()

    # resample data to monthly
    freq = 'MS'
    marketstack_clean = (marketstack_clean.resample(freq).agg(
        {'symbol': 'first',
         'close': 'median',
         'volume': 'max'
         }))

    marketstack_clean = marketstack_clean.dropna()

    marketstack_clean.to_csv(to_path, index=True)

    return marketstack_clean


def predict_volume(df_train, df_test, period):

    # load pretrained ARIMA model to predict volume
    model_volume = joblib.load(VOLUME_MODEL_PATH)

    volume_forecast = model_volume.predict(
        start=len(df_train) - period,
        end=len(df_train) + period,
        dynamic=False
    )

    end = df_test.last_valid_index()
    df_volume_test = volume_forecast.loc[end:][1:].to_frame()
    df_volume_test.columns = ['volume']
    df_volume_test = pd.concat([df_test, df_volume_test])

    return df_volume_test


def predict_stock(df_train, df_test, period):

    # load pretrained ARIMA model to predict stock values
    arima_results = joblib.load(STOCK_MODEL_PATH)

    # predict future volumes
    df_volume_test = predict_volume(df_train, df_test, period)

    start = len(df_train)
    end = len(df_train) + period

    forecast = arima_results.predict(
        start=start,
        end=end,
        exog=df_volume_test['volume'][:end],
        dynamic=False)

    return forecast

def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    # first chart plots arable land from 1990 to 2015 in top 10 economies
    # as a line chart

    # get stock data
    df = clean_data(
        from_path=FILE_MARKETSTACK_RAW,
        to_path=FILE_MARKETSTACK_CLEAN)

    # print stock evolution with std and mean
    graph_one = []

    close_mean = df['close'].expanding().mean()
    close_std = df['close'].expanding().std()

    graph_one.append(
        go.Scatter(
            x=df.index,
            y=df.close,
            mode='lines',
            name=SYMBOL[0],
        )
    )

    graph_one.append(
        go.Scatter(
            x=df.index,
            y=close_mean,
            mode='lines',
            name='Mean',
        )
    )

    graph_one.append(
        go.Scatter(
            x=df.index,
            y=close_std,
            mode='lines',
            name='Standard deviation',
        )
    )

    graph_one.append(
        go.Scatter(
            x=df.index,
            y=df.volume,
            mode='lines',
            name='share volume',
            fillcolor='grey',
        )
    )

    layout_one = dict(title='Stock Price Evolution',
                      xaxis=dict(title='Date'),
                      yaxis=dict(title='Stock value (in USD)'),
                      # height=400,
                      # width=800,
                      )

    # display stocks trend
    graph_two = []

    # calculate the trend
    df['trend'] = df['close'].rolling(window=12, center=True).mean()

    graph_two.append(
        go.Scatter(
            x=df.index,
            y=df['trend'],
            mode='lines',
        )
    )

    layout_two = dict(title='Trend',
                      xaxis=dict(title='Date'),
                      yaxis=dict(title='$'),
                      # height=200,
                      # width=400,
                      )

    # display stocks seasonality
    graph_three = []

    # detrend the series
    df['detrended'] = df['close'] - df['trend']

    # calculate the seasonal component
    df["month"] = df.index.month
    df["seasonality"] = df.groupby("month")["detrended"].transform("mean")

    graph_three.append(
        go.Scatter(
            x=df['2021':].index,
            y=df["seasonality"]["2021":],
            mode='lines',
        )
    )

    layout_three = dict(title='Seasonality',
                        xaxis=dict(title='Date'),
                        yaxis=dict(title='$'),
                        # height=150,
                        # width=300,
                        )

    # display future predictions
    graph_four = []

    df_train = df.loc[:'2023-06']
    df_test = df.loc['2023-07':]
    future_periods = 24

    forecast = predict_stock(df_train, df_test, future_periods)

    graph_four.append(
        go.Scatter(
            x=df_test.index,
            y=df_test.close,
            mode='lines',
            name='actual'
        )
    )

    graph_four.append(
        go.Scatter(
            x=forecast.index,
            y=forecast,
            mode='lines',
            name='forecast'
        )
    )

    layout_four = dict(title='Stock value prediction',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Stock value (in USD)'),
                       )

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))

    return figures


if __name__ == "__main__":

    # gather_data(symbol=SYMBOL)

    clean_data(
        from_path=FILE_MARKETSTACK_RAW,
        to_path=FILE_MARKETSTACK_CLEAN
)