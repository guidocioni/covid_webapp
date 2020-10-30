import pandas as pd
from utils import table_columns, table_columns_eu, get_last_valid_data

TMP_FOLDER='/tmp/'


def read_owid():
    return pd.read_pickle(TMP_FOLDER + 'df_owid.pickle')


def read_jrc():
    return pd.read_pickle(TMP_FOLDER + 'df_jrc.pickle')


def read_hospitalization():
    return pd.read_pickle(TMP_FOLDER + 'df_hospitalization.pickle')


def filter_data(countries=None, start_date=None, threshold=None):
    '''- Specify country if you want data for a single country.
       - Specify threshold if you want to filter out countries that do 
         not have at least threshold cases (cumulative) in the latest update
       - Specify start_date to filter data after this date (included)'''
    df = read_owid()
    # Only filter after start date
    if start_date:
        df = df[df.date >= start_date]

    if countries:
        if df.location.isin(countries).any():
            df = df[df.location.isin(countries)]
        else:
            print('Wrong country specified. Use one of the follwing:')
            print(df.location.unique())
            print('Defaulting to all countries...')

    if (not countries) and threshold:
        latest = df[df.date == df.date.max()]
        countries_filter = latest[latest.total_cases >
                                  threshold].location.unique()
        df = df[df.location.isin(list(countries_filter))]

    return df


def get_countries_list(df, country_variable):
    return list(df[country_variable].unique())


def make_table_data():
    df = filter_data(start_date='2020-03-15')

    df = get_last_valid_data(df, variable='new_cases', time_variable='date')\
    .round(3).sort_values(by="new_cases", ascending=False)

    data = df.to_dict('records')

    return {'columns': table_columns,
            'data': data,
            'df': df}


def make_table_data_eu():
    df = read_jrc()
    df = df[df.Region != "NOT SPECIFIED"]
    df = get_last_valid_data(df, variable='daily_cases', time_variable='Date')\
    .round(3).sort_values(by="daily_cases", ascending=False)

    data = df.to_dict('records')

    return {'columns': table_columns_eu,
            'data': data,
            'df': df}

