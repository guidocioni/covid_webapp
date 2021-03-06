import pandas as pd
from datetime import datetime
import numpy as np
import os
from rt_kalman import process_compute_rt

TMP_FOLDER = '/tmp/'
MAIN_FOLDER = os.path.dirname(os.path.realpath(__file__))

''' In this script we perform the preprocessing of data shown in the dashboard in the background
so that the user doesn't have to wait for computatin when opening the dashboard.
The results are saved in pickle files which are then loded with the same name in the application.
You can run this script every 2 hours, but even less, with cron
0 */2 * * * /usr/bin/python3 /home/covid_webapp/data_processing.py'''


def read_owid():
    '''Reader from OWID which should be a reliable source for many data.'''
    df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv",
                     parse_dates=[3],
                     index_col=[3])

    df = df.sort_index()

    df['total_cases_change'] = df.groupby("location")[
        "new_cases_smoothed_per_million"].transform(lambda x: x.diff().rolling(3).mean())
    df['total_deaths_change'] = df.groupby("location")[
        "new_deaths_smoothed_per_million"].transform(lambda x: x.diff().rolling(3).mean())
    df['positive_rate'] = df['positive_rate'] * 100.

    # df = process_compute_rt(df.reset_index(),
    #                         total_cases_var='total_cases', new_cases_var='new_cases',
    #                         time_var='date', location_var='location')

    df = df.rename(columns={'reproduction_rate': 'R'}).reset_index()

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def read_jrc():
    '''Reader from JRC, regional data for EU'''
    df = pd.read_csv("https://raw.githubusercontent.com/ec-jrc/COVID-19/master/data-by-region/jrc-covid-19-all-days-by-regions.csv",
                     parse_dates=[0],
                     index_col=[0])

    df = df.sort_index()
    pop = pd.read_csv(MAIN_FOLDER+'/nuts_europe_population.csv')

    # cannot be negative
    for column in ['CumulativePositive', 'CumulativeDeceased', 'CumulativeRecovered', 'CurrentlyPositive']:
        df.loc[df[column] < 0, column] = np.nan
    # correct data for France that is missing
    df.loc[(df.iso3 == 'FRA') & (df.index > '2020-03-25'),
           'CumulativePositive'] = np.nan

    out = []
    for grouper, group in df.groupby("Region"):
        temp = group[['CumulativePositive', 'CumulativeDeceased',
                      'CumulativeRecovered']].transform(lambda x: x.diff()).reset_index()
        temp = temp.rename(columns={'CumulativePositive': 'daily_cases',
                           'CumulativeDeceased': 'daily_deaths', 'CumulativeRecovered': 'daily_recovered'})
        temp['Region'] = grouper
        out.append(temp)

    out = pd.concat(out)

    df = df.reset_index().merge(out, on=['Date', 'Region']).set_index('Date')

    df['daily_cases_smoothed'] = df.groupby(
        "Region")['daily_cases'].transform(lambda x: x.rolling(7).mean())
    df['daily_deaths_smoothed'] = df.groupby(
        "Region")['daily_deaths'].transform(lambda x: x.rolling(7).mean())
    df['daily_recovered_smoothed'] = df.groupby(
        "Region")['daily_recovered'].transform(lambda x: x.rolling(7).mean())
    df['total_cases_change'] = df.groupby("Region")[
        "daily_cases_smoothed"].transform(lambda x: x.diff().rolling(3).mean())
    df['total_deaths_change'] = df.groupby("Region")[
        "daily_deaths_smoothed"].transform(lambda x: x.diff().rolling(3).mean())

    df['location'] = df['CountryName'] + ' | ' + df['Region']

    df = process_compute_rt(df.reset_index(),
                            total_cases_var='CumulativePositive', new_cases_var='daily_cases',
                            time_var='Date', location_var='location')

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.merge(pop, left_on='NUTS', right_on='nuts_code').drop(
        columns='nuts_code')

    for column in ['CumulativePositive', 'CumulativeDeceased', 'CumulativeRecovered',
                   'CurrentlyPositive', 'daily_cases', 'daily_deaths',
                   'daily_recovered', 'daily_cases_smoothed', 'daily_deaths_smoothed',
                   'daily_recovered_smoothed']:

        df[column+'_per_million'] = df[column] / df['population'] * 1e6

    return df


def read_hospitalization():
    def dateparse(x):
        return datetime.strptime(x + '-1', "%Y-W%W-%w")
    df = pd.read_csv('https://opendata.ecdc.europa.eu/covid19/hospitalicuadmissionrates/csv/data.csv',
                     parse_dates=[3], date_parser=dateparse).drop(
        columns=['source', 'url'])
    df['date'] = pd.to_datetime(df['date'])
    # fill the date with monday
    df.loc[df.indicator.str.contains(
        'Weekly'), 'date'] = df.loc[df.indicator.str.contains('Weekly'), 'year_week']

    return df


try:
    df_owid = read_owid().to_pickle(TMP_FOLDER + 'df_owid.pickle')
except Exception as e:
    print(e)

try:
    df_jrc = read_jrc().to_pickle(TMP_FOLDER + 'df_jrc.pickle')
except Exception as e:
    print(e)

try:
    df_hospitalization = read_hospitalization().to_pickle(
        TMP_FOLDER + 'df_hospitalization.pickle')
except Exception as e:
    print(e)
