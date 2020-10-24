from utils import compute_r0
import pandas as pd
from datetime import datetime
import numpy as np

TMP_FOLDER = '/tmp/'

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

    df['r0'] = df.groupby("location").new_cases.transform(compute_r0)

    df = df.replace([np.inf, -np.inf], np.nan)

    return df.reset_index()


def read_jrc():
    '''Reader from JRC, regional data for EU'''
    df = pd.read_csv("https://raw.githubusercontent.com/ec-jrc/COVID-19/master/data-by-region/jrc-covid-19-all-days-by-regions.csv",
                     parse_dates=[0],
                     index_col=[0])

    df = df.sort_index()

    df['daily_cases'] = df.groupby(
        "Region")['CumulativePositive'].transform(lambda x: x.diff())
    df['daily_deaths'] = df.groupby(
        "Region")['CumulativeDeceased'].transform(lambda x: x.diff())
    df['daily_recovered'] = df.groupby(
        "Region")['CumulativeRecovered'].transform(lambda x: x.diff())
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

    df['r0'] = df.groupby("location").daily_cases.transform(compute_r0)

    df = df.replace([np.inf, -np.inf], np.nan)

    return df.reset_index()


def read_hospitalization():
    def dateparse(x): return datetime.strptime(x + '-1', "%Y-W%W-%w")
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
except:
    print("Error in downloading/processing df_owid")

try:
    df_jrc = read_jrc().to_pickle(TMP_FOLDER + 'df_jrc.pickle')
except:
    print("Error in downloading/processing df_jrc")

# df_weekly_ecdc = read_weekly_ecdc().to_pickle(
#     TMP_FOLDER + 'df_weekly_ecdc.pickle')
try:
    df_hospitalization = read_hospitalization().to_pickle(
        TMP_FOLDER + 'df_hospitalization.pickle')
except:
    print("Error in downloading/processing df_hospitalization")
