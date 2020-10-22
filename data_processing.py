from utils import compute_r0
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

TMP_FOLDER = '/tmp/'


def read_owid():
    '''Reader from OWID which should be a reliable source for many data.'''
    df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv",
                     parse_dates=[3],
                     index_col=[3])

    df = df.sort_index()

    df['total_cases_change'] = df.groupby(
        "location")['total_cases'].pct_change().rolling(7).mean() * 100.
    df['total_deaths_change'] = df.groupby(
        "location")['total_deaths'].pct_change().rolling(7).mean() * 100.
    df['positive_rate'] = df['positive_rate'] * 100.

    df['r0'] = df.groupby("location").new_cases.transform(compute_r0)

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
        "Region")['CumulativePositive'].transform(lambda x: x.diff().rolling(7).mean())
    df['daily_deaths_smoothed'] = df.groupby(
        "Region")['CumulativeDeceased'].transform(lambda x: x.diff().rolling(7).mean())
    df['daily_recovered_smoothed'] = df.groupby(
        "Region")['CumulativeRecovered'].transform(lambda x: x.diff().rolling(7).mean())

    df['location'] = df['CountryName'] + ' | ' + df['Region']

    df['r0'] = df.groupby("location").daily_cases.transform(compute_r0)

    return df.reset_index()


def read_weekly_ecdc():
    r = requests.get(
        'https://www.ecdc.europa.eu/en/publications-data/weekly-subnational-14-day-notification-rate-covid-19')
    soup = BeautifulSoup(r.text, features="lxml")
    file_url = soup.findAll('a',
                            string="Download data on the weekly subnational 14-day notification rate of new cases per 100 000 inhabitants for COVID-19",
                            href=re.compile(".xls"))[0]['href']

    df = pd.read_excel(file_url)

    # Only return last week otherwise it takes too long to make the picture
    return df[df.year_week == df.year_week.max()]


def read_hospitalization():
    r = requests.get(
        'https://www.ecdc.europa.eu/en/publications-data/download-data-hospital-and-icu-admission-rates-and-current-occupancy-covid-19')
    soup = BeautifulSoup(r.text, features="lxml")
    file_url = soup.findAll('a',
                            string="Download data on hospital and ICU admission rates and current occupancy for COVID-19",
                            href=re.compile("xls"))[0]['href']

    def dateparse(x): return datetime.strptime(x + '-1', "%Y-W%W-%w")
    df = pd.read_excel(file_url, parse_dates=[3], date_parser=dateparse).drop(
        columns=['source', 'url'])
    df['date'] = pd.to_datetime(df['date'])
    # fill the date with monday
    df.loc[df.indicator.str.contains(
        'Weekly'), 'date'] = df.loc[df.indicator.str.contains('Weekly'), 'year_week']

    return df


df_owid = read_owid().to_pickle(TMP_FOLDER + 'df_owid.pickle')
df_jrc = read_jrc().to_pickle(TMP_FOLDER + 'df_jrc.pickle')
df_weekly_ecdc = read_weekly_ecdc().to_pickle(
    TMP_FOLDER + 'df_weekly_ecdc.pickle')
df_hospitalization = read_hospitalization().to_pickle(
    TMP_FOLDER + 'df_hospitalization.pickle')
