import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
from flask_caching import Cache
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

from utils import *

# Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                meta_tags=[
                    # A description of the app, used by e.g.
                    # search engines when displaying search results.
                    {
                        'name': 'description',
                        'content': 'COVID-19 data exploration and forecasts'
                    },
                    # Facebook sharing
                    {
                        'property': 'og:title',
                        'content': 'COVID-19 monitoring'
                    },
                    {
                        'property': 'og:description',
                        'content': 'COVID-19 data exploration and forecasts'
                    },
                    {
                        'property': 'og:image',
                        'content': 'https://i.imgur.com/262RK7z.png'
                    },
                    {
                        'property': 'og:url',
                        'content': 'http://covid-dashboard.guidocioni.it'
                    }
                ],
                suppress_callback_exceptions=True,
                url_base_pathname='/covid/')
server = app.server
app.title = 'COVID-19 live forecast'

cache = Cache(server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '/tmp'
})

TIMEOUT = 3600  # Force update every hour
threshold_chosen = 10000


@cache.memoize(timeout=TIMEOUT)
def read_owid():
  '''Reader from OWID which should be a reliable source for many data. 
  '''
  df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv", 
    parse_dates=[3], index_col=[3])

  df = df.sort_index()

  df['total_cases_change'] = df.groupby("location")['total_cases'].pct_change().rolling(7).mean()*100.
  df['total_deaths_change'] = df.groupby("location")['total_deaths'].pct_change().rolling(7).mean()*100.
  df['positive_rate'] = df['positive_rate'] * 100.

  return df.reset_index()

@cache.memoize(timeout=TIMEOUT)
def read_weekly_ecdc():
  '''Reader from ECDC which should be a reliable source for many data. 
  '''
  r = requests.get('https://www.ecdc.europa.eu/en/publications-data/weekly-subnational-14-day-notification-rate-covid-19')
  soup = BeautifulSoup(r.text, features="lxml")
  file_url = soup.findAll('a',
    string="Download data on the weekly subnational 14-day notification rate of new cases per 100 000 inhabitants for COVID-19",
                       href=re.compile(".xls"))[0]['href']

  df = pd.read_excel(file_url)

  # Only return last week otherwise it takes too long to make the picture
  return df[df.year_week == df.year_week.max()]

@cache.memoize(timeout=TIMEOUT)
def read_hospitalization():
  '''Reader from ECDC which should be a reliable source for many data. 
  '''
  r = requests.get('https://www.ecdc.europa.eu/en/publications-data/download-data-hospital-and-icu-admission-rates-and-current-occupancy-covid-19')
  soup = BeautifulSoup(r.text, features="lxml")
  file_url = soup.findAll('a',
      string="Download data on hospital and ICU admission rates and current occupancy for COVID-19",
      href=re.compile("xls"))[0]['href']

  def dateparse(x): return datetime.strptime(x + '-1', "%Y-W%W-%w")

  df = pd.read_excel(file_url, parse_dates=[3], date_parser=dateparse).drop(columns=['source', 'url'])
  df['date'] = pd.to_datetime(df['date'])
  # fill the date with monday
  df.loc[df.indicator.str.contains('Weekly'), 'date'] = df.loc[df.indicator.str.contains('Weekly'), 'year_week']
  return df


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


@app.callback(
    Output('intermediate-value', 'children'),
    [Input('country-dropdown-multi', 'value')])
def filter_data_for_countries(country):
  return filter_data(countries=country,
                     start_date='2020-03-15',
                     threshold=threshold_chosen).to_json(date_format='iso', orient='split')

# Set up the layout


def serve_layout():
  variable_options = [
     {'value': 'total_cases', 'label': 'Total confirmed cases of COVID-19'},
     {'value': 'new_cases', 'label': 'New confirmed cases of COVID-19'},
     {'value': 'new_cases_smoothed',
      'label': 'New confirmed cases of COVID-19 (7-day smoothed)'},
     {'value': 'total_deaths',
      'label': 'Total deaths attributed to COVID-19'},
     {'value': 'new_deaths', 'label': 'New deaths attributed to COVID-19'},
     {'value': 'new_deaths_smoothed',
      'label': 'New deaths attributed to COVID-19 (7-day smoothed)'},
     {'value': 'total_cases_per_million',
      'label': 'Total confirmed cases of COVID-19 per 1,000,000 people'},
     {'value': 'new_cases_per_million',
      'label': 'New confirmed cases of COVID-19 per 1,000,000 people'},
     {'value': 'new_cases_smoothed_per_million',
      'label': 'New confirmed cases of COVID-19 (7-day smoothed) per 1,000,000 people'},
     {'value': 'total_deaths_per_million',
      'label': 'Total deaths attributed to COVID-19 per 1,000,000 people'},
     {'value': 'new_deaths_per_million',
      'label': 'New deaths attributed to COVID-19 per 1,000,000 people'}
  ]
  variable_options_2 = [
      {'value': 'total_tests', 'label': 'Total tests for COVID-19'},
      {'value': 'new_tests', 'label': 'New tests for COVID-19'},
      {'value': 'new_tests_smoothed',
      'label': "New tests for COVID-19 (7-day smoothed)."},
      {'value': 'total_tests_per_thousand',
      'label': 'Total tests for COVID-19 per 1,000 people'},
      {'value': 'new_tests_per_thousand',
      'label': 'New tests for COVID-19 per 1,000 people'},
      {'value': 'new_tests_smoothed_per_thousand',
      'label': 'New tests for COVID-19 (7-day smoothed) per 1,000 people'},
      {'value': 'tests_per_case',
      'label': 'Tests conducted per new confirmed case of COVID-19'},
      {'value': 'positive_rate',
      'label': 'The share of COVID-19 tests that are positive'}
  ]
  dropdown_options = []
  countries_list = list(filter_data().location.unique())
  for cnt in countries_list:
    dropdown_options.append({"label": cnt, "value": cnt})

  dropdown_options_2 = []
  countries_list_2 = list(read_hospitalization().country.unique())
  for cnt in countries_list_2:
    dropdown_options_2.append({"label": cnt, "value": cnt})

  return html.Div(children=[
      html.Div(html.H1('COVID-19 Monitoring')),
      html.Div('Data are taken from the European Center for Disease Monitoring (ECDC). Choose the relevant tab to show different plots.'),
      html.Div('Last Update: %s' % str(filter_data().date.max())),
      #
      dcc.Tabs(parent_className='custom-tabs', className='custom-tabs-container', 
        children=[
          dcc.Tab(label='Aggregated data', className='custom-tab', selected_className='custom-tab--selected',
            children=[
              html.Div('Click on the legend items to show/hide countries and/or select the countries from the dropdown.'),
              html.Div(
                  'Only countries with more than %s cases at the latest update are included in the search.' % threshold_chosen),
              html.Br(),
              html.Div(
                  dcc.Dropdown(
                      id='country-dropdown-multi',
                      options=dropdown_options,
                      value=['Germany', 'Italy', 'France',
                             'Spain', 'Portugal'],
                      multi=True, style={'width': '800px'})),
              html.Div(id='intermediate-value', style={'display': 'none'}),
              html.Div(
                  dcc.Graph(
                      id='figure-cases',
                      style={'width': '800'}
                  ), style={'display': 'inline-block', 'padding': 10}),
              html.Div(
                  dcc.Graph(
                      id='figure-cumulative',
                      style={'width': '800'}
                  ), style={'display': 'inline-block', 'padding': 10}),
              html.Div(
                  dcc.Graph(
                      id='figure-cumulative-2',
                      style={'width': '800'}
                  ), style={'display': 'inline-block', 'padding': 10}),
              html.Div(
                  dcc.Graph(
                      id='figure-cumulative-3',
                      style={'width': '800'}
                  ), style={'display': 'inline-block', 'padding': 10}),
              html.Div(
                  dcc.Graph(
                      id='figure-increment',
                      style={'width': '800'}
                  ), style={'display': 'inline-block', 'padding': 10}),
              html.Div(
                  dcc.Graph(
                      id='figure-r0',
                      style={'width': '800'}
                  ), style={'display': 'inline-block', 'padding': 10}),

          ]),
          dcc.Tab(label='Testing & Hospitalization', className='custom-tab', selected_className='custom-tab--selected', 
            children=[
              html.Div(['Testing data is weekly. Here is an explanation of the parameters: ', html.Ul(children=[
                  html.Li('positivity rate - 100 x Number of new confirmed cases/number of tests'),
                  html.Li('tests done - total tests performed in a specific country'),
                  html.Li('testing rate - Testing rate per 100 000 population')])]),
            html.Div('Hospitalization data are either weekly or daily depending on the country selected in the dropdown'),
            html.Div(
                  [
                  dcc.Dropdown(
                      id='country-dropdown-testing',
                      options=dropdown_options,
                      value=['Austria','Germany'],
                      multi=True, style={'width': '800px'}),
                  dcc.Dropdown(
                      id='variable-dropdown-2',
                      options=variable_options_2,
                      value='positive_rate',
                      multi=False, style={'width': '800px'}),
                  dcc.Graph(
                      id='figure-testing',
                      style={'width': '800'}
                  )
                  ], style={'display': 'inline-block', 'padding': 10}),
            html.Div(
              [
              dcc.Dropdown(
                      id='country-dropdown-3',
                      options=dropdown_options_2,
                      value='France',
                      multi=False, style={'width': '800px'}),
              dcc.Graph(
                      id='figure-hospitalization',
                      style={'width': '800'}
                  )
              ], style={'display': 'inline-block', 'padding': 10})
            ]),
          dcc.Tab(label='Forecast (logistic)', className='custom-tab', selected_className='custom-tab--selected', 
            children=[
              html.Div('The points show the daily cumulated cases, while the line shows the logistic fit with uncertainty (shaded area).\
                   In the right inset the parameters obtained from the logistic fit are shown: note that these are only parameters, not reliable forecasts!\
                    The value of R2 is also shown: the closer to 1 the better the fit.'),
              html.Div(['Here is an explanation of the parameters: ', html.Ul(children=[
                  html.Li('End = First day without new infections with a threshold of 1/100 on the asymptotic value'),
                  html.Li(
                      'Peak day = Estimated day with maximum growth rate '),
                  html.Li('Max. infected = Asymptotic value for confirmed cases')])]),
              html.Div(
                  'You can choose up to 2 countries to compare side by side.'),
              html.Br(),
              html.Div(
                  [dcc.Dropdown(
                      id='country-dropdown-1',
                      options=dropdown_options,
                      value='Austria'),
                   dcc.Graph(
                      id='figure-fit-1',
                      style={'width': '800'}
                  )],
                  style={'display': 'inline-block', 'padding': 10}),
              html.Div(
                  [dcc.Dropdown(
                      id='country-dropdown-2',
                      options=dropdown_options,
                      value='Germany'),
                   dcc.Graph(
                      id='figure-fit-2',
                      style={'width': '800'}
                  )],
                  style={'display': 'inline-block', 'padding': 10})
          ]),
        dcc.Tab(label='Maps', className='custom-tab', selected_className='custom-tab--selected',
          children=[
            html.Div('Shown is the geographical distribution of many variables. \
              In the first plot you can select the variable to be plotted and explore the daily variation using the slider.\
              In the second plot the subnational distribution of the 14 days reporting ratio is shown only for the most recent data: it is updated every week.'),
            html.Div(
                  [dcc.Dropdown(
                      id='variable-dropdown',
                      options=variable_options,
                      value="new_cases_smoothed_per_million"),
                   dcc.Graph(
                      id='figure-map-world',
                      style={'width': '800'}
                  )],
                  style={'display': 'inline-block', 'padding': 10}),
            html.Div(
                  dcc.Graph(
                      figure=make_fig_map_weekly_europe(),
                      style={'width': '800'}
                  ),
                  style={'display': 'inline-block', 'padding-bottom': 50}),
          ]),
        dcc.Tab(label='Tables (daily data)', className='custom-tab', selected_className='custom-tab--selected',
          children=[
          html.Div('The table shows only the data from the last update. Red shading indicate values exceeding 90th. and 95th. percentiles'),
              html.Div(
                dash_table.DataTable(
                  id='table',
                  columns=make_table()['columns'],
                  data=make_table()['data'],
                  virtualization=True,
                  style_cell={'textAlign': 'left', 'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
                  fixed_rows={'headers': True},
                  style_table={'height': 800},
                  filter_action="native",
                  sort_action="native",
                  sort_mode="multi",
                  style_data_conditional=[
                          {
                              'if': {
                                  'filter_query': '{{{}}} >= {}'.format(col, value),
                                  'column_id': col
                              },
                              'backgroundColor': 'tomato',
                              'color': 'white'
                          } for (col, value) in make_table()['df'].quantile(0.9).iteritems()
                      ] + [
                          {
                              'if': {
                                  'filter_query': '{{{}}} >= {}'.format(col, value),
                                  'column_id': col
                              },
                              'backgroundColor': '#FF4136',
                              'color': 'white'
                          } for (col, value) in make_table()['df'].quantile(0.95).iteritems()
                      ] ,
                  style_header={
                      'backgroundColor': 'rgb(230, 230, 230)',
                      'fontWeight': 'bold',
                      'whiteSpace': 'normal',
                      'height': 'auto',
                  }
))
          ])
  ]),
      html.Div(html.A('Created by Guido Cioni', href='www.guidocioni.it'))
  ], style={'width': '100%', 'display': 'inline-block'})


app.layout = serve_layout

def make_table():
  df = filter_data(start_date='2020-03-15', threshold=1000)
  df = df.loc[df.date == df.date.max()]\
          .round(3).sort_values(by="total_cases_change", ascending=False)

  columns = [
     {'name': 'Continent', 'id': 'continent', 'hideable': True, 'type':'text'},
     {'name': 'Country', 'id': 'location', 'hideable': True, 'type':'text'},
     {'name': 'Daily Cases', 'id': 'new_cases', 'hideable': True, 'type':'numeric'},
     {'name': 'Daily Deaths', 'id': 'new_deaths', 'hideable': True, 'type':'numeric'},
     {'name': 'Cumulative cases', 'id': 'total_cases', 'hideable': True, 'type':'numeric'},
     {'name': 'Cumulative deaths', 'id': 'total_deaths', 'hideable': True, 'type':'numeric'},
     {'name': 'Pct. change of cumulative cases', 'id': 'total_cases_change', 'hideable': True, 'type':'numeric'},
     {'name': 'Pct. change of cumulative deaths','id': 'total_deaths_change', 'hideable': True, 'type':'numeric'},
     {'name': 'Cumulative cases density per 1M inhabitants', 'id': 'total_cases_per_million', 'hideable': True, 'type':'numeric'},
     {'name': 'Cumulative deaths density per 1M inhabitants', 'id': 'total_deaths_per_million', 'hideable': True, 'type':'numeric'}]

  data=df.to_dict('records')

  return {'columns':columns, 'data':data, 'df':df}

def make_fig_map_weekly_europe():
  df = read_weekly_ecdc()

  return make_fig_map_weekly(df)

@app.callback(
    Output('figure-hospitalization', 'figure'),
    [Input('country-dropdown-3', 'value')])
def make_fig_hospitalization(country):
  df = read_hospitalization()

  return make_fig_hospitalization_base(df[df.country == country])

@app.callback(
    Output('figure-testing', 'figure'),
    [Input('variable-dropdown-2', 'value'), Input('country-dropdown-testing', 'value')])
def make_fig_testing(variable, country):
  df = filter_data(countries=country, start_date='2020-03-01')

  return make_fig_testing_base(df, variable)

@app.callback(
    Output('figure-map-world', 'figure'),
    [Input('variable-dropdown', 'value')])
def make_fig_map_world(variable):
  df = filter_data(start_date='2020-06-01', threshold=1000)

  return make_fig_map_base(df, variable)

@app.callback(
    Output('figure-fit-1', 'figure'),
    [Input('country-dropdown-1', 'value')])
def make_fig_fit(country):
  df = filter_data(countries=[country], start_date='2020-05-01')

  return make_fig_fit_base(df)

@app.callback(
    Output('figure-fit-2', 'figure'),
    [Input('country-dropdown-2', 'value')])
def make_fig_fit(country):
  df = filter_data(countries=[country], start_date='2020-05-01')

  return make_fig_fit_base(df)


@app.callback(
    Output('figure-cumulative', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_cumulative_1(df):
  '''Give as input a threshold for the cumulative cases in the most updated
  timestep to filter out countries that do not have many cases.'''
  df = pd.read_json(df, orient='split')
  variable = "total_cases"
  log_y = True
  title = 'Confirmed cases evolution (log. scale, cumulative sum)'

  fig = px.line(df,
                x="date",
                y=variable,
                color="location",
                hover_name="location",
                line_shape="spline",
                render_mode="svg",
                log_y=log_y,
                color_discrete_sequence=px.colors.qualitative.Pastel)

  fig.update_layout(
    template='plotly_white',
      legend_orientation="h",
      width=800,
      height=500,
      title=title,
      xaxis=dict(title=''),
      yaxis=dict(title=''),
      margin=dict(b=0, t=30, l=10),
      legend=dict(
          title=dict(text=''),
          font=dict(
              size=10,
          )
      )
  )

  return fig


@app.callback(
    Output('figure-cumulative-2', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_cumulative_2(df):
  '''Give as input a threshold for the cumulative cases in the most updated
  timestep to filter out countries that do not have many cases.'''
  df = pd.read_json(df, orient='split')

  variable = "total_cases_per_million"
  log_y = False
  title = 'Density of cases (cumulative sum) per 1M inhabitants'

  fig = px.line(df,
                x="date",
                y=variable,
                color="location",
                hover_name="location",
                line_shape="spline",
                render_mode="svg",
                log_y=log_y,
                color_discrete_sequence=px.colors.qualitative.Pastel)

  fig.update_layout(
    template='plotly_white',
      legend_orientation="h",
      margin=dict(b=0, t=30, l=10),
      width=800,
      height=500,
      title=title,
      xaxis=dict(title=''),
      yaxis=dict(title=''),
      legend=dict(
          title=dict(text=''),
          font=dict(
              size=10,
          )
      )
  )

  return fig


@app.callback(
    Output('figure-cumulative-3', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_cumulative_3(df):
  '''Give as input a threshold for the cumulative cases in the most updated
  timestep to filter out countries that do not have many cases.'''
  df = pd.read_json(df, orient='split')

  variable = "total_deaths"
  log_y = True
  title = 'Confirmed deaths evolution (log. scale, cumulative sum)'

  fig = px.line(df,
                x="date",
                y=variable,
                color="location",
                hover_name="location",
                line_shape="spline",
                render_mode="svg",
                log_y=log_y,
                color_discrete_sequence=px.colors.qualitative.Pastel)

  fig.update_layout(
    template='plotly_white',
      margin=dict(b=0, t=30, l=10),
      legend_orientation="h",
      width=800,
      height=500,
      title=title,
      xaxis=dict(title=''),
      yaxis=dict(title=''),
      legend=dict(
          title=dict(text=''),
          font=dict(
              size=10,
          )
      )
  )

  return fig

@app.callback(
    Output('figure-cases', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_cases(df):
  '''Give as input a threshold for the cumulative cases in the most updated
  timestep to filter out countries that do not have many cases.'''
  df = pd.read_json(df, orient='split')

  variable = "new_cases_smoothed"
  title = '7-day smoothed Daily cases evolution'

  fig = px.line(df,
                x="date",
                y=variable,
                color="location",
                hover_name="location",
                line_shape="spline",
                render_mode="svg",
                color_discrete_sequence=px.colors.qualitative.Pastel)

  fig.update_layout(
    template='plotly_white',
      margin=dict(b=0, t=30, l=10),
      legend_orientation="h",
      width=800,
      height=500,
      title=title,
      xaxis=dict(title=''),
      yaxis=dict(title=''),
      legend=dict(
          title=dict(text=''),
          font=dict(
              size=10,
          )
      )
  )

  return fig


@app.callback(
    Output('figure-increment', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_increment(df):
  df = pd.read_json(df, orient='split')
  variable = "total_cases_change"
  title = '7-day smoothed daily increase in confirmed cases'

  fig = px.line(df,
                x="date",
                y=variable,
                color="location",
                hover_name="location",
                line_shape="spline",
                render_mode="svg",
                width=800,
                height=500,
                title=title,
                color_discrete_sequence=px.colors.qualitative.Pastel)

  fig.update_layout(
    template='plotly_white',
      margin=dict(b=0, t=30, l=10),
      legend_orientation="h",
      yaxis=dict(range=[0, 60], title=' % '),
      xaxis=dict(title=''),
      legend=dict(
          title=dict(text=''),
          font=dict(
              size=10,
          )
      )
  )

  return fig


@app.callback(
    Output('figure-r0', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_r0(df):
  df = pd.read_json(df, orient='split')
  title = 'Reproductivity ratio r0 (estimated using RKI method)'

  r0 = df.groupby("location").apply(compute_r0).reset_index(
      level="location").drop(columns="location")

  final = df.merge(r0, right_index=True, left_index=True, how='outer')

  fig = px.line(final,
                x="date",
                y='r0',
                color="location",
                hover_name="location",
                line_shape="spline",
                render_mode="svg",
                width=800,
                height=500,
                title=title,
                color_discrete_sequence=px.colors.qualitative.Pastel)

  fig.update_layout(
    template='plotly_white',
      margin=dict(b=0, t=30, l=10),
      legend_orientation="h",
      yaxis=dict(range=[0, 5], title=''),
      xaxis=dict(title=''),
      legend=dict(
          title=dict(text=''),
          font=dict(
              size=10,
          )
      )
  )

  return fig


if __name__ == '__main__':
  app.run_server()
