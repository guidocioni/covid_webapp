import pandas as pd
import numpy as np
from scipy.optimize import fsolve, curve_fit
import plotly.graph_objs as go
import plotly.express as px
import json
from copy import deepcopy
from dash_table import DataTable

TIMEOUT = 1800  # Force cache update every hour
threshold_chosen = 10000

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Used in the world map
variable_options = [
    {'value': 'total_cases', 'label': 'Total confirmed cases of COVID-19'},
    {'value': 'new_cases', 'label': 'New confirmed cases of COVID-19'},
    {'value': 'total_deaths',
        'label': 'Total deaths'},
    {'value': 'total_cases_per_million',
        'label': 'Total confirmed cases per million'},
    {'value': 'new_cases_per_million',
        'label': 'New confirmed cases per million'},
    {'value': 'total_deaths_per_million',
        'label': 'Total deaths per million'},
    {'value': 'new_deaths_per_million',
        'label': 'New deaths per million'},
    {'value': 'new_tests_per_thousand',
     'label': 'New tests per thousand'},
    {'value': 'total_tests_per_thousand',
     'label': 'Total tests per thousand'},
    {'value': 'positive_rate',
     'label': 'Test positive rate'},
    {'value': 'stringency_index',
     'label': 'Stringency Index'},
    {'value': 'total_cases_change',
     'label': 'Cases change'},
    {'value': 'total_deaths_change',
     'label': 'Deceased change'},
    {'value': 'r0',
     'label': 'Reproductivity number R0'},
]

# Used in the test tab
variable_options_2 = [
    {'value': 'total_tests', 'label': 'Total tests'},
    {'value': 'new_tests', 'label': 'New tests'},
    {'value': 'new_tests_smoothed',
     'label': "New tests (7-day smoothed)"},
    {'value': 'total_tests_per_thousand',
     'label': 'Total tests per 1,000 people'},
    {'value': 'new_tests_per_thousand',
     'label': 'New tests per 1,000 people'},
    {'value': 'new_tests_smoothed_per_thousand',
     'label': 'New tests (7-day smoothed) per 1,000 people'},
    {'value': 'tests_per_case',
     'label': 'Tests conducted per new confirmed case'},
    {'value': 'positive_rate',
     'label': 'The share of tests that are positive'}
]

# Used in the time series of EU aggregated variables
variable_options_eu = [
    {'value': 'daily_cases', 'label': 'Daily positive'},
    {'value': 'daily_cases_per_million', 'label': 'Daily positive per million'},
    {'value': 'daily_deaths', 'label': 'Daily deceased'},
    {'value': 'daily_deaths_per_million', 'label': 'Daily deceased per million'},
    {'value': 'daily_recovered', 'label': 'Daily recovered'},
    {'value': 'daily_recovered_per_million', 'label': 'Daily recovered per million'},
    {'value': 'daily_cases_smoothed',
     'label': 'Daily positive (7-days smoothing)'},
    {'value': 'daily_cases_smoothed_per_million',
     'label': 'Daily positive per million (7-days smoothing)'},
    {'value': 'daily_deaths_smoothed',
     'label': 'Daily deceased (7-days smoothing)'},
    {'value': 'daily_deaths_smoothed_per_million',
     'label': 'Daily deceased per million (7-days smoothing)'},
    {'value': 'daily_recovered_smoothed',
     'label': 'Daily recovered (7-days smoothing)'},
    {'value': 'daily_recovered_smoothed_per_million',
     'label': 'Daily recovered per million (7-days smoothing)'},
    {'value': 'total_cases_change', 'label': 'Smoothed change in daily new cases'},
    {'value': 'total_deaths_change', 'label': 'Smoothed change in daily new deceased'},
    {'value': 'CumulativePositive', 'label': 'Total positive'},
    {'value': 'CumulativePositive_per_million', 'label': 'Total positive per million'},
    {'value': 'CumulativeDeceased', 'label': 'Total deceased'},
    {'value': 'CumulativeDeceased_per_million', 'label': 'Total deceased per million'},
    {'value': 'CumulativeRecovered', 'label': 'Total recovered'},
    {'value': 'CumulativeRecovered_per_million', 'label': 'Total recovered per million'},
    {'value': 'CurrentlyPositive', 'label': 'Currently positive'},
    {'value': 'CurrentlyPositive_per_million', 'label': 'Currently positive per million'},
    {'value': 'r0', 'label': 'Reproductivity number R0'}
]

table_columns = [
    {'name': 'Continent', 'id': 'continent',
     'hideable': False, 'type': 'text'},
    {'name': 'Country', 'id': 'location',
     'hideable': False, 'type': 'text'},
    {'name': 'Last update', 'id': 'date',
     'hideable': True, 'type': 'text'},
    {'name': 'Daily Cases', 'id': 'new_cases',
     'hideable': True, 'type': 'numeric'},
    {'name': 'Daily Deaths', 'id': 'new_deaths',
     'hideable': True, 'type': 'numeric'},
    {'name': 'Change of daily cases', 'id': 'total_cases_change',
     'hideable': True, 'type': 'numeric'},
    {'name': 'Change of daily deaths', 'id': 'total_deaths_change',
     'hideable': True, 'type': 'numeric'},
    {'name': 'Total cases', 'id': 'total_cases',
     'hideable': True, 'type': 'numeric'},
    {'name': 'Total deaths', 'id': 'total_deaths',
     'hideable': True, 'type': 'numeric'},
    {'name': 'Total cases per million', 'id': 'total_cases_per_million',
     'hideable': True, 'type': 'numeric'},
    {'name': 'Total deaths per million', 'id': 'total_deaths_per_million',
     'hideable': True, 'type': 'numeric'},
    {'name': 'New tests per thousand', 'id': 'new_tests_per_thousand',
     'hideable': True, 'type': 'numeric'},
    {'name': 'Total tests per thousand', 'id': 'total_tests_per_thousand',
     'hideable': True, 'type': 'numeric'},
    {'name': 'Positive tests rate', 'id': 'positive_rate',
     'hideable': True, 'type': 'numeric'},
    {'name': 'R0', 'id': 'r0',
     'hideable': True, 'type': 'numeric'}
     ]


table_columns_eu = [
    {'name': 'Country', 'id': 'CountryName',
     'hideable': True, 'type': 'text'},
    {'name': 'Region', 'id': 'Region',
     'hideable': False, 'type': 'text'},
    {'name': 'Daily Cases', 'id': 'daily_cases',
     'hideable': True, 'type': 'numeric'},
    {'name': 'Change of daily cases', 'id': 'total_cases_change',
     'hideable': True, 'type': 'numeric'},
    {'name': 'R0', 'id': 'r0',
     'hideable': True, 'type': 'numeric'},
    {'id': 'daily_cases_per_million', 'name': 'Daily positive per million',
        'hideable': True, 'type': 'numeric'},
    {'id': 'daily_deaths', 'name': 'Daily deceased',
        'hideable': True, 'type': 'numeric'},
    {'id': 'daily_deaths_per_million', 'name': 'Daily deceased per million',
        'hideable': True, 'type': 'numeric'},
    {'id': 'daily_recovered', 'name': 'Daily recovered',
        'hideable': True, 'type': 'numeric'},
    {'id': 'daily_recovered_per_million', 'name': 'Daily recovered per million',
        'hideable': True, 'type': 'numeric'},
    {'id': 'total_deaths_change', 'name': 'Smoothed change in daily new deceased',
        'hideable': True, 'type': 'numeric'},
    {'id': 'CumulativePositive', 'name': 'Total positive',
        'hideable': True, 'type': 'numeric'},
    {'id': 'CumulativePositive_per_million',
        'name': 'Total positive per million', 'hideable': True, 'type': 'numeric'},
    {'id': 'CumulativeDeceased', 'name': 'Total deceased',
        'hideable': True, 'type': 'numeric'},
    {'id': 'CumulativeDeceased_per_million',
        'name': 'Total deceased per million', 'hideable': True, 'type': 'numeric'},
    {'id': 'CumulativeRecovered', 'name': 'Total recovered',
        'hideable': True, 'type': 'numeric'},
    {'id': 'CumulativeRecovered_per_million',
        'name': 'Total recovered per million', 'hideable': True, 'type': 'numeric'},
    {'id': 'CurrentlyPositive', 'name': 'Currently positive',
        'hideable': True, 'type': 'numeric'},
    {'id': 'CurrentlyPositive_per_million',
        'name': 'Currently positive per million', 'hideable': True, 'type': 'numeric'} 
     ]

plot_opts = {
    'daily_cases':{'color_continuous_scale':"YlOrRd", 'range_color':(0, 5000)},
    'daily_deaths':{'color_continuous_scale':"YlOrRd", 'range_color':(0, 50)},
    'daily_recovered':{'color_continuous_scale':"YlOrRd", 'range_color':(0, 2000)},
    'daily_cases_smoothed':{'color_continuous_scale':"YlOrRd", 'range_color':(0, 5000)},
    'daily_deaths_smoothed':{'color_continuous_scale':"YlOrRd", 'range_color':(0, 50)},
    'daily_recovered_smoothed':{'color_continuous_scale':"YlOrRd", 'range_color':(0, 2000)},
    'daily_cases_per_million':{'color_continuous_scale':"YlOrRd"},
    'daily_deaths_per_million':{'color_continuous_scale':"YlOrRd"},
    'daily_recovered_per_million':{'color_continuous_scale':"YlOrRd"},
    'daily_cases_smoothed_per_million':{'color_continuous_scale':"YlOrRd"},
    'daily_deaths_smoothed_per_million':{'color_continuous_scale':"YlOrRd"},
    'daily_recovered_smoothed_per_million':{'color_continuous_scale':"YlOrRd"},
    'total_cases_change':{'color_continuous_scale':"curl", 'range_color':(-200, 200)},
    'total_deaths_change':{'color_continuous_scale':"curl", 'range_color':(-20, 20)},
    'CumulativePositive':{'color_continuous_scale':"YlOrRd"},
    'CumulativeDeceased':{'color_continuous_scale':"YlOrRd"},
    'CumulativeRecovered':{'color_continuous_scale':"YlOrRd"},
    'CurrentlyPositive':{'color_continuous_scale':"YlOrRd"},
    'CumulativePositive_per_million':{'color_continuous_scale':"YlOrRd"},
    'CumulativeDeceased_per_million':{'color_continuous_scale':"YlOrRd"},
    'CumulativeRecovered_per_million':{'color_continuous_scale':"YlOrRd"},
    'CurrentlyPositive_per_million':{'color_continuous_scale':"YlOrRd"},
    'r0':{'color_continuous_scale':"Inferno", 'range_color':(0, 5)},
}

plot_opts_global = {
    'new_cases':{'color_continuous_scale':"amp", 'range_color':(1000, 50000)},
    'new_deaths':{'color_continuous_scale':"amp", 'range_color':(0, 1000)},
    'total_cases':{'color_continuous_scale':"amp", 'range_color':(5e4, 10e6)},
    'total_deaths':{'color_continuous_scale':"amp"},
    'total_cases_per_million':{'color_continuous_scale':"amp"},
    'new_cases_per_million':{'color_continuous_scale':"amp"},
    'total_deaths_per_million':{'color_continuous_scale':"amp"},
    'new_deaths_per_million':{'color_continuous_scale':"amp"},
    'total_cases_change':{'color_continuous_scale':"curl", 'range_color':(-25, 25)},
    'total_deaths_change':{'color_continuous_scale':"curl", 'range_color':(-5, 5)},
    'r0':{'color_continuous_scale':"Inferno", 'range_color':(0, 5)},
    'stringency_index':{'color_continuous_scale':"tempo", 'range_color':(0, 100)},
    'positive_rate':{'color_continuous_scale':"amp", 'range_color':(0, 50)},
    'total_tests_per_thousand':{'color_continuous_scale':"amp"},
    'new_tests_per_thousand':{'color_continuous_scale':"amp"},
}


def get_last_valid_data(df, grouper='location', variable='daily_cases', 
                        time_window='30d', time_variable='Date'):
    '''Get last valid data (no NaN) for variable in time over a grouped dataframe
    with the constraint that it has to be dated between today and today-time_window
    to exclude really old data.'''
    df = df.set_index(time_variable)
    today = pd.to_datetime('today')
    def selector(group, today, time_window):
        date_last = group[variable].last_valid_index() # this is a datetime
        try:
            if today - date_last < pd.to_timedelta(time_window):
                return group[group.index == date_last]
        except TypeError:
            return None

    return df.groupby(grouper).apply(selector, today=today, time_window=time_window).reset_index(level=time_variable)

def compute_r0_old(group, window=7, variable='new_cases'):
  # Compute R0 using RKI method, old method, really slow,
  # kept only for reference, will be removed
  r0 = []
  for t in range((2 * window) - 1, len(group)):
    r0.append(group.iloc[t - window + 1:t + 1][variable].sum() /
              group.iloc[t - (2 * window) + 1:t - window + 1][variable].sum())

  return pd.DataFrame(data={'r0': r0},
                      index=group.index[(2 * window) - 1:])


def compute_r0(group, window=7):
  # Compute R0 using RKI method
  r0 = (group.rolling(window).sum()
        / group.rolling(window=window, min_periods=1).sum().shift(window - 1))

  return r0


def logistic_model(x, a, b, c, d):
  return c / (1 + np.exp(-(x - b) / a)) + d


def r2(y_obs, y_model):
  return 1. - (np.sum((y_obs - y_model)**2) / np.sum(
      (y_obs - np.mean(y_obs))**2))


def parameters(df, log_fit):
  a, b, c, d = log_fit[0]
  day_last_index = int(
      fsolve(lambda x: logistic_model(x, a, b, c, d) - int((c + d) - (c + d) / 100), b))
  day_last = pd.date_range(start=df['date'].iloc[0],
                           periods=1000)[day_last_index]\
      - pd.to_timedelta('20 days')
  day_peak = pd.date_range(start=df['date'].iloc[0],
                           periods=1000)[b.round().astype(int)] \
      - pd.to_timedelta('20 days')

  saturation_value = c + d

  return {
      'day_peak': pd.to_datetime(day_peak).strftime('%d %b %Y'),
      'day_end': pd.to_datetime(day_last).strftime('%d %b %Y'),
      'saturation_value': saturation_value
  }


def scatter_cases(df, variable,
                  color='rgb(252, 141, 98)', symbol="circle"):
  '''Create a trace for the scatter of the cases'''
  trace = go.Scatter(name=variable,
                     x=df['date'],
                     y=df[variable],
                     mode="markers",
                     showlegend=True,
                     marker=dict(size=8,
                                 color=color,
                                 opacity=0.8,
                                 symbol=symbol))
  return trace


def logistic_curve(df, variable,
                   color='rgb(251, 180, 174)', linewidth=2):
  r2_value = r2(df['total_cases'], df['total_cases_prediction'])
  trace_log = go.Scatter(name='Logistic fit, R2=%3.3f' % r2_value,
                         x=df['date'],
                         y=df['total_cases_prediction'],
                         mode="lines",
                         showlegend=True,
                         line=dict(color=color, width=linewidth))

  trace_log_p1sigma = go.Scatter(
      x=df['date'],
      showlegend=False,
      y=df[variable + '_prediction_upper'],
      mode="lines",
      line=dict(color=color, width=0))
  trace_log_m1sigma = go.Scatter(
      x=df['date'],
      showlegend=False,
      y=df[variable + '_prediction_lower'],
      mode="lines",
      line=dict(color=color, width=0),
      fill='tonexty')

  return [trace_log, trace_log_p1sigma, trace_log_m1sigma]


def new_cases(df, variable, color='rgb(252, 141, 98)'):
  scatter = go.Bar(
      name=variable + ' variation',
      x=df['date'],
      y=df[variable].diff(),
      xaxis='x2',
      yaxis='y2',
      showlegend=False,
      marker_color=color)

  line = go.Scatter(
      x=df['date'],
      y=df[variable + '_prediction'].diff(),
      xaxis='x2',
      yaxis='y2',
      showlegend=False,
      mode="lines",
      marker_color=color)

  return [scatter, line]


def prepare_data(df, variable):
  '''Prepare the data needed for the fit. This should be
  already filtered by country.'''
  subset = df.copy()
  # We decide which date to start from finding the minimum of cases
  beginning_date = subset[subset.index == subset.new_cases.rolling(
      7).mean().idxmin()].date.values[0]
  subset = subset[subset.date >= beginning_date]
  date_shift = beginning_date - pd.to_timedelta('20 days')
  subset.loc[:, 'days'] = subset.loc[:, 'date'] - date_shift

  return subset


def fit_data(df, variable, maxfev=80000):
  '''Fit data from the dataframe and obtain fit parameters'''
  input_df = prepare_data(df, variable)
  xdata = input_df['days'].dt.days.values
  ydata = input_df[variable].values
  # some empirical parameters that seem to work
  # [speed, day_inflection, max-min infections, min infections]
  p0_bounds = ([0, 20, 10000, ydata.min()],
               [np.inf, np.inf, np.inf, ydata.min() + 1])
  # Do the fit
  log_fit = curve_fit(f=logistic_model,
                      xdata=xdata,
                      ydata=ydata,
                      bounds=p0_bounds,
                      maxfev=maxfev)

  return log_fit, input_df


def predict_data(df, variable, days_prediction=50):
  '''Predict evolution of variable according to fit'''
  # First fit the data
  log_fit, input_df = fit_data(df, variable)
  # Construct future prediction
  future_df = pd.DataFrame(
      {'date': pd.date_range(start=input_df['date'].max() + pd.to_timedelta('1 D'),
                             periods=days_prediction),
       'days': pd.timedelta_range(input_df['days'].max() + pd.to_timedelta('1 D'),
                                  periods=days_prediction)})

  final = pd.concat([input_df, future_df]).reset_index(drop=True)
  final[variable + '_prediction'] = logistic_model(final['days'].dt.days.values,
                                                   *log_fit[0])
  final[variable + '_prediction_upper'] = logistic_model(final['days'].dt.days.values,
                                                         *(log_fit[0] + np.sqrt(np.diag(log_fit[1]))))
  final[variable + '_prediction_lower'] = logistic_model(final['days'].dt.days.values,
                                                         *(log_fit[0] - np.sqrt(np.diag(log_fit[1]))))
  return final, log_fit


def figure_layout(title, country, day_last_string,
                  day_peak_string, saturation_value):
  '''Set the figure layout'''
  layout = dict(
      template='plotly_white',
      title=title,
      showlegend=True,
      height=500,
      width=800,
      margin=dict(r=5, b=20, t=30, l=10),
      font=dict(family='arial'),
      yaxis=dict(title='Cumulative Cases (%s)' % country),
      legend=dict(x=0.75, y=0.1),
      annotations=[
          dict(
              x=0.95,
              y=0.4,
              xref="paper",
              yref="paper",
              text="<b>Fit parameters (logistic)</b> <br> End = %s <br> Peak day = %s <br> Max. cases = %d"
              % (day_last_string, day_peak_string, saturation_value),
              showarrow=False,
              bgcolor='#E2E2E2')
      ],
      xaxis2=dict(domain=[0.1, 0.4],
                  anchor='y2',
                  tickfont=dict(size=8)),
      yaxis2=dict(domain=[0.7, 0.95],
                  anchor='x2',
                  title='Daily new cases',
                  tickfont=dict(size=8),
                  titlefont=dict(size=10)))
  return layout


def timeseries_plot(df, time_variable, variable, agg_variable, log_y, title):
  fig = px.line(df,
                x=time_variable,
                y=variable,
                color=agg_variable,
                hover_name=agg_variable,
                line_shape="spline",
                render_mode="svg",
                log_y=log_y,
                color_discrete_sequence=px.colors.qualitative.Pastel)

  add_markers_end(fig)
  apply_base_layout(fig, title)

  return fig


def add_markers_end(fig):
  '''Add markers at the end of the lines'''
  new_trace = deepcopy(fig.data)

  for trace, trace_old in zip(new_trace, fig.data):
    trace['marker'] = dict(size=10,
                           color=trace['line']['color'],
                           symbol='circle')
    trace['line'] = None
    trace['mode'] = 'markers'
    trace['showlegend'] = False
    trace['x'] = np.array([trace_old['x'][-1]])
    trace['y'] = np.array([trace_old['y'][-1]])

    fig.add_trace(trace)


def apply_base_layout(fig, title):
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
              size=10))
  )


def make_fig_fit_base(df):
  prediction, fit = predict_data(df, 'total_cases', days_prediction=60)
  parameters_fit = parameters(prediction, fit)

  # Scatter with the cases
  cases = scatter_cases(prediction, 'total_cases')
  traces_logistic = logistic_curve(prediction, 'total_cases')
  trace_daily = new_cases(prediction, 'total_cases')

  plot_traces = traces_logistic + [cases] + trace_daily

  layout = figure_layout(title='', country=df.location.unique()[0],
                         day_last_string=parameters_fit['day_end'],
                         day_peak_string=parameters_fit['day_peak'],
                         saturation_value=parameters_fit['saturation_value'])

  covid_fig = go.Figure(data=plot_traces, layout=layout)

  return covid_fig


def make_fig_map_base(df, variable):
    out = get_last_valid_data(df, variable='new_cases', time_variable='date')

    fig = px.choropleth(out, locations="iso_code",
                      color=variable,
                      hover_data=['location', 'date'],
                      **plot_opts_global[variable])
    fig.update_geos(projection_type="kavrayskiy7")
    fig.update_layout(coloraxis_colorbar=dict(title=""),
                    height=500,
                    width=800,
                    margin={"r": 0, "t": 50, "l": 0, "b": 0})

    return fig


def make_fig_map_weekly(df, variable):
    with open('NUTS_RG_10M_2021_4326.geojson') as file:
        geojson = json.load(file)

    # Filter geojson to retain only the regions that we need, thus speeding up the plotting
    countries_list_df = df.NUTS.unique()
    geojson_filtered = [feature for feature in geojson['features']
                      if feature['id'] in countries_list_df]
    geojson_filtered_2 = {'crs': {'type': 'name',
                                'properties': {'name': 'urn:ogc:def:crs:EPSG::4326'}},
                        'type': 'FeatureCollection',
                        'features': geojson_filtered}

    out = get_last_valid_data(df, variable='daily_cases', time_variable='Date')

    fig = px.choropleth_mapbox(out, hover_data=['location', 'Date'],
                             geojson=geojson_filtered_2,
                             locations='NUTS',
                             color=variable,
                             mapbox_style="carto-positron",
                             zoom=2.5,
                             center={"lat": 53, "lon": 3.1791},
                             opacity=0.7,
                             title='',
                             **plot_opts[variable])

    fig.update_geos(showcountries=False, showcoastlines=True,
                  showland=False, fitbounds="locations")
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0},
                    coloraxis_colorbar=dict(title=""),
                    height=500, width=800)

    return fig


def make_fig_testing_base(df, variable):
  fig = px.line(df.sort_values(by='date'),
                x="date",
                y=variable,
                color="location",
                hover_name="location",
                line_shape="spline",
                render_mode="svg",
                color_discrete_sequence=px.colors.qualitative.Set2 + px.colors.qualitative.Set3)

  add_markers_end(fig)
  apply_base_layout(fig, title='')

  return fig


def make_fig_hospitalization_base(df, time_variable, value_variable, color_variable, hover_variable):
  fig = px.bar(df,
               x=time_variable,
               y=value_variable,
               color=color_variable,
               hover_name=hover_variable,
               color_discrete_sequence=px.colors.qualitative.Pastel)

  apply_base_layout(fig, title='')

  return fig


def make_dash_table(table_data, id):
    return DataTable(id=id,
                    columns=table_data['columns'],
                    data=table_data['data'],
                    hidden_columns=['total_deaths_change', 'total_cases', 'total_deaths',
                                    'total_cases_per_million', 'total_deaths_per_million',
                                    'new_tests_per_thousand', 'total_tests_per_thousand',
                                    'positive_rate', 'Date', 'date', 'daily_cases_per_million',
                                    'daily_deaths_per_million', 'daily_recovered', 'daily_recovered_per_million',
                                    'total_deaths_change', 'CumulativePositive', 'CumulativePositive_per_million',
                                    'CumulativeDeceased', 'CumulativeDeceased_per_million', 'CumulativeRecovered',
                                    'CumulativeRecovered_per_million', 'CurrentlyPositive', 'CurrentlyPositive_per_million',
                                    'daily_deaths'
                                    ],
                    virtualization=True,
                    style_cell={'textAlign': 'left', 'minWidth': '100px', 
                                'width': '100px', 'maxWidth': '100px', 'font-family': 'verdana',
                                'fontSize':13,},
                    fixed_rows={'headers': True},
                    style_table={'height': 600, 'width': 700},
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    style_data_conditional=[
                                        {
                                      'if': {
                                          'filter_query': '{total_cases_change} > 0',
                                          'column_id': 'total_cases_change'
                                      },
                                      'backgroundColor': 'tomato',
                                      'color': 'white'
                                      }] + 
                                      [
                                      {
                                      'if': {
                                          'filter_query': '{total_cases_change} > 10',
                                          'column_id': 'total_cases_change'
                                      },
                                      'backgroundColor': '#FF4136',
                                      'color': 'white'
                                      }] +
                                      [
                                      {
                                      'if': {
                                          'filter_query': '{total_cases_change} < -1',
                                          'column_id': 'total_cases_change'
                                      },
                                      'backgroundColor': '#3D9970',
                                      'color': 'white'
                                      }],
                      style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold',
                                'whiteSpace': 'normal',
                                'height': 'auto',
                            })

