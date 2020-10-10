import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import plotly.graph_objs as go
import plotly.express as px

def compute_r0(group, window=5):
    # Compute R0 using RKI method 
    r0 = []
    for t in range((2 * window)-1, len(group)):
        r0.append(group.iloc[t-window+1:t+1]['cases'].sum() / group.iloc[t-(2*window)+1:t-window+1]['cases'].sum())
    
    return pd.DataFrame(data={'r0':r0}, index=group.index[(2 * window)-1:])


def logistic_model(x, a, b, c):
  return c / (1 + np.exp(-(x - b) / a))


def exponential_model(x, a, b, c):
  return a * np.exp(b * (x - c))


def r2(y_obs, y_model):
  return 1. - (np.sum((y_obs - y_model)**2) / np.sum(
      (y_obs - np.mean(y_obs))**2))


def do_fit(x, y, p0_log=[2, 60, 20000], p0_exp=[1e-4, 1e-6, 1e-6]):
  '''Perform logistic and exponential fit on x, y
  Initial parameters are provided '''
  log_fit = curve_fit(logistic_model, x, y, p0_log)
  exp_fit = curve_fit(exponential_model, x, y, p0_exp)

  return log_fit, exp_fit


def parameters(log_fit, start_date='2020-02-01 00:00:00'):
  '''obtain some important parameters from the fit'''
  # Parameters from the logistic fit
  a, b, c = log_fit[0]
  day_last_index = int(
      fsolve(lambda x: logistic_model(x, a, b, c) - int(c), b))
  try:
    day_last = pd.date_range(start=pd.to_datetime(start_date),
                             periods=1000)[day_last_index]
  except IndexError:
    day_last = pd.date_range(start=pd.to_datetime(start_date),
                             periods=1000)[-1]

  try:
    day_peak = pd.date_range(start=pd.to_datetime(start_date),
                           periods=1000)[b.round().astype(int)]
  except IndexError:
    day_peak = pd.date_range(start=pd.to_datetime(start_date),
                             periods=1000)[-1]

  speed = a
  saturation_value = c

  return day_last, day_peak, speed, saturation_value


def scatter_cases(x, y, color="blue", symbol="circle", hovertext=None):
  '''Create a trace for the scatter of the cases'''
  trace = go.Scatter(name='Confirmed cases',
                     x=x,
                     y=y,
                     mode="markers",
                     showlegend=True,
                     marker=dict(size=8,
                                 color=color,
                                 opacity=0.6,
                                 symbol=symbol),
                     hovertext=hovertext)

  return trace


def scatter_deaths(x, y, color="red", symbol="circle", hovertext=None):
  '''Create a trace for the scatter of the deaths'''
  trace = go.Scatter(name='Confirmed deaths',
                     x=x,
                     y=y,
                     mode="markers",
                     showlegend=True,
                     marker=dict(size=8,
                                 color=color,
                                 opacity=0.6,
                                 symbol=symbol),
                     hovertext=hovertext)

  return trace


def logistic_curve(x, r2, log_fit, color='firebrick', linewidth=2):
  '''Plot logistic together with range plot for uncertainty'''
  # Compute standard deviation for parameters for range plot

  trace_log = go.Scatter(name='Logistic fit, R2=%3.3f' % r2,
                         x=x,
                         y=logistic_model(x, *log_fit[0]),
                         mode="lines",
                         showlegend=True,
                         line=dict(color=color, width=linewidth))

  trace_log_p1sigma = go.Scatter(
      x=x,
      showlegend=False,
      y=logistic_model(x, *(log_fit[0] + np.sqrt(np.diag(log_fit[1])))),
      mode="lines",
      line=dict(color=color, width=0))
  trace_log_m1sigma = go.Scatter(
      x=x,
      showlegend=False,
      y=logistic_model(x, *(log_fit[0] - np.sqrt(np.diag(log_fit[1])))),
      mode="lines",
      line=dict(color=color, width=0),
      fill='tonexty')

  return [trace_log, trace_log_p1sigma, trace_log_m1sigma]


def exponential_curve(x, r2, exp_fit, color='green', linewidth=2):
  '''Plot the exponential fit '''

  trace_exponential = go.Scatter(name='Exponential fit,  R2=%3.3f' % r2,
                                 x=x,
                                 y=exponential_model(x, *exp_fit[0]),
                                 mode="lines",
                                 showlegend=True,
                                 line=dict(color=color, width=linewidth))

  return trace_exponential


def new_cases(x, y):
  trace = go.Bar(name='Daily cases variation',
                 x=x,
                 y=y,
                 xaxis='x2',
                 yaxis='y2',
                 showlegend=False)
  return trace


def prepare_data(df, threshold_cases=10):
  '''Prepare the data needed for the fit. This should be
  already filtered by country.'''
  # We use a fictive data just to have x>>0
  subset = df[df.cumulative_cases > threshold_cases]
  beginning_date = subset.iloc[0].dateRep
  date_shift = beginning_date - pd.to_timedelta('20 days')
  subset.loc[:, 'days'] = subset.loc[:, 'dateRep'] - date_shift

  return subset

def get_countries():
  df = pd.read_csv(
      'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv/',
      usecols=[10])

  return list(df.countriesAndTerritories.unique())


def compute_cumulative(df):
  df = df[['cases', 'deaths'
           ]].sort_index().cumsum().rename(columns={
               'cases': 'cumulative_cases',
               'deaths': 'cumulative_deaths'
           })

  return df


def compute_percentage(df):
  df = (df[['cumulative_cases',
            'cumulative_deaths']].sort_index().pct_change() * 100).rolling(10).mean().rename(columns={
                'cumulative_cases': 'cumulative_cases_pct_change',
                'cumulative_deaths': 'cumulative_deaths_pct_change'
            })

  return df


def figure_layout(title, country, xmax, ymax, xmin, ymin, day_last_string,
                  day_peak_string, shift_day_string, speed, saturation_value):
  '''Set the figure layout'''
  layout = dict(
      title=title,
      showlegend=True,
      height=500,
      width=800,
      margin=dict(r=5, b=20, t=30, l=10),
      font=dict(family='arial'),
      yaxis=dict(range=[ymin, ymax], title='Cases (%s)' % country),
      xaxis=dict(range=[xmin, xmax], title='Days from %s' % shift_day_string),
      legend=dict(x=0.75, y=0.1),
      annotations=[
          dict(
              x=0.95,
              y=0.4,
              xref="paper",
              yref="paper",
              text="<b>Fit parameters (logistic)</b> <br> End = %s <br> Infection speed =  %3.1f <br> Peak day = %s <br> Max. infected = %d"
              % (day_last_string, speed, day_peak_string, saturation_value),
              showarrow=False,
              bgcolor='#E2E2E2')
      ],
      xaxis2=dict(domain=[0.1, 0.4], anchor='y2', tickfont=dict(size=8)),
      yaxis2=dict(domain=[0.7, 0.95],
                  anchor='x2',
                  title='Daily new cases',
                  tickfont=dict(size=8),
                  titlefont=dict(size=10)))
  return layout

def make_fig_fit_base(df):
  subset = prepare_data(df, threshold_cases=20)
  shift_day = subset['dateRep'].iloc[0] - subset['days'].iloc[0]

  x = subset.days.dt.days.values
  y = subset.cumulative_cases.values

  log_fit, exp_fit = do_fit(x, y)

  day_last, day_peak, speed, saturation_value = parameters(
      log_fit, shift_day)

  # y values predicted by logistic and exponential fit, computed on real x values
  y_pred_logistic = logistic_model(x, *log_fit[0])
  # y_pred_exp = exponential_model(x, *exp_fit[0])

  # Just a discrete x to plot the values outside of the observed range
  x_pred = np.arange(x.min() - 10, x.max() + 40)

  # Scatter with the cases
  cases = scatter_cases(x, y, hovertext=subset.dateRep)

  deaths = scatter_deaths(x, subset.cumulative_deaths.values,
                          hovertext=subset.dateRep)

  traces_logistic = logistic_curve(x_pred, r2(y, y_pred_logistic), log_fit)

  # trace_exponential = exponential_curve(x_pred, r2(y, y_pred_exp), exp_fit)

  trace_daily = new_cases(x, np.diff(y))

  # traces_logistic.append(trace_exponential)
  traces_logistic.append(cases)
  traces_logistic.append(deaths)
  traces_logistic.append(trace_daily)

  data = traces_logistic

  xmax, ymax = x_pred.max() - 10, y.max() + 10000
  xmin, ymin = 20, -1000

  layout = figure_layout(title='', country=df.countriesAndTerritories.unique()[0],
                         xmax=xmax,
                         xmin=xmin,
                         ymax=ymax,
                         ymin=ymin,
                         day_last_string=day_last.strftime('%d %b %Y'),
                         day_peak_string=day_peak.strftime('%d %b %Y'),
                         shift_day_string=shift_day.strftime('%d %b %Y'),
                         speed=speed,
                         saturation_value=saturation_value)

  covid_fig = go.Figure(data=data, layout=layout)

  return covid_fig
