import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import plotly.graph_objs as go
import plotly.express as px
import json


def compute_r0(group, window=7):
  # Compute R0 using RKI method
  r0 = []
  for t in range((2 * window) - 1, len(group)):
    r0.append(group.iloc[t - window + 1:t + 1]['cases'].sum() /
              group.iloc[t - (2 * window) + 1:t - window + 1]['cases'].sum())

  return pd.DataFrame(data={'r0': r0}, index=group.index[(2 * window) - 1:])


def logistic_model(x, a, b, c, d):
  return c / (1 + np.exp(-(x - b) / a)) + d


def exponential_model(x, a, b, c):
  return a * np.exp(b * (x - c))


def r2(y_obs, y_model):
  return 1. - (np.sum((y_obs - y_model)**2) / np.sum(
      (y_obs - np.mean(y_obs))**2))


def do_fit(x, y, p0_log=[3, 10, 150000, 236989], maxfev=50000):
  '''Perform logistic and exponential fit on x, y
  Initial parameters are provided '''
  log_fit = curve_fit(f=logistic_model, xdata=x, ydata=y,
                      p0=p0_log, maxfev=maxfev)
  return log_fit


def parameters(log_fit, start_date='2020-06-01 00:00:00'):
  '''obtain some important parameters from the fit'''
  # Parameters from the logistic fit
  a, b, c, d = log_fit[0]
  day_last_index = int(
      fsolve(lambda x: logistic_model(x, a, b, c, d) - int(c + d), b))
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
  saturation_value = c + d

  return day_last, day_peak, speed, saturation_value


def scatter_cases(x, y, color='rgb(252, 141, 98)', symbol="circle", hovertext=None):
  '''Create a trace for the scatter of the cases'''
  trace = go.Scatter(name='Confirmed cases',
                     x=x,
                     y=y,
                     mode="markers",
                     showlegend=True,
                     marker=dict(size=8,
                                 color=color,
                                 opacity=0.8,
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


def logistic_curve(x, r2, log_fit, color='rgb(251, 180, 174)', linewidth=2):
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


def new_cases(x, y, hovertext=None):
  trace = go.Bar(name='Daily cases variation',
                 x=x,
                 y=y,
                 xaxis='x2',
                 yaxis='y2',
                 showlegend=False,
                 hovertext=hovertext)
  return trace


def prepare_data(df):
  '''Prepare the data needed for the fit. This should be
  already filtered by country.'''
  # We decide which date to start from finding the minimum of cases
  subset = df.copy()
  beginning_date = subset[subset.index == subset.cases.rolling(
      7).mean().idxmin()].dateRep.values[0]
  subset = subset[subset.dateRep >= beginning_date]
  subset['cumulative_cases'] = subset['cumulative_cases'] - \
      subset['cumulative_cases'].min()
  date_shift = beginning_date - pd.to_timedelta('20 days')
  subset.loc[:, 'days'] = subset.loc[:, 'dateRep'] - date_shift

  return subset


def get_countries():
  df = pd.read_csv(
      'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv/',
      usecols=[10])

  return list(df.countriesAndTerritories.unique())


def figure_layout(title, country, xmax, ymax, xmin, ymin, day_last_string,
                  day_peak_string, shift_day_string, speed, saturation_value):
  '''Set the figure layout'''
  layout = dict(
      template='plotly_white',
      title=title,
      showlegend=True,
      height=500,
      width=800,
      margin=dict(r=5, b=20, t=30, l=10),
      font=dict(family='arial'),
      yaxis=dict(
          range=[ymin, ymax], title='Cumulative Cases (%s) w.r.t. initial date' % country),
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
  subset = prepare_data(df)
  shift_day = subset['dateRep'].iloc[0] - subset['days'].iloc[0]

  x = subset.days.dt.days.values
  y = subset.cumulative_cases.values

  log_fit = do_fit(x, y, p0_log=[3, 10, 100000, y.min()])

  day_last, day_peak, speed, saturation_value = parameters(log_fit, shift_day)

  # y values predicted by logistic and exponential fit, computed on real x values
  y_pred_logistic = logistic_model(x, *log_fit[0])

  # Just a discrete x to plot the values outside of the observed range
  x_pred = np.arange(x.min() - 10, x.max() + 50)

  # Scatter with the cases
  cases = scatter_cases(x, y, hovertext=subset.dateRep)

  traces_logistic = logistic_curve(x_pred, r2(y, y_pred_logistic), log_fit)

  trace_daily = new_cases(x, subset.cases, hovertext=subset.dateRep)

  traces_logistic.append(cases)
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


def make_fig_map_base(df, variable):
  fig = px.choropleth(df, locations="countryterritoryCode",
                      color=variable,
                      hover_name="countriesAndTerritories",
                      animation_frame=df.dateRep.astype(str),
                      color_continuous_scale="YlOrRd")
  fig.update_geos(projection_type="kavrayskiy7")
  fig.update_layout(coloraxis_colorbar=dict(title=""),
                    height=500,
                    width=800, margin={"r": 0, "t": 50, "l": 0, "b": 0})
  fig['layout']['updatemenus'][0]['pad'] = dict(r=10, t=0)
  fig['layout']['sliders'][0]['pad'] = dict(r=10, t=0,)
  fig.layout.sliders[0]['active'] = len(fig.frames) - 1
  fig.update_traces(
      z=fig.frames[-1].data[0].z, hovertemplate=fig.frames[-1].data[0].hovertemplate)

  return fig


def make_fig_map_weekly(df):
  with open('NUTS_RG_10M_2021_4326.geojson') as file:
    geojson = json.load(file)

  # Filter geojson to retain only the regions that we need, thus speeding up the plotting
  countries_list_df = df.nuts_code.unique()
  geojson_filtered = [feature for feature in geojson['features'] if feature['id'] in countries_list_df]
  geojson_filtered_2 = {'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::4326'}},
   'type': 'FeatureCollection',
   'features':geojson_filtered}

  fig = px.choropleth_mapbox(df, geojson=geojson_filtered_2,
                             locations='nuts_code', color='rate_14_day_per_100k',
                             mapbox_style="carto-positron",
                             zoom=2.5, center={"lat": 51.4816, "lon": 3.1791},
                             opacity=0.7,
                             color_continuous_scale="YlOrRd", title='14 days reporting ratio per 100k')

  fig.update_geos(showcountries=False, showcoastlines=True,
                  showland=False, fitbounds="locations")
  fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0},
    coloraxis_colorbar=dict(title=""),
    height=500, width=800,)

  return fig

def make_fig_testing_base(df, variable):
  fig = px.line(df.sort_values(by='year_week'),
            x="year_week",
            y=variable,
            color="country",
            hover_name="country",
            line_shape="spline",
            render_mode="svg",
            color_discrete_sequence=px.colors.qualitative.Set2 + px.colors.qualitative.Set3)

  fig.update_layout(
  template='plotly_white',
  legend_orientation="h",
  width=800,
  height=500,
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

def make_fig_hospitalization_base(df):
  fig = px.bar(df,
            x="date",
            y="value",
            color="indicator",
            hover_name="country",
            color_discrete_sequence=px.colors.qualitative.Pastel)

  fig.update_layout(
  template='plotly_white',
  legend_orientation="h",
  width=800,
  height=500,
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
