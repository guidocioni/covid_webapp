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

def parameters(df, log_fit):
    a, b, c, d = log_fit[0]
    day_last_index = int(
      fsolve(lambda x: logistic_model(x, a, b, c, d) - int((c + d) - (c+d)/100), b))
    day_last = pd.date_range(start=df['dateRep'].iloc[0],
                             periods=1000)[day_last_index]\
                            - pd.to_timedelta('20 days')
    day_peak = pd.date_range(start=df['dateRep'].iloc[0],
                                 periods=1000)[b.round().astype(int)] \
                                - pd.to_timedelta('20 days')
        
    saturation_value = c+d
    
    return {
        'day_peak' : pd.to_datetime(day_peak).strftime('%d %b %Y'),
        'day_end' : pd.to_datetime(day_last).strftime('%d %b %Y'),
        'saturation_value' :saturation_value}


def scatter_cases(df, variable, 
                  color='rgb(252, 141, 98)', symbol="circle"):
    '''Create a trace for the scatter of the cases'''
    trace = go.Scatter(name=variable,
                     x=df['dateRep'],
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
    '''Plot logistic together with range plot for uncertainty'''
    r2_value = r2(df['cumulative_cases'], df['cumulative_cases_prediction'])
    trace_log = go.Scatter(name='Logistic fit, R2=%3.3f' % r2_value,
                         x=df['dateRep'],
                         y=df['cumulative_cases_prediction'],
                         mode="lines",
                         showlegend=True,
                         line=dict(color=color, width=linewidth))

    trace_log_p1sigma = go.Scatter(
      x=df['dateRep'],
      showlegend=False,
      y=df[variable+'_prediction_upper'],
      mode="lines",
      line=dict(color=color, width=0))
    trace_log_m1sigma = go.Scatter(
      x=df['dateRep'],
      showlegend=False,
      y=df[variable+'_prediction_lower'],
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



def new_cases(df, variable, color='rgb(252, 141, 98)',):
    scatter = go.Bar(name=variable+' variation',
                 x=df['dateRep'],
                 y=df[variable].diff(),
                 xaxis='x2',
                 yaxis='y2',
                 showlegend=False,
                    marker_color=color)
    line = go.Scatter(
                 x=df['dateRep'],
                 y=df[variable+'_prediction'].diff(),
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
    beginning_date = subset[subset.index == subset.cases.rolling(
      7).mean().idxmin()].dateRep.values[0]
    subset = subset[subset.dateRep >= beginning_date]
    date_shift = beginning_date - pd.to_timedelta('20 days')
    subset.loc[:, 'days'] = subset.loc[:, 'dateRep'] - date_shift

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
    {'dateRep' : pd.date_range(start=input_df['dateRep'].max() + pd.to_timedelta('1 D'),
              periods=days_prediction),
     'days' : pd.timedelta_range(input_df['days'].max() + pd.to_timedelta('1 D'),
                                 periods=days_prediction)
    })

    final = pd.concat([input_df, future_df]).reset_index(drop=True)
    final[variable+'_prediction'] = logistic_model(final['days'].dt.days.values,
                                                      *log_fit[0])
    final[variable+'_prediction_upper'] = logistic_model(final['days'].dt.days.values,
                                                      *(log_fit[0] + np.sqrt(np.diag(log_fit[1]))))
    final[variable+'_prediction_lower'] = logistic_model(final['days'].dt.days.values,
                                                      *(log_fit[0] - np.sqrt(np.diag(log_fit[1]))))
    return final, log_fit


def get_countries():
  df = pd.read_csv(
      'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv/',
      usecols=[10])

  return list(df.countriesAndTerritories.unique())


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
      xaxis2=dict(domain=[0.1, 0.4], anchor='y2', tickfont=dict(size=8)),
      yaxis2=dict(domain=[0.7, 0.95],
                  anchor='x2',
                  title='Daily new cases',
                  tickfont=dict(size=8),
                  titlefont=dict(size=10)))
    return layout



def make_fig_fit_base(df):
    prediction, fit = predict_data(df, 'cumulative_cases', days_prediction=60)
    parameters_fit = parameters(prediction, fit)

    # Scatter with the cases
    cases = scatter_cases(prediction, 'cumulative_cases')
    traces_logistic = logistic_curve(prediction, 'cumulative_cases')
    trace_daily = new_cases(prediction, 'cumulative_cases')

    plot_traces = traces_logistic + [cases] + trace_daily

    layout = figure_layout(title='', country=df.countriesAndTerritories.unique()[0],
                         day_last_string=parameters_fit['day_end'],
                         day_peak_string=parameters_fit['day_peak'],
                         saturation_value=parameters_fit['saturation_value'])

    covid_fig = go.Figure(data=plot_traces, layout=layout)

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
