from utils import threshold_chosen, variable_options_2, variable_options, variable_options_eu, make_dash_table
import dash_core_components as dcc
from datetime import date
import dash_html_components as html


def get_aggregated_tab(dropdown_options):
    return [
              html.Div('Click on the legend items to show/hide countries and/or select the countries from the dropdown.'),
              html.Div(
                  'Only countries with more than %s cases at the latest update are included in the search.' % threshold_chosen),
              html.Br(),
              html.Div(
                  [dcc.Dropdown(
                                id='country-dropdown-multi',
                                options=dropdown_options,
                                value=['Germany', 'Italy', 'France',
                                       'Spain', 'Portugal'],
                                multi=True, style={'width': '800px' , 'padding': '2px'}),
                  html.Div(['Start date:  ',
                  dcc.DatePickerSingle(
                                      id='date-picker-single',
                                      min_date_allowed='2019-12-31',
                                      max_date_allowed=date.today().strftime('%Y-%m-%d'),
                                      date='2020-04-01',
                                      display_format='DD MMM YYYY',
                                      placeholder='Starting date', 
                                      style={'padding': '2px'})
                    ]),
]),
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
          ]


def get_aggregated_eu_tab(region_options):
    return [  
            html.Div(
                  [
                    dcc.Dropdown(
                        id='region-dropdown-eu',
                        options=region_options,
                        value=['Italy | Lombardia', 'Italy | Toscana'],
                        multi=True, style={'width': '800px', 'padding':'2px'}),
                    dcc.Dropdown(
                        id='variable-dropdown-eu',
                        options=variable_options_eu,
                        value='total_cases_change',
                        multi=False, style={'width': '800px', 'padding':'2px'}),
                    dcc.Graph(
                        id='figure-eu',
                        style={'width': '800'}),
                  ], 
                  style={'display': 'inline-block', 'padding': 10}),
             html.Div(
                      dcc.Graph(
                                id='figure-r0-eu',
                                style={'width': '800'}),
                  style={'display': 'inline-block', 'padding': 10}),
             html.Div(
              [
                dcc.Dropdown(
                        id='region-dropdown-eu-2',
                        options=region_options,
                        value='Italy | Lombardia',
                        multi=False, style={'width': '800px'}),
                dcc.Graph(
                        id='figure-hospitalization-eu',
                        style={'width': '800'}
                  )
              ], style={'display': 'inline-block', 'padding': 10})
                  ]

def get_testing_tab(dropdown_options, dropdown_options_2):
    return [
            # html.Div(['Testing data is weekly. Here is an explanation of the parameters: ',
            #       html.Ul(children=[
            #                         html.Li('positivity rate - 100 x Number of new confirmed cases/number of tests'),
            #                         html.Li('tests done - total tests performed in a specific country'),
            #                         html.Li('testing rate - Testing rate per 100 000 population')])]),
            html.Div('Hospitalization data are either weekly or daily depending on the country selected in the dropdown'),
            html.Div(
                  [
                    dcc.Dropdown(
                        id='country-dropdown-testing',
                        options=dropdown_options,
                        value=['Austria','Germany'],
                        multi=True, style={'width': '800px', 'padding':'4px'}),
                    dcc.Dropdown(
                        id='variable-dropdown-2',
                        options=variable_options_2,
                        value='positive_rate',
                        multi=False, style={'width': '800px', 'padding':'4px'}),
                    dcc.Graph(
                        id='figure-testing',
                        style={'width': '800'}
                  )
                  ], 
                  style={'display': 'inline-block', 'padding': 10}),
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
            ]

def get_forecast_tab(dropdown_options):
    return [
              html.Div('The points show the daily cumulated cases, while the line shows the logistic fit with uncertainty (shaded area).\
                   In the right inset the parameters obtained from the logistic fit are shown: note that these are only parameters, not reliable forecasts!\
                    The value of R2 is also shown: the closer to 1 the better the fit.'),
              html.Div(['Here is an explanation of the parameters: ',
                  html.Ul(children=[
                                    html.Li('End = First day without new infections with a threshold of 1/100 on the asymptotic value'),
                                    html.Li(
                                        'Peak day = Estimated day with maximum growth rate '),
                                    html.Li('Max. infected = Asymptotic value for confirmed cases')])]),
              html.Div(
                  'You can choose up to 2 countries to compare side by side.'),
              html.Br(),
              html.Div([
                      dcc.Dropdown(
                        id='country-dropdown-1',
                        options=dropdown_options,
                        value='Austria'),
                      dcc.Graph(
                        id='figure-fit-1',
                        style={'width': '800'}
                  )],
                  style={'display': 'inline-block', 'padding': 10}),
              html.Div([
                        dcc.Dropdown(
                            id='country-dropdown-2',
                            options=dropdown_options,
                            value='Germany'),
                       dcc.Graph(
                          id='figure-fit-2',
                          style={'width': '800'}
                  )],
                  style={'display': 'inline-block', 'padding': 10})
          ]

def get_maps_tab():
    return [
            html.Div('Shown is the geographical distribution of many variables. \
              In the first plot you can select the variable to be plotted and explore the daily variation using the slider.\
              In the second plot only the most recent data for every region in Europe is shown.'),
            html.Div([
                  dcc.Dropdown(
                      id='variable-dropdown',
                      options=variable_options,
                      value="new_cases_smoothed_per_million"),
                  dcc.Graph(
                      id='figure-map-world',
                      style={'width': '800'}
                  )],
                  style={'display': 'inline-block', 'padding': 10}),
            html.Div([
                      dcc.Dropdown(
                              id='variable-dropdown-map-eu',
                              options=variable_options_eu,
                              value="total_cases_change"),
                      dcc.Graph(
                          id='figure-map-eu',
                          style={'width': '800', 'padding-bottom': '10px'}),
                      ],
                  style={'display': 'inline-block'}),
          ]

def get_table_tab(table_data, table_data_eu):
    df = table_data["df"]
    df = df[df.location == "World"]
    return [
              html.Div([html.H1("🌍 %d new cases" % df.new_cases), html.P("(%4.2f change)" % df.total_cases_change)],
                        style={'display': 'inline-block', 'padding': 10}),
              html.Div([html.H1("%d new deceased" % df.new_deaths), html.P("(%4.2f change)" % df.total_deaths_change)],
                        style={'display': 'inline-block', 'padding': 10}),
              html.Div('The worst countries in the world in the latest update'),
              html.Div(make_dash_table(table_data, id='table'), 
                       style={'display': 'inline-block', 'padding': 10}),
              html.Div('The worst regions in Europe in the latest update'),
              html.Div(make_dash_table(table_data_eu, id='table-eu'), 
                        style={'display': 'inline-block', 'padding': 10}),

            ]