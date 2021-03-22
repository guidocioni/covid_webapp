from utils import variable_options, variable_options_eu, make_dash_table
import dash_core_components as dcc
from datetime import date
import dash_html_components as html


def get_aggregated_tab(dropdown_options):
    return [
        html.Div([
            html.Div('Select variable to be shown in the map and timeseries',
                     style={'width': '800px', 'margin': '10px'}),
            dcc.Dropdown(
                id='variable-dropdown',
                options=variable_options,
                value="total_cases_change",
                style={'width': '780px', 'margin': '10px'}),
            dcc.Graph(
                id='figure-map-world',
                style={'margin': '10px', 'width': '780px'}
            )],
            style={
            'margin': '10px',
            'text-align': 'center',
            'display': 'inline-block',
            'align-items': 'center',
            'justify-content': 'center',
            'border': '1px solid lightgrey',
            'border-radius': '4px',
            'vertical-align': 'top'}),
        html.Div([
            # html.Div('Select the countries to be shown and the start date for the timeseries'),
            dcc.Dropdown(
                id='country-dropdown-multi',
                options=dropdown_options,
                value=['Germany', 'Italy', 'France', 'Spain'],
                multi=True, style={'width': '400px', 'margin': '10px', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div(['Start date:  ',
                      dcc.DatePickerSingle(
                          id='date-picker-single',
                          min_date_allowed='2019-12-31',
                          max_date_allowed=date.today().strftime('%Y-%m-%d'),
                          date='2020-07-01',
                          display_format='DD MMM YYYY',
                          placeholder='Starting date',
                          style={'margin-right': '5px', 'width': '50px', 'display': 'inline-block'}),
                      dcc.Checklist(
                          id='log_y_on',
                          options=[
                              {'label': 'Log Y Axis', 'value': 'log_y'}
                          ],
                          value='', style={'margin-left': '80px', 'width': '100px', 'display': 'inline-block'}), ],
                     style={'display': 'inline-block', 'margin': '10px', 'vertical-align': 'top'}),
            dcc.Graph(
                id='figure-cases',
                style={'width': '800px', 'margin': '10px'}
            )
        ], style={'display': 'inline-block',
                  'text-align': 'center',
                  'align-items': 'center',
                  'justify-content': 'center',
                  'border': '1px solid lightgrey',
                  'border-radius': '4px',
                  'margin': '10px'
                  }),
    ]


def get_aggregated_eu_tab(region_options):
    return [
        html.Div([
            html.Div('Select variable to be shown in the map and timeseries',
                     style={'width': '800px', 'margin': '10px'}),
            dcc.Dropdown(
                id='variable-dropdown-eu',
                options=variable_options_eu,
                value="total_cases_change",
                style={'width': '780px', 'margin': '10px'}),
            dcc.Graph(
                id='figure-map-eu',
                style={'margin': '10px', 'width': '780px'}),
        ],
            style={
            'margin': '10px',
            'text-align': 'center',
            'display': 'inline-block',
            'align-items': 'center',
            'justify-content': 'center',
            'border': '1px solid lightgrey',
            'border-radius': '4px',
            'vertical-align': 'top'}),
        html.Div([
            dcc.Dropdown(
                id='region-dropdown-eu',
                options=region_options,
                value=['Italy | Lombardia', 'Italy | Toscana'],
                multi=True,
                style={'width': '400px', 'margin': '10px', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div(['Start date:  ',
                      dcc.DatePickerSingle(
                          id='date-picker-single-2',
                          min_date_allowed='2019-12-31',
                          max_date_allowed=date.today().strftime('%Y-%m-%d'),
                          date='2020-07-01',
                          display_format='DD MMM YYYY',
                          placeholder='Starting date',
                          style={'margin-right': '5px', 'width': '50px', 'display': 'inline-block'}), ],
                     style={'display': 'inline-block', 'margin': '10px', 'vertical-align': 'top'}),
            dcc.Graph(
                id='figure-eu',
                style={'width': '800px', 'margin': '10px'}
            )
        ], style={'display': 'inline-block',
                  'text-align': 'center',
                  'align-items': 'center',
                  'justify-content': 'center',
                  'border': '1px solid lightgrey',
                            'border-radius': '4px',
                            'margin': '10px'}),
    ]


def get_hosp_tab(dropdown_options, dropdown_options_2, region_options):
    return [
        html.Div(
            [
                dcc.Dropdown(
                    id='country-dropdown-3',
                    options=dropdown_options_2,
                    value='Italy',
                    multi=False, style={'width': '780px', 'margin': '10px'}),
                dcc.Graph(
                    id='figure-hospitalization',
                    style={'width': '800', 'margin': 10}
                )
            ], style={'display': 'inline-block', 'margin': 10,
                      'border': '1px solid lightgrey',
                      'border-radius': '4px', }),
        html.Div(
            [
                dcc.Dropdown(
                    id='region-dropdown-eu-2',
                    options=region_options,
                    value='Italy | Lombardia',
                    multi=False, style={'width': '780px', 'margin': '10px'}),
                dcc.Graph(
                    id='figure-hospitalization-eu',
                    style={'width': '800', 'margin': 10}
                )
            ], style={'display': 'inline-block', 'margin': 10,
                      'border': '1px solid lightgrey',
                      'border-radius': '4px', })
    ]


def get_forecast_tab(dropdown_options):
    return [
        html.Div([
            dcc.Dropdown(
                id='country-dropdown-1',
                options=dropdown_options,
                value='Italy',
                style={'width': '780px', 'margin': '10px'}),
            dcc.Graph(
                id='figure-fit-1',
                style={'width': '800', 'margin': 10}
            )],
            style={'display': 'inline-block', 'margin': 10,
                   'border': '1px solid lightgrey',
                   'border-radius': '4px', }),
        html.Div([
            dcc.Dropdown(
                id='country-dropdown-2',
                options=dropdown_options,
                value='Germany',
                style={'width': '780px', 'margin': '10px'}),
            dcc.Graph(
                id='figure-fit-2',
                style={'width': '800', 'margin': 10}
            )],
            style={'display': 'inline-block', 'margin': 10,
                   'border': '1px solid lightgrey',
                   'border-radius': '4px', }),
        html.Div(
            [
                html.Div('The points show the daily cumulated cases, while the line shows the logistic fit with uncertainty (shaded area).', style={
                    'margin-left': 10}),
                html.Div('In the right inset the parameters obtained from the logistic fit are shown: note that these are only parameters, not reliable forecasts!', style={
                    'margin-left': 10}),
                html.Div('The value of R2 is also shown: the closer to 1 the better the fit.', style={
                    'margin-left': 10}),
                html.Div(['Here is an explanation of the parameters: ',
                          html.Ul(children=[
                              html.Li(
                                  'End = First day without new infections with a threshold of 1/100 on the asymptotic value'),
                              html.Li(
                                  'Peak day = Estimated day with maximum growth rate '),
                              html.Li('Max. infected = Asymptotic value for confirmed cases')])], style={'margin-left': 10}),
                html.Div(
                    'You can choose up to 2 countries to compare side by side.', style={'margin-left': 10}),
            ], style={'margin': 10,
                      'border': '1px solid lightgrey',
                      'border-radius': '4px', })
    ]


def get_table_tab(table_data, table_data_eu):
    df = table_data["df"]
    df = df[df.location == "World"]
    return [
        html.Div([
            html.Div([html.H2("🌍 %d new cases" % df.new_cases), html.P("(%4.2f change)" % df.total_cases_change)],
                     style={'display': 'inline-block', 'padding': 10}),
            html.Div([html.H2("%d new deceased" % df.new_deaths), html.P("(%4.2f change)" % df.total_deaths_change)],
                     style={'display': 'inline-block', 'padding': 10}),
        ]
        ),
        html.Div([
            html.H5('Countries in the World', style={
                    'float': 'right', 'margin': 10}),
            make_dash_table(table_data, id='table')],
            style={'display': 'inline-block', 'margin': '10px'}),
        # html.Div('The worst regions in Europe in the latest update'),
        html.Div([
            html.H5('Regions in Europe', style={
                    'float': 'right', 'padding-right': 10}),
            make_dash_table(table_data_eu, id='table-eu')],
            style={'display': 'inline-block', 'margin': '10px'}),

    ]
