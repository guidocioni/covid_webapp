import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from utils import *
from data import *
from meta import tags
from tabs import get_aggregated_tab, get_hosp_tab,\
    get_forecast_tab, get_table_tab,\
    get_aggregated_eu_tab

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                meta_tags=tags,
                suppress_callback_exceptions=True,
                url_base_pathname='/covid/')
server = app.server
app.title = 'COVID-19 live forecast'


def serve_layout():
    dropdown_options = []
    countries_list = get_countries_list(read_owid(), "location")
    for cnt in countries_list:
        dropdown_options.append({"label": cnt, "value": cnt})

    dropdown_options_2 = []
    countries_list_2 = get_countries_list(read_hospitalization(), "country")
    for cnt in countries_list_2:
        dropdown_options_2.append({"label": cnt, "value": cnt})

    region_eu = []
    region_eu_list = get_countries_list(read_jrc(), "location")
    for cnt in region_eu_list:
        region_eu.append({"label": cnt, "value": cnt})

    return html.Div(children=[
        html.Div([html.H1('COVID-19 Monitoring', style={'display': 'inline'}), html.H5('Last update: %s' % filter_data().date.max().strftime('%a %d %b %Y') )]),
        html.P('Data is obtained from the European Center for Disease Monitoring (ECDC)'),
        html.P('Although new data is downloaded and updated every 2 hours in this dashboard it may not reflect any change in the source.\
         Moreover data during the weekend and in the morning may be incomplete.'),
        html.Hr(),
        #
        dcc.Tabs(parent_className='custom-tabs',
                 # className='custom-tabs-container',
                 children=[
                    dcc.Tab(label='Daily overview',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_table_tab(make_table_data(), make_table_data_eu())),
                     # -----------------------------------------------------  #
                     dcc.Tab(label='Global',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_aggregated_tab(dropdown_options)),
                     # -----------------------------------------------------  #
                     dcc.Tab(label='EU regions',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_aggregated_eu_tab(region_eu)),
                     # -----------------------------------------------------  #
                     dcc.Tab(label='Hospitalization',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_hosp_tab(dropdown_options,
                                                    dropdown_options_2,
                                                    region_eu)),
                     # -----------------------------------------------------  #
                     dcc.Tab(label='Forecast',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_forecast_tab(dropdown_options)),
                 ]),
        html.Div(html.A('Created by Guido Cioni', href='http://guidocioni.altervista.org/nuovosito/'))
    ],
        style={'width': '100%', 'display': 'inline-block'})


app.layout = serve_layout


@app.callback(
    Output('figure-map-eu', 'figure'),
    [Input('variable-dropdown-eu', 'value')])
def make_fig_map_weekly_europe(variable):
    df = read_jrc()

    return make_fig_map_weekly(df, variable)


@app.callback(
    Output('figure-eu', 'figure'),
    [Input('region-dropdown-eu', 'value'), 
     Input('variable-dropdown-eu', 'value'),
     Input('date-picker-single-2', 'date'),])
def make_fig_eu(regions, variable, date_value):
    '''Give as input a threshold for the cumulative cases in the most updated
    timestep to filter out countries that do not have many cases.'''
    df = read_jrc()
    df = df.loc[df.location.isin(regions)]
    df = df.loc[df.Date > date_value]

    for v in variable_options_eu:
      if v['value'] == variable:
        title = v['label']

    fig = timeseries_plot(df,
                          time_variable="Date",
                          variable=variable,
                          agg_variable="location",
                          log_y=False,
                          title=title)

    return fig


@app.callback(
    Output('figure-hospitalization-eu', 'figure'),
    [Input('region-dropdown-eu-2', 'value')])
def make_fig_hospitalization_eu(region):
    df = read_jrc()
    df = df.loc[df.location == region]

    df = df[['location', 'Date', 'IntensiveCare', 'Hospitalized']]\
        .set_index(['location', 'Date'])\
        .stack().reset_index()\
        .rename(columns={'level_2': 'indicator', 0: 'value'})

    return make_fig_hospitalization_base(df,
                                         "Date",
                                         "value",
                                         "indicator",
                                         "location")


@app.callback(
    Output('figure-hospitalization', 'figure'),
    [Input('country-dropdown-3', 'value')])
def make_fig_hospitalization(country):
    df = read_hospitalization()

    return make_fig_hospitalization_base(df[df.country == country],
                                         "date",
                                         "value",
                                         "indicator",
                                         "country")


@app.callback(
    Output('figure-map-world', 'figure'),
    [Input('variable-dropdown', 'value')])
def make_fig_map_world(variable):
    df = filter_data(start_date='2020-06-01')

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
    Output('figure-cases', 'figure'),
    [Input('country-dropdown-multi', 'value'),
     Input('date-picker-single', 'date'),
     Input('variable-dropdown', 'value'),
     Input('log_y_on', 'value')])
def make_fig_cases(countries, date_value, variable, log_y):
    '''Give as input a threshold for the cumulative cases in the most updated
    timestep to filter out countries that do not have many cases.'''
    df = filter_data(countries=countries,
                       start_date=date_value)

    log_y_activated = False

    if log_y and (log_y[0] == 'log_y'):
      log_y_activated = True

    for v in variable_options:
        if v['value'] == variable:
            title = v['label']

    fig = timeseries_plot(df, time_variable="date",
                          variable=variable,
                          agg_variable="location",
                          log_y=log_y_activated,
                          title=title)

    return fig


if __name__ == '__main__':
    app.run_server()
