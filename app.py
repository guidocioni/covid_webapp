import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# from flask_caching import Cache
from utils import *
from meta import tags
from tabs import get_aggregated_tab, get_testing_tab,\
    get_forecast_tab, get_maps_tab, get_table_tab,\
    get_aggregated_eu_tab

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                meta_tags=tags,
                suppress_callback_exceptions=True,
                url_base_pathname='/covid/')
server = app.server
app.title = 'COVID-19 live forecast'

# cache = Cache(server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': '/tmp'
# })
TMP_FOLDER='/tmp/'


# @cache.memoize(timeout=TIMEOUT)
def read_owid():
    return pd.read_pickle(TMP_FOLDER + 'df_owid.pickle')


# @cache.memoize(timeout=TIMEOUT)
def read_jrc():
    return pd.read_pickle(TMP_FOLDER + 'df_jrc.pickle')

# @cache.memoize(timeout=TIMEOUT)
def read_hospitalization():
    return pd.read_pickle(TMP_FOLDER + 'df_hospitalization.pickle')


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


def get_countries_list(df, country_variable):
    return list(df[country_variable].unique())


@app.callback(
    Output('intermediate-value', 'children'),
    [Input('country-dropdown-multi', 'value'), Input('date-picker-single', 'date')])
def filter_data_for_countries(country, date_value):
    return filter_data(countries=country,
                       start_date=date_value,
                       threshold=threshold_chosen).to_json(date_format='iso', orient='split')


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
        html.Div(html.H1('COVID-19 Monitoring')),
        html.Div('Data are taken from the European Center for Disease Monitoring (ECDC). Choose the relevant tab to show different plots.'),
        html.Div('Last Update: %s' % str(filter_data().date.max())),
        #
        dcc.Tabs(parent_className='custom-tabs',
                 className='custom-tabs-container',
                 children=[
                    dcc.Tab(label='Daily data overview',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_table_tab(make_table_data(), make_table_data_eu())),
					# -----------------------------------------------------  #
                     dcc.Tab(label='Plots (global)',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_aggregated_tab(dropdown_options)),
                     # -----------------------------------------------------  #
                     dcc.Tab(label='Plots (EU regions)',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_aggregated_eu_tab(region_eu)),
                     # -----------------------------------------------------  #
                     dcc.Tab(label='Testing & Hospitalization',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_testing_tab(dropdown_options, dropdown_options_2)),
                     # -----------------------------------------------------  #
                     dcc.Tab(label='Forecast',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_forecast_tab(dropdown_options)),
                     # -----------------------------------------------------  #
                     dcc.Tab(label='Maps',
                             className='custom-tab',
                             selected_className='custom-tab--selected',
                             children=get_maps_tab()),
                     # -----------------------------------------------------  #
                 ]),
        html.Div(html.A('Created by Guido Cioni', href='www.guidocioni.it'))
    ],
        style={'width': '100%', 'display': 'inline-block'})


app.layout = serve_layout


def make_table_data():
    df = filter_data(start_date='2020-03-15',
                     threshold=1000)
    df = df.loc[df.date == df.date.max()]\
        .round(3).sort_values(by="total_cases_change",
                              ascending=False)

    data = df.to_dict('records')

    return {'columns': table_columns,
            'data': data,
            'df': df}


def make_table_data_eu():
    df = read_jrc()
    df = df.loc[df.Date == df.Date.max()]\
        .round(3).sort_values(by="total_cases_change",
                              ascending=False)

    data = df.to_dict('records')

    return {'columns': table_columns_eu,
            'data': data,
            'df': df}


@app.callback(
    Output('figure-map-eu', 'figure'),
    [Input('variable-dropdown-map-eu', 'value')])
def make_fig_map_weekly_europe(variable):
    df = read_jrc()

    return make_fig_map_weekly(df, variable)


@app.callback(
    Output('figure-eu', 'figure'),
    [Input('region-dropdown-eu', 'value'), Input('variable-dropdown-eu', 'value')])
def make_fig_eu(regions, variable):
    '''Give as input a threshold for the cumulative cases in the most updated
    timestep to filter out countries that do not have many cases.'''
    df = read_jrc()
    df = df.loc[df.location.isin(regions)]

    fig = timeseries_plot(df,
                          time_variable="Date",
                          variable=variable,
                          agg_variable="location",
                          log_y=False,
                          title="")

    return fig

@app.callback(
    Output('figure-r0-eu', 'figure'),
    [Input('region-dropdown-eu', 'value')])
def make_fig_r0_eu(regions):
    df = read_jrc()
    df = df.loc[df.location.isin(regions)]

    fig = timeseries_plot(df,
                          time_variable="Date",
                          variable="r0",
                          agg_variable="location",
                          log_y=False,
                          title='Reproductivity ratio r0 (estimated using RKI method)')

    fig.update_yaxes(range=[0, 5])

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
    Output('figure-testing', 'figure'),
    [Input('variable-dropdown-2', 'value'),
     Input('country-dropdown-testing', 'value')])
def make_fig_testing(variable, country):
    df = filter_data(countries=country, start_date='2020-03-01')
    df = df[df[variable].notna()]

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

    fig = timeseries_plot(df, time_variable="date",
                          variable="total_cases", agg_variable="location",
                          log_y=True,
                          title='Confirmed cases evolution (log. scale, cumulative sum)')

    return fig


@app.callback(
    Output('figure-cumulative-2', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_cumulative_2(df):
    '''Give as input a threshold for the cumulative cases in the most updated
    timestep to filter out countries that do not have many cases.'''
    df = pd.read_json(df, orient='split')

    fig = timeseries_plot(df, time_variable="date",
                          variable="total_cases_per_million",
                          agg_variable="location",
                          log_y=False,
                          title='Density of cases (cumulative sum) per 1M inhabitants')

    return fig


@app.callback(
    Output('figure-cumulative-3', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_cumulative_3(df):
    '''Give as input a threshold for the cumulative cases in the most updated
    timestep to filter out countries that do not have many cases.'''
    df = pd.read_json(df, orient='split')

    fig = timeseries_plot(df, time_variable="date",
                          variable="total_deaths",
                          agg_variable="location",
                          log_y=True,
                          title='Confirmed deaths evolution (log. scale, cumulative sum)')

    return fig


@app.callback(
    Output('figure-cases', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_cases(df):
    '''Give as input a threshold for the cumulative cases in the most updated
    timestep to filter out countries that do not have many cases.'''
    df = pd.read_json(df, orient='split')

    fig = timeseries_plot(df, time_variable="date",
                          variable="new_cases_smoothed",
                          agg_variable="location",
                          log_y=False,
                          title='7-day smoothed Daily cases evolution')

    return fig


@app.callback(
    Output('figure-increment', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_increment(df):
    df = pd.read_json(df, orient='split')

    fig = timeseries_plot(df, time_variable="date",
                          variable="total_cases_change",
                          agg_variable="location",
                          log_y=False,
                          title='7-day smoothed daily increase in confirmed cases')

    fig.update_yaxes(range=[0, 20])

    return fig


@app.callback(
    Output('figure-r0', 'figure'),
    [Input('intermediate-value', 'children')])
def make_fig_r0(df):
    df = pd.read_json(df, orient='split')

    fig = timeseries_plot(df, time_variable="date",
                          variable="r0",
                          agg_variable="location",
                          log_y=False,
                          title='Reproductivity ratio r0 (estimated using RKI method)')

    return fig

if __name__ == '__main__':
    app.run_server()
