# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
# PLOTLY / CUFFLINKS for iplots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cufflinks as cf
cf.go_offline()
import json
import statspack as spak

#external_stylesheets = ['style.css']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets = [dbc.themes.CYBORG]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#ffffff',#'#111111', 
    'text': '#000000' #'#7FDBFF'
}
#colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]

# LOAD DATA
data = pd.read_csv('data/newyork_1996-2018.csv', parse_dates=True)
NY = pd.DataFrame(data)
df = spak.makeTime(NY, idx='DateTime')

####### Forecast Prediction Values
df_preds = pd.read_csv('data/ny_predictions.csv', parse_dates=True)
df_preds = df_preds.drop('RegionID', axis=1)
df_preds = spak.makeTime(df_preds, idx='DateTime')
# df_preds = df_preds.loc[df_preds.index >= '2016-02-01'] 


preds = pd.read_csv('data/pred_vs_actual.csv', parse_dates=True)
preds = pd.DataFrame(preds)

forecast = pd.read_csv('data/forecast.csv', parse_dates=True)
forecast = pd.DataFrame(forecast)


# Train Lines
NY_Newhaven=pd.read_csv('data/newhaven.csv')
NY_Harlem=pd.read_csv('data/harlem.csv')
NY_Hudson=pd.read_csv('data/hudson.csv')


# DICTIONARIES
available_zipcodes = df['RegionName'].unique()
# Create dicts
# NYC: dict of cities and zip codes
# nyc: dict of dataframes for each zip code
NYC, nyc, city_zip = spak.cityzip_dicts(df=df, col1='RegionName', col2='City')


txd = spak.time_dict(d=NYC, xcol='RegionName', ycol='MeanValue')


# FIGURES
fig=go.Figure()
for k,v in txd.items():
    fig.add_trace(go.Line(x=preds['Month'].loc[preds['RegionName']==k], y=preds['MeanValue'].loc[preds['RegionName']==k], name=f'{k} actual', line_color='lightgrey'))
    fig.add_trace(go.Line(x=preds['Month'].loc[preds['RegionName']==k], y=preds['predicted'].loc[preds['RegionName']==k], name=f'{k} pred', line_color='royalblue'))
    fig.add_trace(go.Line(x=forecast['Month'].loc[forecast['RegionName']==k], y=forecast['predicted'].loc[forecast['RegionName']==k], name=f'{k} forecast', line_color='lightseagreen'))

#fig.add_trace(go.Line(x=NY['Month'], y=NY['MeanValue'], name='Actual', line_color='lightgrey'))
#fig = px.scatter(preds, x=preds.index, y='predicted')
# fig = preds.iplot(kind='bar', x=index, y='predicted', title='Time Series with Range Slider and Selectors', asFigure=True)
fig.update_layout(title_text='Predictions and Forecast', xaxis_rangeslider_visible=True)
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# fig1

fig1 = go.Figure()
# fig1.add_trace(go.Scatter(x=df.index, y=df['MeanValue'], name="Mean Home Value",line_color='crimson'))
# fig1.add_trace(go.Scatter(x=FC.index, y=FC['pred_mean'], name="Forecast Value",line_color='deepskyblue'))
# fig1.add_trace(go.Scatter(x=NY_Hudson['DateTime'], y=NY_Hudson['MeanValue'], name="Hudson MeanValue",
#                          line_color='lightgreen'))
# fig1.update_layout(title_text='MeanValues by Train Line',
#                   xaxis_rangeslider_visible=True)

for k,v in txd.items():
    fig1.add_trace(go.Line(x=NY['Month'].loc[NY['RegionName']==k], y=NY['MeanValue'].loc[NY['RegionName']==k], name=str(k)))

fig1.update_layout(title_text='Westchester County NY - Mean Home Values',
                  xaxis_rangeslider_visible=True)


fig1.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# fig2 = preds.iplot(kind='bar', x=index, y='predicted', title='Time Series with Range Slider and Selectors', asFigure=True)


#### FIG 3

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=NY_Newhaven['DateTime'], y=NY_Newhaven['MeanValue'], name="NewHaven MeanValue",
                         line_color='crimson'))
fig3.add_trace(go.Scatter(x=NY_Harlem['DateTime'], y=NY_Harlem['MeanValue'], name="Harlem MeanValue",
                         line_color='deepskyblue'))
fig3.add_trace(go.Scatter(x=NY_Hudson['DateTime'], y=NY_Hudson['MeanValue'], name="Hudson MeanValue",
                         line_color='lightgreen'))
fig3.update_layout(title_text='MeanValues by Train Line',
                  xaxis_rangeslider_visible=True)


top5 = df.loc[(df['RegionName'] == 10708) | (df['RegionName']==10706) | (df['RegionName']==10803) | (df['RegionName']==10514) | (df['RegionName']==10605) ]
top5_fc = forecast.loc[(forecast['RegionName'] ==10708) | (forecast['RegionName']==10706) | (forecast['RegionName'] ==10514) | (forecast['RegionName']==10605)]

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=top5['Month'], y=top5['MeanValue']))
# px.scatter(top5, x='Month', y='MeanValue')
fig4.add_trace(go.Scatter(x=top5_fc['Month'], y=top5_fc['predicted']))
fig4.update_layout(title_text='Top 5 Zip Code Forecasts',
                  xaxis_rangeslider_visible=True)


# DATA TABLES

def generate_table(dataframe, max_rows=5):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ]), 
    ])


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='RealtyRabbit',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.H4(
        children='A real estate forecasting application for home buyers.', 
        style={
        'textAlign': 'center',
        'color': colors['text']
        }
    ),

    dcc.Graph(
        id='forecast-graph',
        figure=fig
    ),

    # dcc.Store(
    # id='forecast-figure-store',
    # data=[{
    #      'x': preds[preds['RegionName'] == '10701']['Month']
    #      'y': preds[preds['RegionName'] == '10701']['predicted']
    # }]
    # ),
    # 'Indicator',
    # dcc.Dropdown(
    #     id='forecast-graph-indicator',
    #     options=[
    #         {'label': 'Predicted', 'value': 'predicted'},
    #         {'label': 'Mean Value', 'value': df_preds['MeanValue']}
   
    #         #{'label': 'Forecast', 'value': ''}
    #     ], 
    #     value='predicted'
    # ),
    # 'RegionName',
    # dcc.Dropdown(
    #      id='forecast-graph-zipcode',
    #      options=[
    #          {'label': RegionName, 'value': RegionName}
    #          for RegionName in available_zipcodes
    #      ],
    #      value='10701'
    # ),
    # 'Graph scale',
    # dcc.RadioItems(
    #     id='forecast-graph-scale',
    #     options=[
    #         {'label': x, 'value': x} for x in ['linear', 'log']
    #     ],
    #     value='linear'
    # ),
    # html.Hr(),
    # html.Details([
    #     html.Summary('Contents of figure storage'),
    #     dcc.Markdown(
    #         id='forecast-figure-json'
    #     )
    # ]),

    dcc.Graph(
        id='ts',
        figure=fig1
    ),

    dcc.Graph(
        id='ts_trainlines',
        figure=fig3
    ),

    dcc.Graph(
        id='top5-zipcodes',
        figure=fig4
    ),

    generate_table(df_preds),

    dcc.Graph(
        id='clientside-graph'
    ),
    dcc.Store(
        id='clientside-figure-store',
        data=[{
            'x': preds[preds['RegionName'] == '10701'].index,
            'y': preds[preds['RegionName'] == '10701']['MeanValue']
        }]
    ),
    'Indicator',
    dcc.Dropdown(
        id='clientside-graph-indicator',
        options=[
            {'label': 'Mean Value', 'value': 'MeanValue'},
            {'label': 'Predicted', 'value': 'predicted'},
            # {'label': 'Rolling Average', 'value': 'RollingAvg'}
        ], 
        value='MeanValue'
    ),
    'RegionName',
    dcc.Dropdown(
        id='clientside-graph-zipcode',
        options=[
            {'label': RegionName, 'value': RegionName}
            for RegionName in available_zipcodes
        ],
        value='10701'
    ),
    'Graph scale',
    dcc.RadioItems(
        id='clientside-graph-scale',
        options=[
            {'label': x, 'value': x} for x in ['linear', 'log']
        ],
        value='linear'
    ),
    html.Hr(),
    html.Details([
        html.Summary('Contents of figure storage'),
        dcc.Markdown(
            id='clientside-figure-json'
        )
    ])
  # dcc.Input(
    #     id='number-in',
    #     value=10701,
    #     style={'fontSize':28}
    # ),
    # html.Button(
    #     id='submit-button',
    #     n_clicks=0,
    #     children='Submit',
    #     style={'fontSize':28}
    # ),
    # html.H1(id='number-out'),
])

# @app.callback(
#     Output('number-out', 'children'),
#     [Input('submit-button', 'n_clicks')],
#     [State('number-in', 'value')])
# def output(n_clicks, number):
#     return '{} displayed after {} clicks'.format(number,n_clicks)


# @app.callback(
#     Output('forecast-figure-store', 'data'),
#     [Input('forecast-graph-indicator', 'value'),
#      Input('forecast-graph-zipcode', 'value')]
# )
# def update_store_data(indicator, zipcode):
#     dff = preds[preds['RegionName'] == zipcode]

#     return [{
#         'x': dff['Month'],
#         'y': dff[indicator],
#         'mode': 'markers'
#     }]


# app.clientside_callback(
#     """
#     function(data, scale) {
#         return {
#             'data': data,
#             'layout': {
#                  'yaxis': {'type': scale}
#              }
#         }
#     }
#     """,
#     Output('forecast-graph', 'figure'),
#     [Input('forecast-figure-store', 'data'),
#      Input('forecast-graph-scale', 'value')]
# )


# @app.callback(
#     Output('forecast-figure-json', 'children'),
#     [Input('forecast-figure-store', 'data')]
# )
# def generated_figure_json(data):
#     return '```\n'+json.dumps(data, indent=2)+'\n```'


@app.callback(
    Output('clientside-figure-store', 'data'),
    [Input('clientside-graph-indicator', 'value'),
     Input('clientside-graph-zipcode', 'value')]
)
def update_store_data(indicator, zipcode):
    dff = preds[preds['RegionName'] == zipcode]
    return [{
        'x': dff.index,
        'y': dff[indicator],
        'mode': 'markers'
    }]


app.clientside_callback(
    """
    function(data, scale) {
        return {
            'data': data,
            'layout': {
                 'yaxis': {'type': scale}
             }
        }
    }
    """,
    Output('clientside-graph', 'figure'),
    [Input('clientside-figure-store', 'data'),
     Input('clientside-graph-scale', 'value')]
)


@app.callback(
    Output('clientside-figure-json', 'children'),
    [Input('clientside-figure-store', 'data')]
)
def generated_figure_json(data):
    return '```\n'+json.dumps(data, indent=2)+'\n```'



if __name__ == '__main__':
    app.run_server(debug=True,host='127.0.0.1')
