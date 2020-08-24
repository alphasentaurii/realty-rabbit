# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
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

external_stylesheets = ['style.css']
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets = [dbc.themes.CYBORG]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

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

preds = pd.read_csv('data/pred_vs_actual.csv', parse_dates=True)
preds = pd.DataFrame(preds)

forecast = pd.read_csv('data/forecast.csv', parse_dates=True)
forecast = pd.DataFrame(forecast)


# DICTIONARIES
available_zipcodes = df['RegionName'].unique()
# Create dicts
# NYC: dict of cities and zip codes
# nyc: dict of dataframes for each zip code
NYC, nyc, city_zip = spak.cityzip_dicts(df=df, col1='RegionName', col2='City')

txd = spak.time_dict(d=NYC, xcol='RegionName', ycol='MeanValue')

# Train Lines
NY_Newhaven=pd.read_csv('data/newhaven.csv')
NY_Harlem=pd.read_csv('data/harlem.csv')
NY_Hudson=pd.read_csv('data/hudson.csv')

# FIGURES

# Fig
fig=go.Figure()
for k,v in txd.items():
    fig.add_trace(go.Scatter(x=preds['Month'].loc[preds['RegionName']==k], y=preds['MeanValue'].loc[preds['RegionName']==k], name=f'{k} actual', line_color='lightgrey'))
    fig.add_trace(go.Scatter(x=preds['Month'].loc[preds['RegionName']==k], y=preds['predicted'].loc[preds['RegionName']==k], name=f'{k} pred', line_color='royalblue'))
    fig.add_trace(go.Scatter(x=forecast['Month'].loc[forecast['RegionName']==k], y=forecast['predicted'].loc[forecast['RegionName']==k], name=f'{k} forecast', line_color='lightseagreen'))

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

for k,v in txd.items():
    fig1.add_trace(go.Scatter(x=NY['Month'].loc[NY['RegionName']==k], y=NY['MeanValue'].loc[NY['RegionName']==k], name=str(k)))

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

#### FIG 2

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=NY_Newhaven['DateTime'], y=NY_Newhaven['MeanValue'], name="NewHaven MeanValue",
                         line_color='crimson'))
fig2.add_trace(go.Scatter(x=NY_Harlem['DateTime'], y=NY_Harlem['MeanValue'], name="Harlem MeanValue",
                         line_color='deepskyblue'))
fig2.add_trace(go.Scatter(x=NY_Hudson['DateTime'], y=NY_Hudson['MeanValue'], name="Hudson MeanValue",
                         line_color='lightgreen'))
fig2.update_layout(title_text='MeanValues by Train Line',
                  xaxis_rangeslider_visible=True)



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

    dcc.Graph(
        id='ts',
        figure=fig1
    ),
    
    dcc.Graph(
        id='ts_trainlines',
        figure=fig2
    )
 
])


if __name__ == '__main__':
    app.run_server(debug=True)
