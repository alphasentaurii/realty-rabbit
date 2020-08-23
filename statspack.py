#         _ __ _   _
#  /\_/\ | '__| | | |
#  [===] | |  | |_| |
#   \./  |_|   \__,_| 
#
# /***************//***************//***************/
# /* statspack.py *//* Ru Kein *//* www.hakkeray.com */ 
# /***************//***************//***************/
#  ________________________
# | hakkeray |  Updated:  |
# | v3.0.0   |  8.12.2020 |
# ------------------------
#
# * note: USZIPCODE pypi library is required to run zip_stats()
# Using pip in the notebook:
# !pip install -U uszipcode

# fsds tool required
# !pip install -U fsds_100719


# STANDARD libraries
import pandas as pd
from pandas import Series
import numpy as np
from numpy import log 

# PLOTTING
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython.display as dp
plt.style.use('seaborn-bright')
mpl.style.use('seaborn-bright')
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 24}
mpl.rc('font', **font)

import seaborn as sns
sns.set_style('whitegrid')
#ignore pink warnings
import warnings
warnings.filterwarnings('ignore')
# Allow for large # columns
pd.set_option('display.max_columns', 0)
# pd.set_option('display.max_rows','')


# STATSMODELS
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
#import statsmodels.formula.api as ols
import statsmodels.stats.multicomp
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# SCIPY
import scipy.stats as stats
from scipy.stats import normaltest as normtest # D'Agostino and Pearson's omnibus test
from collections import Counter

# SKLEARN
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import RobustScaler
# ADDITIONAL LIBRARIES
#import researchpy as rp
import uszipcode
from uszipcode import SearchEngine

# PLOTLY / CUFFLINKS for iplots
import plotly
import plotly.offline as pyo
import plotly.graph_objects as go
import cufflinks as cf
cf.go_offline()

def timeMap(d, xcol='RegionName', ycol='MeanValue', zipcodes=None):
    """
    'Maps' a timeseries plot of zipcodes 
    
    # fig,ax = mapTime(d=HUDSON, xcol='RegionName', ycol='MeanValue', MEAN=True, vlines=None)
    
    **ARGS
    d: takes a dictionary of dataframes
    xcol: column in dataframe containing x-axis values (ex: zipcode)
    ycol: column in dataframe containing y-axis values (ex: price)
    X: list of x values to plot on x-axis (defaults to all x in d if empty)
    
    
    *Ex2: `d` = dictionary of dataframes
    mapTime(d=NYC, xcol='RegionName', y='MeanValue')
    """ 
    # zipcodes to plot
    if zipcodes is None:
        zipcodes = list(d.keys())

    # create empty dictionary for plotting 
    txd = {}
    colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]
    # create different linestyles for zipcodes (easier to distinguish if list is long)
    # create figure for timeseries plot
    for i,zc in enumerate(zipcodes):
        # store each zipcode as ts  
        ts = d[zc][ycol].rename(zc)
        ### PLOT each zipcode as timeseries `ts`
        #fig = ts.iplot(kind='line', title="Mean Home Values", asFigure=False)
        txd[zc] = ts

    return txd

# # HOT_STATS() function: display statistical summaries of a feature column
# def hot_stats(data, column, verbose=False, t=None):
#     """
#     Scans the values of a column within a dataframe and displays its datatype, 
#     nulls (incl. pct of total), unique values, non-null value counts, and 
#     statistical info (if the datatype is numeric).
    
#     ---------------------------------------------
    
#     Parameters:
    
#     **args:
    
#         data: accepts dataframe
    
#         column: accepts name of column within dataframe (should be inside quotes '')
    
#     **kwargs:
    
#         verbose: (optional) accepts a boolean (default=False); verbose=True will display all 
#         unique values found.   
    
#         t: (optional) accepts column name as target to calculate correlation coefficient against 
#         using pandas data.corr() function. 
    
#     -------------
    
#     Examples: 
    
#     hot_stats(df, 'str_column') --> where df = data, 'string_column' = column you want to scan
    
#     hot_stats(df, 'numeric_column', t='target') --> where 'target' = column to check correlation value
    
#     -----------------
#     Future:
#     #todo: get mode(s)
#     #todo: functionality for string objects
#     #todo: pass multiple columns at once and display all
#     -----------------
    
#     """
#     # assigns variables to call later as shortcuts 
#     feature = data[column]
#     rdash = "-------->"
#     ldash = "<--------"
    
#     # figure out which hot_stats to display based on dtype 
#     if feature.dtype == 'float':
#         hot_stats = feature.describe().round(2)
#     elif feature.dtype == 'int':
#         hot_stats = feature.describe()
#     elif feature.dtype == 'object' or 'category' or 'datetime64[ns]':
#         hot_stats = feature.agg(['min','median','max'])
#         t = None # ignores corr check for non-numeric dtypes by resetting t
#     else:
#         hot_stats = None

#     # display statistics (returns different info depending on datatype)
#     print(rdash)
#     print("HOT!STATS")
#     print(ldash)
    
#     # display column name formatted with underline
#     print(f"\n{feature.name.upper()}")
    
#     # display the data type
#     print(f"Data Type: {feature.dtype}\n")
    
#     # display the mode
#     print(hot_stats,"\n")
#     print(f"à-la-Mode: \n{feature.mode()}\n")
    
#     # find nulls and display total count and percentage
#     if feature.isna().sum() > 0:  
#         print(f"Found\n{feature.isna().sum()} Nulls out of {len(feature)}({round(feature.isna().sum()/len(feature)*100,2)}%)\n")
#     else:
#         print("\nNo Nulls Found!\n")
    
#     # display value counts (non-nulls)
#     print(f"Non-Null Value Counts:\n{feature.value_counts()}\n")
    
#     # display count of unique values
#     print(f"# Unique Values: {len(feature.unique())}\n")
#     # displays all unique values found if verbose set to true
#     if verbose == True:
#         print(f"Unique Values:\n {feature.unique()}\n")
        
#     # display correlation coefficient with target for numeric columns:
#     if t != None:
#         corr = feature.corr(data[t]).round(4)
#         print(f"Correlation with {t.upper()}: {corr}")


# # NULL_HUNTER() function: display Null counts per column/feature
# def null_hunter(data):
#     print(f"Columns with Null Values")
#     print("------------------------")
#     for column in data:
#         if data[column].isna().sum() > 0:
#             print(f"{data[column].name}: \n{data[column].isna().sum()} out of {len(data[column])} ({round(data[column].isna().sum()/len(data[column])*100,2)}%)\n")


# # CORRCOEF_DICT() function: calculates correlation coefficients assoc. with features and stores in a dictionary
# def corr_dict(data, X, y):
#     corr_coefs = []
#     for x in X:
#         corr = data[x].corr(data[y])
#         corr_coefs.append(corr)
    
#     corr_dict = {}
    
#     for x, c in zip(X, corr_coefs):
#         corr_dict[x] = c
#     return corr_dict

# # SUB_SCATTER() function: pass list of features (x_cols) and compare against target (or another feature)
# def sub_scatter(data, x_cols, y, color=None, nrows=None, ncols=None):
#     """
#     Desc: displays set of scatterplots for multiple columns or features of a dataframe.
#     pass in list of column names (x_cols) to plot against y-target (or another feature for 
#     multicollinearity analysis)
    
#     args: data, x_cols, y
    
#     kwargs: color (default is magenta (#C839C5))
    
#     example:
    
#     x_cols = ['col1', 'col2', 'col3']
#     y = 'col4'
    
#     sub_scatter(df, x_cols, y)
    
#     example with color kwarg:
#     sub_scatter(df, x_cols, y, color=#)
    
#     alternatively you can pass the column list and target directly:
#     sub_scatter(df, ['col1', 'col2', 'col3'], 'price')

#     """   
#     if nrows == None:
#         nrows = 1
#     if ncols == None:
#         ncols = 3
#     if color == None:
#         color = '#C839C5'

#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16,4))
#     for x_col, ax in zip(x_cols, axes):
#         data.plot(kind='scatter', x=x_col, y=y, ax=ax, color=color)
#         ax.set_title(x_col.capitalize() + " vs. " + y.capitalize())


# # SUB_HISTS() function: plot histogram subplots
# def sub_hists(data):
#     plt.style.use('seaborn-bright')
#     for column in data.describe():
#         fig = plt.figure(figsize=(12, 5))
#         ax = fig.add_subplot(121)
#         ax.hist(data[column], density=True, label = column+' histogram', bins=20)
#         ax.set_title(column.capitalize())
#         ax.legend()
#         fig.tight_layout()

# --------- ZIP_STATS() --------- #

def zip_stats(zipcodes, 
              minimum=0, maximum=5000000,
              simple=True):
    """
    Lookup median home values for zipcodes or return zip codes of a min and max median home value
    #TODO: add input options for city state county
    #TODO: add input options for other keywords besides median home val
    
    *Prerequisites: USZIPCODE() pypi package is a required dependency
    
    **ARGS
    zipcodes: dataframe or array of strings (zipcodes) 
    > Example1: zipcodes=df[zipcode']
    > Example2: zipcodes=['01267','90025']
    
    minimum: integer for dollar amount min threshold (default is 0)
    maximum: integer for dollar amount max threshold (default is 5000000, i.e. no maximum)
    
    **KWARGS
    simple: default=True
    > set simple_zipcode=False to use rich info database (will only apply once TODOs above are added)
    """
    # pypi package for retrieving information based on us zipcodes
    import uszipcode
    from uszipcode import SearchEngine
    
    # set simple_zipcode=False to use rich info database
    if simple:
        search = SearchEngine(simple_zipcode=True)
    else:
        search = SearchEngine(simple_zipcode=False)
        
    # create empty dictionary 
    dzip = {}

    # search pypi uszipcode library to retrieve data for each zipcode
    for code in zipcodes:
        z = search.by_zipcode(code)
        dzip[code] = z.to_dict()
    
    keyword='median_home_value'
    # # pull just the median home values from dataset and append to list
    # create empty lists for keys and vals
    keys = []
    zips = []
    
    for index in dzip:
        keys.append(dzip[index][keyword])

    # put zipcodes in other list
    for index in dzip:
        zips.append(dzip[index]['zipcode'])
        
    # zip both lists into dictionary
    zipkey = dict(zip(zips, keys))

    zipvals = {}
    
    for k,v in zipkey.items():
        if v > minimum and v < maximum:
            zipvals[k]=v
    return zipvals


"""

>>>>>>>>>>>>>>>>>> TIME SERIES <<<<<<<<<<<<<<<<<<<<<<

* makeTime()
* checkTime()
* mapTime()


"""

def makeTime(data, idx):
    """
    Converts a column (`idx`) to datetime formatted index for a dataframe (`data`)
    Returns copy of original dataframe

    new_df = makeTime(df_original, 'DateTime')  
    """
    df = data.copy()
    df[idx] = pd.to_datetime(df[idx], errors='coerce')
    df['DateTime'] = df[idx].copy()
    df.set_index(idx, inplace=True, drop=True)
    return df

def melt_data(df): # from flatiron starter notebook
    melted = pd.melt(df, id_vars=['RegionID','RegionName', 'City', 'State', 'Metro', 'CountyName', 
                                  'SizeRank'], var_name='Month', value_name='MeanValue')
    melted['Month'] = pd.to_datetime(melted['Month'], format='%Y-%m')
    melted = melted.dropna(subset=['MeanValue'])
    return melted    

def cityzip_dicts(df, col1, col2):
    """
    Creates 3 dictionaries:
    # dc1 : Dictionary of cities and zipcodes for quick referencing
    # dc2: Dictionary of dataframes for each zipcode.
    # city_zip: dictionary of zipcodes for each city

    dc1 key: zipcodes
    dc2 key: cities
    city_zip key: city name
    
    Returns dc1, dc2, city_zip
    
    Ex:
    NYC, nyc, city_zip = cityzip_dictionaries(df=NY, col1='RegionName', col2='City')
    
    # dc1: returns dataframe for a given zipcode, or dict values of given column
    NYC[10549] --> dataframe
    NYC[10549]['MeanValue'] --> dict 
    
    # dc2: return dataframe for a given city, or just zipcodes for a given city:
    nyc['New Rochelle'] --> dataframe
    nyc['New Rochelle']['RegionName'].unique() --> dict of zip codes
    
    # city_zip: returns dict of all zip codes in a city
    city_zip['Yonkers']
    
    """
    dc1 = {}
    dc2 = {}
    for zipcode in df[col1].unique():
        dc1[zipcode] = df.groupby(col1).get_group(zipcode).resample('MS').asfreq()  
        for city in df[col2].unique():
            dc2[city] = df.groupby(col2).get_group(city)
            
    # create reference dict of city and zipcode matches
    #zipcodes, cities in westchester
    zips = df.RegionName.unique() #cities
    cities = df.City.unique()
    print("# ZIP CODES: ", len(zips))
    print("# CITIES: ", len(cities))
    city_zip = {}
    for city in cities:
        c = str(f'{city}')
        city = df.loc[df['City'] == city]
        zc = list(city['RegionName'].unique())
        city_zip[c] = zc
        
    return dc1, dc2, city_zip

def time_dict(d, xcol='RegionName', ycol='MeanValue'):
    # zipcodes to plot
    zipcodes = list(d.keys())

    # create empty dictionary for plotting 
    txd = {}
    for i,zc in enumerate(zipcodes):
        # store each zipcode as ts  
        ts = d[zc][ycol].rename(zc)
        txd[zc] = ts

    return txd

def mapTime(d, xcol, ycol='MeanValue', X=None, vlines=None, MEAN=True):
    """
    'Maps' a timeseries plot of zipcodes 
    
    # fig,ax = mapTime(d=HUDSON, xcol='RegionName', ycol='MeanValue', MEAN=True, vlines=None)
    
    **ARGS
    d: takes a dictionary of dataframes OR a single dataframe
    xcol: column in dataframe containing x-axis values (ex: zipcode)
    ycol: column in dataframe containing y-axis values (ex: price)
    X: list of x values to plot on x-axis (defaults to all x in d if empty)
    
    **kw_args
    mean: plots the mean of X (default=True)
    vlines : default is None: shows MIN_, MAX_, crash 
    
    *Ex1: `d` = dataframe
    mapTime(d=NY, xcol='RegionName', ycol='MeanValue', X=list_of_zips)
    
    *Ex2: `d` = dictionary of dataframes
    mapTime(d=NYC, xcol='RegionName', y='MeanValue')
    """    
    # create figure for timeseries plot
    fig = plt.subplots(figsize=(21,13))
    plt.title(label=f'Time Series Plot: {str(ycol)}')
    ax = fig.gca()
    ax.set(title='Mean Home Values', xlabel='Year', ylabel='Price($)')  
    
    zipcodes = []
    #check if `d` is dataframe or dictionary
    if type(d) == pd.core.frame.DataFrame:
        # if X is empty, create list of all zipcodes
        if len(X) == 0:
            zipcodes = list(d[xcol].unique())
        else:
            zipcodes = X
        # cut list in half  
        breakpoint = len(zipcodes)//2
        
        for zc in zipcodes:
            if zc < breakpoint:
                ls='-'
            else:
                ls='--'
            ts = d[zc][ycol].rename(zc)#.loc[zc]
            ts = d[ycol].loc[zc]
            ### PLOT each zipcode as timeseries `ts`
            ts.plot(label=str(zc), ax=ax, ls=ls)
        ## Calculate and plot the MEAN
        
        if MEAN:
            mean = d[ycol].mean(axis=1)
            mean.plot(label='Mean',lw=5,color='black')
    # if type(d) == dict():
    elif type(d) == dict:
        # if X passed in as empty list, create list of all zipcodes
        if len(X) == 0:
            zipcodes = list(d.keys())
        else:
            zipcodes = X
        # cut list in half  
        breakpoint = len(zipcodes)//2
        
        # create empty dictionary for plotting 
        txd = {}
        # create different linestyles for zipcodes (easier to distinguish if list is long)
        for i,zc in enumerate(zipcodes):
            if i < breakpoint:
                ls='-'
            else:
                ls='--'
            # store each zipcode as ts  
            ts = d[zc][ycol].rename(zc)
            ### PLOT each zipcode as timeseries `ts`
            ts.plot(label=str(zc), ax=ax, ls=ls, lw=2);
            txd[zc] = ts
            
        if MEAN:
            mean = pd.DataFrame(txd).mean(axis=1)
            mean.plot(label='Mean',lw=5,color='black')
            
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", ncol=2)
            
    if vlines:
        ## plot crash, min and max vlines
        crash = '01-2009'
        ax.axvline(crash, label='Housing Index Drops',color='red',ls=':',lw=2)
        MIN_ = ts.loc[crash:].idxmin()
        MAX_ = ts.loc['2004':'2010'].idxmax()
        ax.axvline(MIN_, label=f'Min Price Post Crash {MIN_}', color='black',lw=2)    
        ax.axvline(MAX_,label='Max Price', color='black', ls=':',lw=2) 

    return fig, ax


# # Check Seasonality 
def freeze_time(ts, mode='A'):
    """
    Calculates and plots Seasonal Decomposition for a time series
    ts : time-series
    mode : 'A' for 'additive' or 'M' for 'multiplicative'
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    if mode == 'A': #default
        decomp = seasonal_decompose(ts, model='additive')
    elif mode == 'M':
        decomp = seasonal_decompose(ts, model='multiplicative')
    
    freeze = decomp.plot()
    ts_seas = decomp.seasonal

    plt.figure()
    plt.tight_layout()

    ax = ts_seas.plot(c='green')
    fig = ax.get_figure()
    fig.set_size_inches(12,5)

    ## Get min and max idx
    min_ = ts_seas.idxmin()
    max_ = ts_seas.idxmax()
    min_2 = ts_seas.loc[max_:].idxmin()

    ax.axvline(min_,label=min_,c='red')
    ax.axvline(max_,c='red',ls=':', lw=2)
    ax.axvline(min_2,c='red', lw=2)

    period = min_2 - min_
    ax.set_title(f'Season Length = {period}')

    return freeze

#### clockTime() --- time-series snapshot statistical summary ###
#
#  /\    /\    /\    /\    
# / CLOCKTIME STATS /
#     \/    \/    \/
#  

# """
# clockTime()

# Dependencies include the following METHODS:
# - check_time(data, time) >>> convert to datetimeindex
# - test_time(TS, y) >>> dickey-fuller (stationarity) test
# - roll_time() >>> rolling mean
# - freeze_time() >>> seasonality check
# - diff_time() >>> differencing 
# - autoplot() >>> autocorrelation and partial autocorrelation plots

# """
# class clockTime():
#     def __init__(data, time, x1, x2, y, freq=None):

#         self.data = data
#         self.time = time
#         self.x1 = x1
#         self.x2 = x2
#         self.y = y
#         self.freq = freq

def clockTime(ts, lags, d, TS, y):
    """    
     /\    /\    /\    /\  ______________/\/\/\__-_-_
    / CLOCKTIME STATS /  \/
        \/    \/    \/    

    # clockTime(ts, lags=43, d=5, TS=NY, y='MeanValue',figsize=(13,11))
    #
    # ts = df.loc[df['RegionName']== zc]["MeanValue"].rename(zc).resample('MS').asfreq()
    """
    # import required libraries
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import log 
    import pandas as pd
    from pandas import Series
    from pandas.plotting import autocorrelation_plot
    from pandas.plotting import lag_plot
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf  
    
    print(' /\\   '*3+' /')
    print('/ CLOCKTIME STATS')
    print('    \/'*3)

    #**************#   
    # Plot Time Series
    #original
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(21,13))
    ts.plot(label='Original', ax=axes[0,0],c='red')
    # autocorrelation 
    autocorrelation_plot(ts, ax=axes[0,1], c='magenta') 
    # 1-lag
    autocorrelation_plot(ts.diff().dropna(), ax=axes[1,0], c='green')
    lag_plot(ts, lag=1, ax=axes[1,1])
    
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    # DICKEY-FULLER Stationarity Test
    # TS = NY | y = 'MeanValue'
    dtest = adfuller(TS[y].dropna())
    if dtest[1] < 0.05:
        ## difference data before checking autoplot
        stationary = False
        r = 'rejected'
    else:
        ### skip differencing and check autoplot
        stationary = True 
        r = 'accepted'

    #**************#
    # ts orders of difference
    ts1 = ts.diff().dropna()
    ts2 = ts.diff().diff().dropna()
    ts3 = ts.diff().diff().diff().dropna()
    ts4 = ts.diff().diff().diff().diff().dropna()
    tdiff = [ts1,ts2,ts3,ts4]
    # Calculate Standard Deviation of Differenced Data
    sd = []
    for td in tdiff:
        sd.append(np.std(td))
    
    #sd = [np.std(ts1), np.std(ts2),np.std(ts3),np.std(ts4)]
    SD = pd.DataFrame(data=sd,index=['ts1',' ts2', 'ts3', 'ts4'], columns={'sd'})
    #SD['sd'] = [np.std(ts1), np.std(ts2),np.std(ts3),np.std(ts4)]
    SD['D'] = ['d=1','d=2','d=3','d=4']
    MIN = SD.loc[SD['sd'] == np.min(sd)]['sd']

    # Extract and display full test results 
    output = dict(zip(['ADF Stat','p-val','# Lags','# Obs'], dtest[:4]))
    for key, value in dtest[4].items():
        output['Crit. Val (%s)'%key] = value
    output['min std dev'] = MIN
    output['NULL HYPOTHESIS'] = r
    output['STATIONARY'] = stationary
     
    # Finding optimal value for order of differencing
    # from pmdarima.arima.utils import ndiffs
    # adf = ndiffs(x=ts, test='adf')
    # kpss = ndiffs(x=ts, test='kpss')
    # pp = ndiffs(x=ts, test='pp')
        
    # output['adf,kpss,pp'] = [adf,kpss,pp]

    #**************#
    # show differencing up to `d` on single plot (default = 5)
    fig2 = plt.figure(figsize=(13,5))
    ax = fig2.gca()
    for i in range(d):
        ax = ts.diff(i).plot(label=i)
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left", ncol=2)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # DIFFERENCED SERIES
    fig3 = plt.figure(figsize=(13,5))
    ts1.plot(label='d=1',figsize=(13,5), c='blue',lw=1,alpha=.7)
    ts2.plot(label='d=2',figsize=(13,5), c='red',lw=1.2,alpha=.8)
    ts3.plot(label='d=3',figsize=(13,5), c='magenta',lw=1,alpha=.7)
    ts4.plot(label='d=4',figsize=(13,5), c='green',lw=1,alpha=.7)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=True, 
               fancybox=True, facecolor='lightgray')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    #**************#
    
    # Plot ACF, PACF
    fig4,axes = plt.subplots(nrows=2, ncols=2, figsize=(21,13))
    plot_acf(ts1,ax=axes[0,0],lags=lags)
    plot_pacf(ts1, ax=axes[0,1],lags=lags)
    plot_acf(ts2,ax=axes[1,0],lags=lags)
    plot_pacf(ts2, ax=axes[1,1],lags=lags)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # plot rolling mean and std
    #Determine rolling statistics
    rolmean = ts.rolling(window=12, center=False).mean()
    rolstd = ts.rolling(window=12, center=False).std()
        
    #Plot rolling statistics
    fig = plt.figure(figsize=(13,5))
    orig = plt.plot(ts, color='red', label='original')
    mean = plt.plot(rolmean, color='cyan', label='rolling mean')
    std = plt.plot(rolstd, color='orange', label='rolling std')
    
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left") 
    plt.title('Rolling mean and standard deviation')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
    
    #**************#
    # # Check Seasonality 
    """
    Calculates and plots Seasonal Decomposition for a time series
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomp = seasonal_decompose(ts, model='additive') # model='multiplicative'

    decomp.plot()
    ts_seas = decomp.seasonal

    ax = ts_seas.plot(c='green')
    fig = ax.get_figure()
    fig.set_size_inches(13,11)

    ## Get min and max idx
    min_ = ts_seas.idxmin()
    max_ = ts_seas.idxmax()
    min_2 = ts_seas.loc[max_:].idxmin()

    ax.axvline(min_,label=min_,c='red')
    ax.axvline(max_,c='red',ls=':', lw=2)
    ax.axvline(min_2,c='red', lw=2)

    period = min_2 - min_
    ax.set_title(f'Season Length = {period}')

    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show();
   
    #*******#
    clock = pd.DataFrame.from_dict(output, orient='index')
    print(' /\\   '*3+' /')
    print('/ CLOCK-TIME STATS')
    print('    \/'*3)
    
    #display results
    print('---'*9)
    return clock

"""

>>>>>>>>>>>>>>>>>> Machine Learning MODELS <<<<<<<<<<<<<<<<<<<<<<

* ttXsplit()

"""
#### ----> ttXsplit()
def ttXsplit(tx, tSIZE, tMIN):
    """
    Performs a train-test split on timeseries data
    # train, test = ttXsplit(ts, 0.2, 2)
    """
    # idXsplit
    import math
    idx_split = math.floor(len(tx.index)*(1-tSIZE))
    
    n = len(tx.iloc[idx_split:]) 
    if n < tMIN:
        idx_split = (len(tx) - tMIN)

    train = tx.iloc[:idx_split]
    test = tx.iloc[idx_split:]
    print(f'train: {len(train)} | test: {len(test)}')
    
    return train, test


def mind_your_PDQs(P=range(0,3), D=range(1,3), Q=range(0,3), s=None):
    """

    pdqs = mind_your_PDQs()
    pdqs['pdq']

    pdq = pdqs['pdq']
    """
    import itertools
    pdqs = {}
    
    if s is None:
        pdqs['pdq'] = list(itertools.product(P,D,Q))
    else:
        pdqs['PDQs'] = list(itertools.product(P,D,Q,s))
    return pdqs


def stopwatch(time='time'): 
    """
    # stopwatch('stop')

    """
    import datetime as dt
    import tzlocal as tz
    if time == 'now':
        now = dt.datetime.now(tz=tz.get_localzone())
        print(now)
    if time=='start':
        now = dt.datetime.now(tz=tz.get_localzone())
        start = now.strftime('%m/%d/%Y - %I:%M:%S %p')
        print('start:', start)
        
    elif time == 'stop':
        now = dt.datetime.now(tz=tz.get_localzone())
        stop = now.strftime('%m/%d/%Y - %I:%M:%S %p')
        print('stop:', stop)
    
    elif time == 'time':
        now = dt.datetime.now(tz=tz.get_localzone())
        time = now.strftime('%m/%d/%Y - %I:%M:%S %p')
        print(time,'|', now)
        
    return time


# Run a grid with pdq and seasonal pdq parameters calculated above and get the best AIC value

def gridMAX(ts, pdq, PDQM=None, verbose=False):
    """
    Runs a gridsearch with pdq and seasonal pdq parameters to get the best AIC value
    Returns grid and best params

    Ex:
    gridX, best_params = gridMAX(ts,pdq=pdq)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import statsmodels.api as sm   
    stopwatch('start')

    print(f'[*] STARTING GRID SEARCH')
    
    # store to df_res
    grid = [['pdq','PDQM','AIC']]
    
    for comb in pdq:
        if PDQM is None:
            PDQM=[(0, 0, 0, 0)]
        for combs in PDQM:
            mod = sm.tsa.statespace.SARIMAX(ts,
                                            order=comb,
                                            seasonal_order=combs,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            output = mod.fit()

            grid.append([comb, combs, output.aic])
            if verbose:
                print('ARIMA {} x {}12 : AIC Calculated ={}'.format(comb, 
                                                                    combs, 
                                                                   output.aic))           
    
    stopwatch('stop')
    print(f"[**] GRID SEARCH COMPLETE")
    gridX = pd.DataFrame(grid[1:], columns=grid[0])
    gridX = gridX.sort_values('AIC').reset_index()
    best_params = dict(order=gridX.iloc[0].loc['pdq'])
    best_pdq = gridX.iloc[0][1]
    best_pdqm = gridX.iloc[0][2]
    display(gridX, best_params)
    return gridX, best_params      

def calcROI(investment, final_value):
    """This function takes in a series of forecasts to predict the return
    on investment spanning the entire forecast.

    r = calcROI(investment, final_value)
    """
    r = np.round(((final_value - investment) / investment)*100,3)
    return r


# From James Irving (Bootcamp) https://github.com/jirvingphd/fsds/blob/master/fsds/jmi/jmi.py
def thiels_U(ys_true=None, ys_pred=None,display_equation=True,display_table=True):
    """Calculate's Thiel's U metric for forecasting accuracy.
    Accepts true values and predicted values.
    Returns Thiel's U"""


    from IPython.display import Markdown, Latex, display
    import numpy as np
    display(Markdown(""))
    eqn=" $$U = \\sqrt{\\frac{ \\sum_{t=1 }^{n-1}\\left(\\frac{\\bar{Y}_{t+1} - Y_{t+1}}{Y_t}\\right)^2}{\\sum_{t=1 }^{n-1}\\left(\\frac{Y_{t+1} - Y_{t}}{Y_t}\\right)^2}}$$"

    # url="['Explanation'](https://docs.oracle.com/cd/E57185_01/CBREG/ch06s02s03s04.html)"
    markdown_explanation ="|Thiel's U Value | Interpretation |\n\
    | --- | --- |\n\
    | <1 | Forecasting is better than guessing| \n\
    | 1 | Forecasting is about as good as guessing| \n\
    |>1 | Forecasting is worse than guessing| \n"


    if display_equation and display_table:
        display(Latex(eqn),Markdown(markdown_explanation))#, Latex(eqn))
    elif display_equation:
        display(Latex(eqn))
    elif display_table:
        display(Markdown(markdown_explanation))

    if ys_true is None and ys_pred is None:
        return

    # sum_list = []
    num_list=[]
    denom_list=[]
    for t in range(len(ys_true)-1):
        num_exp = (ys_pred[t+1] - ys_true[t+1])/ys_true[t]
        num_list.append([num_exp**2])
        denom_exp = (ys_true[t+1] - ys_true[t])/ys_true[t]
        denom_list.append([denom_exp**2])
    U = np.sqrt( np.sum(num_list) / np.sum(denom_list))
    return U

# From James Irving
def model_evaluation(ts_true,ts_pred,show=True,show_u_info=False):
    import fsds_100719 as fs
    from sklearn.metrics import mean_squared_error,r2_score

    res= [['Metric','Value']]
    
    res.append(['RMSE', np.sqrt(mean_squared_error(ts_true,ts_pred))])
    
    res.append(['R2',r2_score(ts_true,ts_pred)])
    res.append(["Thiel's U", thiels_U(ts_true,ts_pred,
                                            display_equation=show_u_info,
                                           display_table=show_u_info)])
    res = fs.list2df(res)
    
    if show:
        display(res)
    return res

#ts = NYC[zc]['MeanValue'].rename(zc)
def forecastX(model_output, train, test, start=None, end=None, get_metrics=False):
    """
    Runs a forecast model using training data to make predictions for the test data.

    Uses get_prediction=() and conf_int() methods from statsmodels 
        get_prediction (exog,transform,weightsrow_labels,pred_kwds)
    """
    if start is None:
        start = test.index[0]     
    if end is None:
        end = test.index[-1]    
        
    # Get predictions starting from 2013 and calculate confidence intervals.
    prediction = model_output.get_prediction(start=start,end=end, dynamic=True)
    
    forecast = prediction.conf_int()
    forecast['predicted_mean'] = prediction.predicted_mean
    fc_plot = pd.concat([forecast, train], axis=1)

    ## Get ROI Forecast:
    r = calcROI(investment=forecast['predicted_mean'].iloc[0], 
                final_value=forecast['predicted_mean'].iloc[-1])

    zc = train.name

    fig, ax = plt.subplots(figsize=(21,13))
    train.plot(ax=ax,label='Training Data',lw=4) # '1996-04-01, # 2013-11-01
    test.plot(ax=ax,label='Test Data',lw=4) # '2013-12-01 , '2018-04-01
    
    forecast['predicted_mean'].plot(ax=ax, label='Forecast', color='magenta',lw=4)

    ax.fill_between(forecast.index, 
                    forecast.iloc[:,0], 
                    forecast.iloc[:,1],
                    color="white", 
                    alpha=.5, 
                    label = 'conf_int')
    
    ax.fill_betweenx(ax.get_ylim(), test.index[0], test.index[-1], color='darkslategray',alpha=0.5, zorder=-1)
    ax.fill_betweenx(ax.get_ylim(), start, end, color='darkslategray',zorder=-1)
    
    ax.legend(loc="upper left",bbox_to_anchor=(1.04,1), ncol=2,fontsize='small',frameon=True, fancybox=True, framealpha=.15, facecolor='k')
    ax.set(title=f"Predictions for {zc}: ROI = {r}%")
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Home Value $USD')
    fig = ax.get_figure()
    fc_plot['zipcode']= train.name
    plt.show()
    
    if get_metrics == True:
        metrics = model_evaluation(ts_true=test, ts_pred=forecast['predicted_mean'])

    return r, forecast, fig, ax

# r,forecast, fig, ax = forecastX(model_output, train, test, get_metrics=True)
# forecast
# r



def gridMAXmeta(KEYS, s=False):
    """
    Makes a forecast prediction based on combined train + test sets of a trained model 

    Opt1: gridMAXmeta(KEYS=NYC, s=False)
    
    KEYS: dict of dfs or timeseries
    NOTE: if passing in dict of full dataframes, s=True
    (gridMAXmeta will create dict of ts for you)
    
    Opt2: gridMAXmeta(KEYS=txd, s=True)
    KEYS: dictionary of ts - skip the meta ts creation
    """
    import statsmodels.api as sm
    import statsmodels.stats.api as sms
    import statsmodels.formula.api as smf
    #import statsmodels.formula.api as ols
    import statsmodels.stats.multicomp
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    # SCIPY
    import scipy.stats as stats
    from scipy.stats import normaltest as normtest # D'Agostino and Pearson's omnibus test
    from collections import Counter

    # SKLEARN
    from sklearn.metrics import mean_squared_error as mse
    ########

    if s is False:
        # loop thru each zipcode to create timeseries from its df in KEYS dfdict
        txd = {}
        for i,zc in enumerate(KEYS):
            # store each zipcode as ts  
            ts = KEYS[zc]['MeanValue'].rename(zc)
            txd[zc] = ts
    else:
        txd = KEYS
        
    pdqs = mind_your_PDQs()
    metagrid = {} 
    ROI = {}

    for zc,ts in txd.items():
        print('\n')
        print('---'*30)
        print('---'*30)
        print(f'ZIPCODE: {zc}')
        ## Train test split
        train, test = ttXsplit(ts, 0.1, 2)


        ## gridMAX gridsearch
        ###### TEST DATA ####
        gridX, best_params = gridMAX(train, pdq=pdqs['pdq'])
        metagrid[zc]={}
        metagrid[zc]['gridX']= gridX.iloc[0]
        metagrid[zc]['pdq'] = best
        metagrid[zc]['aic'] = gridX.iloc[0][3]
        
        ## Using best params
        best_params
        
        ##### SARIMAX: USING ENTIRE TIME SERIES ###
        model_output = SARIMAX(ts,
                               **best_params,
                               enforce_invertibility=False,
                               enforce_stationarity=False).fit()

        metagrid[zc]['model'] = model_output

        r, forecast,fig,ax = forecastX(model_output,
                                       train, test, 
                                       start=ts.index[-1],
                                       end=ts.index.shift(24)[-1],
                                       get_metrics=False)
        metagrid[zc]['forecast'] = forecast
        ROI[zc] = r
        metagrid[zc]['ROI'] = r
        ROI[zc] = r
        
    return metagrid, ROI

#metagrid, ROI = gridMAXmeta(KEYS=NYC, s=False)



# def forecastX_plotly(model_output, train, test, start=None, end=None, get_metrics=False):
#     """
#     Uses get_prediction=() and conf_int() methods from statsmodels 
#         get_prediction (exog,transform,weightsrow_labels,pred_kwds)
#     """

#     if start is None:
#         start = test.index[0]     
#     if end is None:
#         end = test.index[-1]    
        
#     # Get predictions starting from 2013 and calculate confidence intervals.
#     prediction = model_output.get_prediction(start=start,end=end, dynamic=True)
    
#     forecast = prediction.conf_int()
#     forecast['predicted_mean'] = prediction.predicted_mean
#     fc_plot = pd.concat([forecast, train], axis=1)

#     ## Get ROI Forecast:
#     r = calcROI(investment=forecast['predicted_mean'].iloc[0], 
#                 final_value=forecast['predicted_mean'].iloc[-1])

#     zc = train.name

#     fig=go.Figure()
#     #fig, ax = plt.subplots(figsize=(21,13))
#     fig.add_trace(go.Line(x=train.index, y=train.MeanValue))
#     fig.add_trace(go.Line(x=test.index, y=test.MeanValue))
#     fig.add_trace(go.Line(x))




      #train.plot(ax=ax,label='Training Data',lw=4) # train.index[0] '1996-04-01, train.index[-1] # 2013-11-01
    #test.plot(ax=ax,label='Test Data',lw=4) # test.index[0] '2013-12-01 , test.index[-1] '2018-04-01
    
    #forecast['predicted_mean'].plot(ax=ax, label='Forecast', color='magenta',lw=4)


# fig1.add_trace(go.Scatter(x=df.index, y=df['MeanValue'], name="Mean Home Value",line_color='crimson'))
# fig1.add_trace(go.Scatter(x=FC.index, y=FC['pred_mean'], name="Forecast Value",line_color='deepskyblue'))
# fig1.add_trace(go.Scatter(x=NY_Hudson['DateTime'], y=NY_Hudson['MeanValue'], name="Hudson MeanValue",
#                          line_color='lightgreen'))
# fig1.update_layout(title_text='MeanValues by Train Line',
#                   xaxis_rangeslider_visible=True)

    
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
    #train.plot(ax=ax,label='Training Data',lw=4) # train.index[0] '1996-04-01, train.index[-1] # 2013-11-01
    #test.plot(ax=ax,label='Test Data',lw=4) # test.index[0] '2013-12-01 , test.index[-1] '2018-04-01
    
    #forecast['predicted_mean'].plot(ax=ax, label='Forecast', color='magenta',lw=4)

    #ax.fill_between(forecast.index, 
    #                forecast.iloc[:,0], 
    #                forecast.iloc[:,1],
    #                color="white", 
    #                alpha=.5, 
    #                label = 'conf_int')
    
    #ax.fill_betweenx(ax.get_ylim(), test.index[0], test.index[-1], color='darkslategray',alpha=0.5, zorder=-1)
    #ax.fill_betweenx(ax.get_ylim(), start, end, color='darkslategray',zorder=-1)
    
    #ax.legend(loc="upper left",bbox_to_anchor=(1.04,1), ncol=2,fontsize='small',frameon=True, fancybox=True, framealpha=.15, facecolor='k')
    #ax.set(title=f"Predictions for {zc}: ROI = {r}%")
    #ax.set_xlabel('Year')
    #ax.set_ylabel('Mean Home Value $USD')
    #fig = ax.get_figure()
    fc_plot['zipcode']= train.name
    #plt.show()
    
    #if get_metrics == True:
    #   metrics = spak.model_evaluation(ts_true=test, ts_pred=forecast['predicted_mean'])

    return r, forecast, fig

#forecast, fig, ax = forecastX(model_output, train, test)