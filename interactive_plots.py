import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import holoviews as hv
from holoviews import opts
hv.extension('bokeh', 'matplotlib')
import hvplot.pandas

def preprocess(filename):
    """Preprocess the csv data and return a pd.DataFrame.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value

    """

    assert isinstance(filename, str), "dateset link should be a string"

    df = pd.read_csv(filename, header=0, index_col=0)
    # TODO: group numbers like year and price into percentiles
    df['state'] = df['state'].str.upper()

    df['posting_date'] = pd.to_datetime(df['posting_date'])
    df['price_range'] = pd.cut(df['price'],
                               [-np.inf, 5000, 15000, 25000, np.inf],
                               labels=['<$5000', '$5000-15000', '$15000-25000', '>$25000']
                               )
    df['year_range'] = pd.cut(df['year'],
                               [-np.inf, 2005, 2010, 2015, np.inf],
                               labels=['<2005', '2005-2010', '2010-2015', '>2015']
                               )
    df['odometer_range'] = pd.cut(df['odometer'],
                              [-np.inf, 40000, 80000, 120000, np.inf],
                              labels=['<40000', '40000-80000', '80000-120000', '>120000']
                              )
    df.to_csv('data/cyx_processed.csv')
    return df

def interactive_plots_preprocess(df):
    """Specific preprocessing for interactive plots.

    Choose the data rows with the year after 2000 and price not equal to 0. Drop some irrelevant cols.
    Convert the pd.DataFrame to hv.Dataset

    Args:
        arg1 (pd.DataFrame): Takes in the pd.DataFrame of the dataset

    Returns:
        pd.DataFrame: The pd.DataFrame of the dataset after preprocessing
        hv.Dataset: The hv.Dataset of the dataset after preprocessing

    """
    assert isinstance(df, pd.DataFrame), "input should be a pandas DataFrame"

    df = df.dropna().drop(columns=['lat', 'url', 'region_url', 'long', 'posting_date', 'VIN'])
    df = df[df['year'] >= 2000]
    df = df[df['price'] != 0]
    df['year'] = df['year'].astype(int)

    vdims = ['price']
    kdims = ['year', 'state', 'manufacturer', 'model', 'condition', 'odometer', 'fuel', 'size', 
    'type', 'cylinders', 'drive', 'transmission', 'paint_color', 'price_range','year_range', 'odometer_range']
    edata = hv.Dataset(data=df,kdims=kdims, vdims=vdims)

    return df, edata

def price_trendency_plot(edata, kdim):
    """Holoview curve plot combined with errorbar plot of average price trendency.

    Aggregate the data given the k-dimension and generate the errorbar plot and curve plot.

    Args:
        arg1 (hv.Dataset): The hv.Dataset of the dataset
        arg2 (str): The key-dimension

    Returns:
        hv.core.overlay.Overlay: The holoview curve plot combined with errorbar plot

    """
    assert isinstance(kdim, str), "Please input your interesting key dimension as string"

    agg = edata.aggregate(kdim, function=np.mean, spreadfn=np.std).sort()
    errorbars = hv.ErrorBars(agg,vdims=['price', 'price_std']).iloc[::1]
    overlay =  (hv.Curve(agg) * errorbars).redim.range(price=(0, None))


    if len(agg) >= 20:
        overlay.opts(width=800)
    elif 10 <= len(agg) < 20:
        overlay.opts(width=600)
    else:
        overlay.opts(width=400)
    overlay.opts(xrotation=45)

    return overlay


def count_plot(df, kdim):
    """Holoview count bars plot of the k-dimension.

    Groupby the k-dimension, get the counts and sort them.
    Generate the Holoviews bars plot using pandas.hvplot.

    Args:
        arg1 (pd.DataFrame): The DataFrame of the dataset
        arg2 (str): The key-dimension

    Returns:
        hv.element.chart.Bars: The holoview bars plot.

    """
    assert isinstance(kdim, str), "Please input your interesting key dimension as string"

    series = df.groupby([kdim]).size().rename('count').sort_values()

    if len(series) >= 20:
        height = 600
    else:
        height = 400


    plot = series.hvplot.barh().options(height=height)

    return plot
    
def price_tredency_plot_given(edata, kdim):
    """Holoview curve plot combined with pulldown widget of average price trendency of year given the k-dimension.

    Aggregate the data given the k-dimension and 'year'. Generate the curve plot with the pulldown widget.

    Args:
        arg1 (hv.Dataset): The hv.Dataset of the dataset
        arg2 (str): The key-dimension

    Returns:
        hv.core.spaces.HoloMap: The holoview curve plot with the pulldown widget

    """
    assert isinstance(kdim, str), "Please input your interesting key dimension as string"

    agg = edata.aggregate(['year', kdim], function=np.mean).sort()
    plot = agg.to(hv.Curve, 'year', 'price', groupby=kdim).options(width=600,height=300)

    return plot
